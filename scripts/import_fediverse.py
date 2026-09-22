#!/usr/bin/env python3
"""Import public image/video attachments from a Mastodon-compatible account.

Python 3.9+ stdlib only. Uploads use the existing import_images.sh helper,
including its GIF conversion and content-hash duplicate handling.
"""

import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

IMAGE_LIMIT = 64 * 1024 * 1024
VIDEO_LIMIT = 2 * 1024 * 1024 * 1024
EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".gif",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
}
MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/avif": ".avif",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}


def web_url(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("expected an HTTP(S) URL without embedded credentials")
    return parsed


def parse_profile(value):
    parsed = web_url(value)
    parts = unquote(parsed.path).strip("/").split("/")
    if len(parts) == 2 and parts[0] == "users":
        username = parts[1]
    elif len(parts) == 1 and parts[0].startswith("@"):
        username = parts[0][1:]
    else:
        raise ValueError(
            "profile URL must look like https://server/@user or https://server/users/user"
        )
    if not username or any(char.isspace() for char in username):
        raise ValueError("invalid account name")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")), username


class WebRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        web_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class SourceBackoffError(RuntimeError):
    """Stop the run rather than sending more requests to a throttled source."""


class Fetcher:
    """Unauthenticated HTTP client, intentionally separate from the uploader."""

    def __init__(self, delay=1.0, attempts=3, sleep=time.sleep):
        self.delay = delay
        self.attempts = attempts
        self.sleep = sleep
        self.opener = build_opener(WebRedirects())

    def _fetch(self, url, destination=None, max_bytes=8 * 1024 * 1024):
        web_url(url)
        for attempt in range(self.attempts):
            if self.delay:
                self.sleep(self.delay)
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": "imgsearch-fediverse-import/1.0",
                        "Accept": "*/*",
                    },
                )
                with self.opener.open(request, timeout=60) as response:
                    length = response.headers.get("Content-Length")
                    if length and int(length) > max_bytes:
                        raise ValueError(
                            "response exceeds download limit (%d bytes)" % max_bytes
                        )
                    target = open(destination, "wb") if destination else io.BytesIO()
                    with target:
                        total = 0
                        while True:
                            chunk = response.read(
                                min(1024 * 1024, max_bytes + 1 - total)
                            )
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes:
                                raise ValueError(
                                    "response exceeds download limit (%d bytes)"
                                    % max_bytes
                                )
                            target.write(chunk)
                        if length and total != int(length):
                            raise ConnectionError("incomplete response body")
                        payload = b"" if destination else target.getvalue()
                    return payload, response.headers
            except HTTPError as error:
                retry_after = error.headers.get("Retry-After")
                error.close()
                retryable = error.code in (429, 500, 502, 503, 504)
                wait = retry_delay(retry_after, attempt)
                if retryable and wait > 300:
                    raise SourceBackoffError(
                        "server requests a retry after %.0fs; retry this import later"
                        % wait
                    ) from error
                if error.code == 429 and attempt + 1 == self.attempts:
                    raise SourceBackoffError(
                        "HTTP 429 retries exhausted; retry this import later"
                    ) from error
                if not retryable or attempt + 1 == self.attempts:
                    raise RuntimeError(
                        "HTTP %d fetching %s (only public, unauthenticated access is supported)"
                        % (error.code, url)
                    ) from error
                self.sleep(wait)
            except (URLError, TimeoutError, ConnectionError, IncompleteRead) as error:
                if attempt + 1 == self.attempts:
                    raise RuntimeError(
                        "network error fetching %s: %s" % (url, error)
                    ) from error
                self.sleep(2**attempt)
        raise RuntimeError("request attempts exhausted")

    def json(self, url):
        body, headers = self._fetch(url)
        return json.loads(body), headers

    def download(self, url, path, max_bytes):
        _, headers = self._fetch(url, destination=path, max_bytes=max_bytes)
        return headers


def retry_delay(value, attempt):
    if value:
        try:
            return max(0, int(value))
        except ValueError:
            try:
                return max(
                    0,
                    (
                        parsedate_to_datetime(value) - datetime.now(timezone.utc)
                    ).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return 2**attempt


class Attachment(NamedTuple):
    url: str
    kind: str
    posted_at: datetime


def next_page(headers, current, last_id):
    current_url = web_url(current)
    for header in headers.get_all("Link", []):
        for match in re.finditer(r"<([^>]+)>([^<]*)", header):
            rel = re.search(r'\brel\s*=\s*"?([^";,]+)', match[2], re.I)
            if rel and "next" in rel[1].split():
                candidate = urljoin(current, match[1])
                parsed = web_url(candidate)
                if (parsed.scheme, parsed.netloc, parsed.path) != (
                    current_url.scheme,
                    current_url.netloc,
                    current_url.path,
                ):
                    raise ValueError("unsafe cross-origin/path pagination link")
                return candidate
    # Some compatible servers omit Link. IDs are opaque: never do ID arithmetic.
    params = dict(parse_qsl(current_url.query))
    params.pop("min_id", None)
    params["max_id"] = last_id
    return urlunsplit(current_url._replace(query=urlencode(params)))


def iter_media(profile, cutoff, until, fetcher):
    origin, username = parse_profile(profile)
    account, _ = fetcher.json(
        origin + "/api/v1/accounts/lookup?" + urlencode({"acct": username})
    )
    account_id = account.get("id") if isinstance(account, dict) else None
    if not isinstance(account_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", account_id
    ):
        raise ValueError("account lookup did not return a valid account ID")
    page_url = (
        origin
        + "/api/v1/accounts/%s/statuses?" % account_id
        + urlencode({"limit": 40, "only_media": "true", "exclude_reblogs": "true"})
    )
    seen_pages, seen_statuses, seen_urls = set(), set(), set()
    while page_url:
        if page_url in seen_pages:
            raise ValueError("pagination repeated a page")
        seen_pages.add(page_url)
        statuses, headers = fetcher.json(page_url)
        if not isinstance(statuses, list):
            raise ValueError("statuses endpoint did not return a list")
        if not statuses:
            return
        dates = []
        progressed = False
        for status in statuses:
            status_id = status.get("id")
            if not isinstance(status_id, str) or not status_id:
                raise ValueError("status has no ID")
            if status_id in seen_statuses:
                continue
            progressed = True
            seen_statuses.add(status_id)
            try:
                posted_at = datetime.fromisoformat(
                    status["created_at"].replace("Z", "+00:00")
                )
                if posted_at.tzinfo is None:
                    raise ValueError("missing timezone")
            except (KeyError, ValueError, TypeError, AttributeError) as error:
                raise ValueError(
                    "invalid created_at for status %s" % status_id
                ) from error
            if not status.get("pinned"):
                dates.append(posted_at)
            if (
                not cutoff <= posted_at <= until
                or status.get("reblog") is not None
                or status.get("account", {}).get("id") != account_id
            ):
                continue
            for attachment in status.get("media_attachments", []):
                kind = attachment.get("type")
                if kind not in ("image", "video", "gifv"):
                    continue
                url = attachment.get("url") or attachment.get("remote_url")
                if not isinstance(url, str) or not url:
                    raise ValueError(
                        "missing original attachment URL for status %s" % status_id
                    )
                web_url(url)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                yield Attachment(url, kind, posted_at)
        if not progressed:
            raise ValueError("pagination made no progress; refusing to loop")
        # Inspect the entire page: an old pinned post must not hide recent posts.
        # Account timelines are reverse chronological; one extra boundary page is
        # preferable to stopping early on a pinned/out-of-order item.
        if dates and all(date < cutoff for date in dates):
            return
        page_url = next_page(headers, page_url, statuses[-1]["id"])


def media_filename(url, headers):
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    ext = MIME_EXTENSIONS.get(headers.get_content_type(), Path(name).suffix.lower())
    if ext not in EXTENSIONS:
        raise ValueError(
            "attachment has no supported image/video extension or content type"
        )
    stem = (
        re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem).strip("._-")[:100] or "media"
    )
    return stem + ext


def upload_file(path, api_base_url, max_video_bytes):
    helper = Path(__file__).resolve().with_name("import_images.sh")
    env = dict(os.environ, IMGSEARCH_IMPORT_MAX_VIDEO_BYTES=str(max_video_bytes))
    result = subprocess.run(
        ["bash", str(helper), str(Path(path).parent), api_base_url],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    summary = re.search(
        r"Import summary: total=(\d+) created=(\d+) duplicates=(\d+) converted=(\d+) failed=(\d+)",
        result.stdout,
    )
    if result.returncode == 0 and summary:
        total, created, duplicates, _, failed = map(int, summary.groups())
        if total == 1 and created + duplicates == 1 and failed == 0:
            return created, duplicates
    # In particular, the old helper can exit successfully after skipping an
    # oversized video. Do not report that as an imported attachment.
    raise RuntimeError("upload/conversion failed or skipped: " + result.stdout.strip())


def run(
    profile,
    days,
    api_base_url,
    *,
    dry_run=False,
    fetcher=None,
    upload=upload_file,
    now=None,
    out=None
):
    out = out or sys.stdout
    now = now or datetime.now(timezone.utc)
    created = duplicates = failed = selected = 0
    try:
        parse_profile(profile)
        web_url(api_base_url)
        if days <= 0:
            raise ValueError("days must be positive")
        cutoff = now - timedelta(days=days)
        max_video_bytes = int(
            os.environ.get("IMGSEARCH_IMPORT_MAX_VIDEO_BYTES", VIDEO_LIMIT)
        )
        if max_video_bytes <= 0:
            raise ValueError("IMGSEARCH_IMPORT_MAX_VIDEO_BYTES must be positive")
        fetcher = fetcher or Fetcher()
        print(
            "Public media since %s (UTC); boosts/audio excluded." % cutoff.isoformat(),
            file=out,
        )
        for attachment in iter_media(profile, cutoff, now, fetcher):
            selected += 1
            print(
                "%s %s %s"
                % (
                    "WOULD IMPORT" if dry_run else "IMPORT",
                    attachment.posted_at.isoformat(),
                    attachment.url,
                ),
                file=out,
            )
            if dry_run:
                continue
            try:
                # One attachment at a time: temporary storage does not grow with
                # account history. Nothing is retained after upload or failure.
                with tempfile.TemporaryDirectory(prefix="imgsearch-fediverse-") as tmp:
                    partial = Path(tmp) / "download.part"
                    limit = (
                        IMAGE_LIMIT if attachment.kind == "image" else max_video_bytes
                    )
                    headers = fetcher.download(attachment.url, partial, limit)
                    path = Path(tmp) / media_filename(attachment.url, headers)
                    partial.rename(path)
                    added, existing = upload(path, api_base_url, max_video_bytes)
                    created += added
                    duplicates += existing
            except SourceBackoffError:
                raise
            except (OSError, ValueError, RuntimeError) as error:
                failed += 1
                print("FAIL %s: %s" % (attachment.url, error), file=out)
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        failed += 1
        print("FAIL import stopped: %s" % error, file=out)
    print(
        "%s summary: selected=%d created=%d duplicates=%d failed=%d"
        % ("Dry-run" if dry_run else "Import", selected, created, duplicates, failed),
        file=out,
    )
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "profile_url", help="https://server/users/user or https://server/@user"
    )
    parser.add_argument(
        "api_base_url",
        nargs="?",
        default=os.environ.get("IMGSEARCH_API_URL", "http://127.0.0.1:8080"),
    )
    parser.add_argument(
        "--days", type=int, required=True, help="rolling number of days to import (UTC)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list matching original URLs; no downloads or uploads",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="pause before each source HTTP request (seconds; default 1)",
    )
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")
    if not 0 <= args.delay <= 300:
        parser.error("--delay must be between 0 and 300 seconds")
    try:
        return run(
            args.profile_url,
            args.days,
            args.api_base_url,
            dry_run=args.dry_run,
            fetcher=Fetcher(delay=args.delay),
        )
    except KeyboardInterrupt:
        print("Import interrupted; temporary files cleaned up.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
