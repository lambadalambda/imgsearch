#!/usr/bin/env python3
"""Offline regressions for the Fediverse importer (stdlib only)."""

from datetime import datetime, timedelta, timezone
from email.message import Message
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import import_fediverse as importer

NOW = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
CUTOFF = NOW - timedelta(days=7)
PROFILE = "https://social.example/users/alice"
LOOKUP = "https://social.example/api/v1/accounts/lookup?acct=alice"
TIMELINE = "https://social.example/api/v1/accounts/local-id/statuses?limit=40&only_media=true&exclude_reblogs=true"


def headers(**values):
    result = Message()
    for key, value in values.items():
        result[key.replace("_", "-")] = value
    return result


def post(id, date=NOW, attachments=None, **extra):
    return {
        "id": id,
        "created_at": date.isoformat(),
        "account": {"id": "local-id"},
        "reblog": None,
        "media_attachments": attachments or [],
        **extra,
    }


def media(url, kind="image", **extra):
    return {
        "type": kind,
        "url": url,
        "preview_url": "https://cdn.example/preview.jpg",
        **extra,
    }


class FakeFetcher:
    def __init__(self, pages):
        self.pages = {LOOKUP: ({"id": "local-id"}, headers()), **pages}
        self.calls = []

    def json(self, url):
        self.calls.append(url)
        return self.pages[url]

    def download(self, url, path, max_bytes):
        raise AssertionError("dry-run must not download media")


class DiscoveryTests(unittest.TestCase):
    def test_profile_urls(self):
        for url in (PROFILE, "https://social.example/@alice", PROFILE + "/"):
            self.assertEqual(
                importer.parse_profile(url), ("https://social.example", "alice")
            )
        for url in (
            "file:///users/alice",
            "https://social.example/@alice/123",
            "https://u:pw@social.example/@alice",
            "https://social.example/",
        ):
            with self.assertRaises(ValueError):
                importer.parse_profile(url)

    def test_pagination_cutoff_boosts_pins_and_original_urls(self):
        next_url = TIMELINE + "&max_id=opaque-2"
        fetcher = FakeFetcher(
            {
                TIMELINE: (
                    [
                        post(
                            "old-pin",
                            CUTOFF - timedelta(days=20),
                            [media("https://cdn.example/pin.jpg")],
                            pinned=True,
                        ),
                        post(
                            "new",
                            NOW - timedelta(hours=1),
                            [
                                media("https://cdn.example/full.jpg"),
                                media("https://cdn.example/movie.mp4", "video"),
                            ],
                        ),
                        post(
                            "boost",
                            attachments=[media("https://cdn.example/boost.jpg")],
                            reblog={"id": "other"},
                        ),
                        post(
                            "other",
                            attachments=[media("https://cdn.example/other.jpg")],
                            account={"id": "another"},
                        ),
                        post(
                            "opaque-2",
                            attachments=[
                                media("https://cdn.example/song.mp3", "audio")
                            ],
                        ),
                    ],
                    headers(Link=f'<{next_url}>; rel="next"'),
                ),
                next_url: (
                    [
                        post(
                            "boundary",
                            CUTOFF,
                            [
                                media("https://cdn.example/full.jpg"),
                                media("https://cdn.example/animation.mp4", "gifv"),
                            ],
                        ),
                        post(
                            "old",
                            CUTOFF - timedelta(seconds=1),
                            [media("https://cdn.example/old.jpg")],
                        ),
                    ],
                    headers(),
                ),
                next_url.replace("opaque-2", "old"): (
                    [post("older", CUTOFF - timedelta(days=1))],
                    headers(Link=f'<{TIMELINE}&max_id=older>; rel="next"'),
                ),
            }
        )
        found = list(importer.iter_media(PROFILE, CUTOFF, NOW, fetcher))
        self.assertEqual(
            [item.url for item in found],
            [
                "https://cdn.example/full.jpg",
                "https://cdn.example/movie.mp4",
                "https://cdn.example/animation.mp4",
            ],
        )
        self.assertEqual(len(fetcher.calls), 4)

    def test_no_link_falls_back_to_opaque_max_id(self):
        next_url = TIMELINE + "&max_id=not-a-number"
        fetcher = FakeFetcher(
            {
                TIMELINE: (
                    [
                        post(
                            "not-a-number",
                            attachments=[media("https://cdn.example/a.png")],
                        )
                    ],
                    headers(),
                ),
                next_url: ([], headers()),
            }
        )
        self.assertEqual(
            len(list(importer.iter_media(PROFILE, CUTOFF, NOW, fetcher))), 1
        )
        self.assertEqual(fetcher.calls[-1], next_url)

    def test_repeated_pages_fail_instead_of_looping(self):
        statuses = [post("same")]
        fetcher = FakeFetcher(
            {
                TIMELINE: (statuses, headers()),
                TIMELINE + "&max_id=same": (statuses, headers()),
            }
        )
        with self.assertRaisesRegex(ValueError, "progress|repeat"):
            list(importer.iter_media(PROFILE, CUTOFF, NOW, fetcher))

    def test_cross_origin_next_link_is_rejected(self):
        fetcher = FakeFetcher(
            {
                TIMELINE: (
                    [post("1")],
                    headers(Link='<https://evil.example/api>; rel="next"'),
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "pagination"):
            list(importer.iter_media(PROFILE, CUTOFF, NOW, fetcher))

    def test_bad_timestamps_or_missing_original_fail_loudly(self):
        for status in (
            post("1", created_at="not-a-time"),
            post("1", attachments=[media(None)]),
        ):
            fetcher = FakeFetcher({TIMELINE: ([status], headers())})
            with self.assertRaises(ValueError):
                list(importer.iter_media(PROFILE, CUTOFF, NOW, fetcher))

    def test_dry_run_does_not_download_or_upload(self):
        fetcher = FakeFetcher(
            {
                TIMELINE: (
                    [post("1", attachments=[media("https://cdn.example/a.jpg")])],
                    headers(),
                ),
                TIMELINE + "&max_id=1": ([], headers()),
            }
        )

        def forbidden(*args):
            self.fail("dry-run uploaded data")

        output = io.StringIO()
        result = importer.run(
            PROFILE,
            7,
            "http://imgsearch.invalid",
            dry_run=True,
            fetcher=fetcher,
            upload=forbidden,
            now=NOW,
            out=output,
        )
        self.assertEqual(result, 0)
        self.assertIn("https://cdn.example/a.jpg", output.getvalue())
        self.assertIn("selected=1", output.getvalue())

    def test_individual_failures_continue_and_return_nonzero(self):
        urls = [
            "https://cdn.example/bad.jpg",
            "https://cdn.example/good",
            "https://cdn.example/duplicate.jpg",
        ]
        fetcher = FakeFetcher(
            {
                TIMELINE: (
                    [post("1", attachments=[media(url) for url in urls])],
                    headers(),
                ),
                TIMELINE + "&max_id=1": ([], headers()),
            }
        )
        downloaded = []

        def download(url, path, max_bytes):
            if url == urls[0]:
                raise OSError("download failed")
            Path(path).write_bytes(b"image")
            return headers(Content_Type="image/jpeg")

        fetcher.download = download

        def upload(path, destination, max_video_bytes):
            self.assertEqual(Path(path).suffix, ".jpg")
            self.assertEqual(destination, "http://imgsearch.invalid")
            downloaded.append(Path(path))
            return (1, 0) if len(downloaded) == 1 else (0, 1)

        output = io.StringIO()
        result = importer.run(
            PROFILE,
            7,
            "http://imgsearch.invalid",
            fetcher=fetcher,
            upload=upload,
            now=NOW,
            out=output,
        )
        self.assertEqual(result, 1)
        self.assertIn("created=1 duplicates=1 failed=1", output.getvalue())
        self.assertTrue(all(not path.exists() for path in downloaded))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.attempts = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append((self.path, dict(self.headers)))
                if self.path in ("/limited", "/backoff"):
                    self.send_response(429 if self.path == "/limited" else 503)
                    self.send_header(
                        "Retry-After", "0" if self.path == "/limited" else "301"
                    )
                    self.end_headers()
                    return
                if self.path == "/retry":
                    owner.attempts += 1
                    if owner.attempts == 1:
                        self.send_response(429)
                        self.send_header("Retry-After", "0")
                        self.end_headers()
                        return
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/file")
                    self.end_headers()
                    return
                if self.path == "/truncated":
                    self.send_response(200)
                    self.send_header("Content-Length", "100")
                    self.end_headers()
                    self.wfile.write(b"xx")
                    self.close_connection = True
                    return
                if self.path.startswith("/api/v1/accounts/lookup?"):
                    payload = json.dumps({"id": "local-id"}).encode()
                elif self.path.startswith("/api/v1/accounts/local-id/statuses?"):
                    statuses = (
                        []
                        if "max_id=" in self.path
                        else [
                            post(
                                "newest",
                                datetime.now(timezone.utc) - timedelta(seconds=1),
                                [media(owner.base + "/file")],
                            )
                        ]
                    )
                    payload = json.dumps(statuses).encode()
                else:
                    payload = (
                        b"x" * 100
                        if self.path in ("/file", "/stream")
                        else b'{"ok":true}'
                    )
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    (
                        "image/png"
                        if self.path in ("/file", "/stream")
                        else "application/json"
                    ),
                )
                if self.path != "/stream":
                    self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                owner.requests.append((self.path, dict(self.headers)))
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                owner.upload_body = body
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"created":1,"duplicates":0,"failed":0}')

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_port
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_retry_and_no_imgsearch_credentials_on_any_get(self):
        sleeps = []
        fetcher = importer.Fetcher(delay=0, sleep=sleeps.append)
        with patch.dict(
            os.environ,
            {
                "IMGSEARCH_IMPORT_API_KEY": "private-test-key",
                "IMGSEARCH_API_KEY": "other-test-key",
            },
        ):
            result, _ = fetcher.json(self.base + "/retry")
            with tempfile.TemporaryDirectory() as tmp:
                fetcher.download(self.base + "/redirect", Path(tmp) / "media", 1024)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(self.attempts, 2)
        for _, values in self.requests:
            self.assertFalse(
                any(
                    key.lower() in ("authorization", "x-imgsearch-api-key", "cookie")
                    for key in values
                )
            )
            self.assertNotIn("private-test-key", str(values))

    def test_cli_uploads_through_real_helper_and_keeps_credentials_separate(self):
        env = dict(
            os.environ,
            IMGSEARCH_IMPORT_API_KEY="destination-only-test-key",
            IMGSEARCH_IMPORT_CONVERT="never",
        )
        script = Path(__file__).resolve().with_name("import_fediverse.py")
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    self.base + "/users/alice",
                    self.base,
                    "--days",
                    "7",
                    "--delay",
                    "0",
                ],
                env=env,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("selected=1 created=1 duplicates=0 failed=0", result.stdout)
        self.assertIn(b"x" * 100, self.upload_body)
        self.assertNotIn("destination-only-test-key", result.stdout + result.stderr)
        for path, values in self.requests:
            if path in ("/healthz", "/api/upload"):
                self.assertEqual(
                    values.get("X-Imgsearch-API-Key"), "destination-only-test-key"
                )
            else:
                self.assertNotIn("X-Imgsearch-API-Key", values)

    def test_source_backoff_stops_before_next_attachment(self):
        for endpoint, attempts in (("/limited", 3), ("/backoff", 1)):
            with self.subTest(endpoint=endpoint):
                self.requests.clear()
                pauses = []
                fetcher = FakeFetcher(
                    {
                        TIMELINE: (
                            [
                                post(
                                    "1",
                                    attachments=[
                                        media(self.base + endpoint),
                                        media(self.base + "/file"),
                                    ],
                                )
                            ],
                            headers(),
                        ),
                        TIMELINE + "&max_id=1": ([], headers()),
                    }
                )
                fetcher.download = importer.Fetcher(
                    delay=0, sleep=pauses.append
                ).download
                output = io.StringIO()
                result = importer.run(
                    PROFILE,
                    7,
                    "http://imgsearch.invalid",
                    fetcher=fetcher,
                    upload=lambda *args: (1, 0),
                    now=NOW,
                    out=output,
                )
                self.assertEqual(result, 1)
                self.assertIn(
                    "selected=1 created=0 duplicates=0 failed=1", output.getvalue()
                )
                self.assertEqual(
                    [path for path, _ in self.requests], [endpoint] * attempts
                )
                self.assertNotIn(301, pauses)

    def test_truncated_download_retries_then_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "network|incomplete"):
                importer.Fetcher(delay=0, sleep=lambda _: None).download(
                    self.base + "/truncated", Path(tmp) / "media", 1024
                )
        self.assertEqual(sum(path == "/truncated" for path, _ in self.requests), 3)

    def test_download_limit_with_and_without_content_length(self):
        for path in ("/file", "/stream"):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "limit|large"):
                    importer.Fetcher(delay=0).download(
                        self.base + path, Path(tmp) / "media", 10
                    )


class UploadTests(unittest.TestCase):
    def test_calls_existing_importer_and_checks_actual_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "image.jpg"
            source.write_bytes(b"image")
            responses = [
                (
                    0,
                    "Import summary: total=1 created=1 duplicates=0 converted=0 failed=0",
                    (1, 0),
                ),
                (
                    0,
                    "Import summary: total=1 created=0 duplicates=1 converted=0 failed=0",
                    (0, 1),
                ),
                (
                    0,
                    "Import summary: total=1 created=0 duplicates=0 converted=0 failed=0",
                    None,
                ),
                (
                    1,
                    "Import summary: total=1 created=0 duplicates=0 converted=0 failed=1",
                    None,
                ),
            ]
            for code, output, want in responses:
                with patch.object(
                    importer.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], code, output),
                ) as call:
                    if want is None:
                        with self.assertRaises(RuntimeError):
                            importer.upload_file(source, "http://localhost:8080", 2000)
                    else:
                        self.assertEqual(
                            importer.upload_file(source, "http://localhost:8080", 2000),
                            want,
                        )
                    self.assertIn("import_images.sh", str(call.call_args.args[0]))
                    self.assertEqual(
                        call.call_args.kwargs["env"][
                            "IMGSEARCH_IMPORT_MAX_VIDEO_BYTES"
                        ],
                        "2000",
                    )


if __name__ == "__main__":
    unittest.main()
