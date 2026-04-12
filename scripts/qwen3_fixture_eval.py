#!/usr/bin/env python3
"""Evaluate fixture retrieval expectations against a sidecar endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif")


def parse_expectations(contents: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    pattern = re.compile(r'^"(.+?)"\s*[:\-]\s*(.+)$')
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("clusters by"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        query = match.group(1).strip()
        names = [name.strip() for name in match.group(2).split(",") if name.strip()]
        if query and names:
            out.append((query, names))
    return out


def find_missing_expected_filenames(
    image_names: list[str], expectations: list[tuple[str, list[str]]]
) -> set[str]:
    image_name_set = set(image_names)
    missing: set[str] = set()
    for _, names in expectations:
        for name in names:
            if name not in image_name_set:
                missing.add(name)
    return missing


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0

    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv

    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def query_hit_count(scored: list[tuple[str, float]], expected_names: list[str]) -> int:
    expected_set = set(expected_names)
    topk = [name for name, _ in scored[: len(expected_names)]]
    return sum(1 for name in topk if name in expected_set)


def query_passes(hit_count: int, expected_count: int, strict: bool) -> bool:
    if expected_count <= 0:
        return False
    if strict:
        return hit_count == expected_count
    return hit_count >= max(1, expected_count - 1)


def post_json(base_url: str, path: str, payload: dict, timeout_seconds: int) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_ready(base_url: str, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    url = base_url.rstrip("/") + "/readyz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixture retrieval expectations against a sidecar"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN3_VL_URL", "http://127.0.0.1:9010"),
    )
    parser.add_argument(
        "--images-dir",
        default="./fixtures/images",
    )
    parser.add_argument(
        "--expected",
        default="./fixtures/images/expected.txt",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--ready-timeout-seconds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require all expected images to appear in top-k",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not wait_for_ready(args.base_url, args.ready_timeout_seconds):
        print(
            f"sidecar not ready at {args.base_url} within {args.ready_timeout_seconds}s",
            file=sys.stderr,
        )
        return 2

    if not os.path.isdir(args.images_dir):
        print(f"images dir not found: {args.images_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.expected):
        print(f"expected mapping not found: {args.expected}", file=sys.stderr)
        return 2

    image_names = sorted(
        name
        for name in os.listdir(args.images_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_names:
        print(f"no image fixtures found in {args.images_dir}", file=sys.stderr)
        return 2

    with open(args.expected, "r", encoding="utf-8") as f:
        expectations = parse_expectations(f.read())
    if not expectations:
        print(f"no expectations parsed from {args.expected}", file=sys.stderr)
        return 2

    missing = find_missing_expected_filenames(image_names, expectations)
    if missing:
        print(
            "warning: expected filenames missing from fixtures: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )

    print("images:", ", ".join(image_names), flush=True)

    image_vectors: dict[str, list[float]] = {}
    for name in image_names:
        image_path = os.path.join(args.images_dir, name)
        with open(image_path, "rb") as f:
            raw = f.read()
        payload = {"image_b64": base64.b64encode(raw).decode("ascii")}

        started = time.time()
        response = post_json(
            args.base_url, "/embed/image-bytes", payload, args.timeout_seconds
        )
        image_vectors[name] = response["embedding"]
        print(f"embedded {name} in {time.time() - started:.1f}s", flush=True)

    overall_ok = True
    for query, expected_names in expectations:
        started = time.time()
        response = post_json(
            args.base_url,
            "/embed/text",
            {"text": query, "prompt_name": "query"},
            args.timeout_seconds,
        )
        query_vector = response["embedding"]
        query_seconds = time.time() - started

        scored = [
            (name, cosine(query_vector, image_vectors[name])) for name in image_names
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        hit_count = query_hit_count(scored, expected_names)
        expected_count = len(expected_names)
        passes = query_passes(hit_count, expected_count, args.strict)
        if not passes:
            overall_ok = False

        expected_set = set(expected_names)
        print(f"\nquery: {query} (embedded in {query_seconds:.1f}s)")
        print("expected:", ", ".join(expected_names))
        print(f"top{args.top_n}:")
        for rank, (name, score) in enumerate(scored[: args.top_n], start=1):
            marker = "*" if name in expected_set else " "
            print(f"  {rank}. {name:18s} {score:.4f}{marker}")
        print(
            f"top{expected_count} expected hits: {hit_count}/{expected_count} "
            f"(strict={hit_count == expected_count} relaxed={hit_count >= max(1, expected_count - 1)})"
        )

    print("\noverall:", "PASS" if overall_ok else "MIXED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
