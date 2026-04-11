#!/usr/bin/env python3
"""
HTTP sidecar for Qwen3-VL-Embedding-8B.

Compatible with the existing imgsearch sidecar API:
  - POST /embed/text
  - POST /embed/image
  - POST /embed/image-bytes
  - GET  /healthz
  - GET  /readyz
  - GET  /stats

This server expects the official Qwen3-VL-Embedding repo to be available so
`src.models.qwen3_vl_embedding.Qwen3VLEmbedder` can be imported.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import resource
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO


MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def _maxrss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


class SidecarMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.requests_total = 0
        self.requests_ok = 0
        self.requests_error = 0
        self.in_flight = 0
        self.embed_text_ok = 0
        self.embed_text_error = 0
        self.embed_image_ok = 0
        self.embed_image_error = 0
        self.total_latency_ms = 0
        self.max_latency_ms = 0
        self.last_error: dict[str, object] | None = None

    def begin(self) -> None:
        with self.lock:
            self.requests_total += 1
            self.in_flight += 1

    def end(self, path: str, status: int, duration_ms: int, error: str | None) -> None:
        with self.lock:
            if self.in_flight > 0:
                self.in_flight -= 1

            self.total_latency_ms += duration_ms
            if duration_ms > self.max_latency_ms:
                self.max_latency_ms = duration_ms

            is_image_path = path in {"/embed/image", "/embed/image-bytes"}

            if status >= 400:
                self.requests_error += 1
                if path == "/embed/text":
                    self.embed_text_error += 1
                if is_image_path:
                    self.embed_image_error += 1
                self.last_error = {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "path": path,
                    "status": status,
                    "message": error or "request failed",
                }
            else:
                self.requests_ok += 1
                if path == "/embed/text":
                    self.embed_text_ok += 1
                if is_image_path:
                    self.embed_image_ok += 1

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            avg_ms = 0.0
            if self.requests_total > 0:
                avg_ms = self.total_latency_ms / self.requests_total
            return {
                "uptime_seconds": int(time.time() - self.started_at),
                "requests_total": self.requests_total,
                "requests_ok": self.requests_ok,
                "requests_error": self.requests_error,
                "in_flight": self.in_flight,
                "embed_text_ok": self.embed_text_ok,
                "embed_text_error": self.embed_text_error,
                "embed_image_ok": self.embed_image_ok,
                "embed_image_error": self.embed_image_error,
                "latency_ms_avg": round(avg_ms, 2),
                "latency_ms_max": self.max_latency_ms,
                "maxrss_bytes": _maxrss_bytes(),
                "last_error": self.last_error,
            }


class Qwen3VLService:
    def __init__(
        self,
        model_id: str,
        repo_path: str,
        allowed_dirs: list[str],
        max_image_pixels: int,
        attn_implementation: str,
        torch_dtype: str,
        query_instruction: str,
        passage_instruction: str,
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency torch. Install torch in this environment first."
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.repo_path = self._resolve_repo_path(repo_path)
        self.allowed_dirs = [os.path.realpath(d) for d in allowed_dirs]
        self.max_image_pixels = max(3136, max_image_pixels)
        self.query_instruction = query_instruction.strip()
        self.passage_instruction = passage_instruction.strip()

        qwen_embedder_class = self._load_embedder_class(self.repo_path)
        kwargs: dict[str, object] = {
            "max_pixels": self.max_image_pixels,
            "attn_implementation": attn_implementation,
        }

        resolved_dtype = self._resolve_dtype(torch_dtype)
        if resolved_dtype is not None:
            kwargs["torch_dtype"] = resolved_dtype

        self.embedder = qwen_embedder_class(
            model_name_or_path=self.model_id,
            **kwargs,
        )

    def _resolve_repo_path(self, explicit: str) -> str:
        candidate = explicit.strip()
        if not candidate:
            candidate = os.environ.get("QWEN3_VL_REPO_PATH", "").strip()
        if not candidate:
            raise RuntimeError(
                "missing Qwen3-VL repo path (set --repo-path or QWEN3_VL_REPO_PATH)"
            )

        real = os.path.realpath(candidate)
        expected = os.path.join(real, "src", "models", "qwen3_vl_embedding.py")
        if not os.path.exists(expected):
            raise RuntimeError(f"invalid qwen repo path {real}: expected {expected}")
        return real

    def _load_embedder_class(self, repo_path: str):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        except Exception as exc:
            raise RuntimeError(
                "cannot import Qwen3VLEmbedder from repo path; ensure repo dependencies are installed"
            ) from exc
        return Qwen3VLEmbedder

    def _resolve_dtype(self, requested: str):
        req = requested.strip().lower()
        if req == "auto":
            if self.torch.cuda.is_available():
                return self.torch.bfloat16
            return self.torch.float32
        if req == "bfloat16":
            return self.torch.bfloat16
        if req == "float16":
            return self.torch.float16
        if req == "float32":
            return self.torch.float32
        raise RuntimeError(
            "invalid torch dtype (expected: auto, bfloat16, float16, float32)"
        )

    def resolve_allowed_path(self, candidate: str) -> str:
        real = os.path.realpath(candidate)
        for root in self.allowed_dirs:
            if real == root or real.startswith(root + os.sep):
                return real
        raise PermissionError("image path outside allowed directories")

    def embed_text(self, text: str, prompt_name: str) -> list[float]:
        instruction = self.query_instruction
        if prompt_name == "passage":
            instruction = self.passage_instruction
        vec = self.embedder.process(
            [
                {
                    "text": text,
                    "instruction": instruction,
                }
            ]
        )
        return self._to_vector(vec)

    def embed_image(self, path: str) -> list[float]:
        vec = self.embedder.process(
            [
                {
                    "image": path,
                    "instruction": self.passage_instruction,
                }
            ]
        )
        return self._to_vector(vec)

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        if not image_bytes:
            raise RuntimeError("missing image bytes")

        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")

        vec = self.embedder.process(
            [
                {
                    "image": rgb,
                    "instruction": self.passage_instruction,
                }
            ]
        )
        return self._to_vector(vec)

    def _to_vector(self, value) -> list[float]:
        current = value
        if hasattr(current, "detach"):
            current = current.detach()
        if hasattr(current, "float"):
            current = current.float()
        if hasattr(current, "cpu"):
            current = current.cpu()
        if hasattr(current, "numpy"):
            current = current.numpy()
        if hasattr(current, "tolist"):
            current = current.tolist()

        if isinstance(current, list) and current and isinstance(current[0], list):
            current = current[0]

        if not isinstance(current, list) or not current:
            raise RuntimeError("embedding output is empty")

        return [float(x) for x in current]


class Handler(BaseHTTPRequestHandler):
    service: Qwen3VLService
    metrics = SidecarMetrics()
    inference_lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        request_id = self.headers.get("X-Request-ID", "") or uuid.uuid4().hex[:12]
        started = time.monotonic()
        status = 500
        err_message: str | None = None
        Handler.metrics.begin()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_REQUEST_BODY_BYTES:
                status = 413
                return self._send(status, {"error": "request body too large"})
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}

            if self.path == "/embed/text":
                text = payload.get("text", "")
                if not isinstance(text, str) or not text:
                    status = 400
                    return self._send(status, {"error": "missing text"})

                prompt_name = payload.get("prompt_name", "query")
                if not isinstance(prompt_name, str) or prompt_name not in {
                    "query",
                    "passage",
                }:
                    status = 400
                    return self._send(status, {"error": "invalid prompt_name"})

                with Handler.inference_lock:
                    embedding = self.service.embed_text(text, prompt_name)
                status = 200
                return self._send(status, {"embedding": embedding})

            if self.path == "/embed/image":
                path = payload.get("path", "")
                if not isinstance(path, str) or not path:
                    status = 400
                    return self._send(status, {"error": "missing path"})

                safe_path = self.service.resolve_allowed_path(path)
                if not os.path.exists(safe_path):
                    status = 404
                    return self._send(status, {"error": "image path not found"})

                with Handler.inference_lock:
                    embedding = self.service.embed_image(safe_path)
                status = 200
                return self._send(status, {"embedding": embedding})

            if self.path == "/embed/image-bytes":
                image_b64 = payload.get("image_b64")
                if image_b64 is None:
                    image_b64 = payload.get("bytes_b64", "")
                if not isinstance(image_b64, str) or not image_b64:
                    status = 400
                    return self._send(status, {"error": "missing image_b64"})

                try:
                    raw = base64.b64decode(image_b64, validate=True)
                except Exception:
                    status = 400
                    return self._send(status, {"error": "invalid image_b64"})
                if len(raw) > MAX_IMAGE_BYTES:
                    status = 413
                    return self._send(status, {"error": "image payload too large"})

                with Handler.inference_lock:
                    embedding = self.service.embed_image_bytes(raw)
                status = 200
                return self._send(status, {"embedding": embedding})

            status = 404
            return self._send(status, {"error": "not found"})
        except PermissionError as exc:  # pragma: no cover
            status = 403
            err_message = str(exc)
            return self._send(status, {"error": err_message})
        except Exception as exc:  # pragma: no cover
            status = 500
            err_message = str(exc)
            traceback.print_exc()
            return self._send(status, {"error": err_message})
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            Handler.metrics.end(self.path, status, duration_ms, err_message)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            line = (
                f"{stamp} req={request_id} method=POST path={self.path} "
                f"status={status} duration_ms={duration_ms}"
            )
            if err_message:
                line += f" error={err_message}"
            print(line, flush=True)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            return self._send(200, {"status": "ok"})
        if self.path == "/readyz":
            return self._send(200, {"status": "ready"})
        if self.path == "/stats":
            payload = Handler.metrics.snapshot()
            payload["model_id"] = self.service.model_id
            payload["repo_path"] = self.service.repo_path
            payload["max_image_pixels"] = self.service.max_image_pixels
            payload["cuda_available"] = bool(self.service.torch.cuda.is_available())
            return self._send(200, payload)
        return self._send(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL embedding sidecar server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-8B")
    parser.add_argument(
        "--repo-path",
        default="",
        help="Path to cloned https://github.com/QwenLM/Qwen3-VL-Embedding repository",
    )
    parser.add_argument(
        "--allow-dir",
        action="append",
        default=[],
        help="Allowed image root directory. May be passed multiple times.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=1_843_200,
        help="Maximum pixels for image encoding (default: 1,843,200)",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Transformers attention implementation (default: sdpa)",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="auto, bfloat16, float16, or float32",
    )
    parser.add_argument(
        "--query-instruction",
        default="Represent this query for retrieving relevant images and text.",
    )
    parser.add_argument(
        "--passage-instruction",
        default="Represent this image or text for retrieval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed = args.allow_dir or [os.path.join(os.getcwd(), "data", "images")]

    service = Qwen3VLService(
        model_id=args.model_id,
        repo_path=args.repo_path,
        allowed_dirs=allowed,
        max_image_pixels=args.max_image_pixels,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        query_instruction=args.query_instruction,
        passage_instruction=args.passage_instruction,
    )
    Handler.service = service

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"qwen3-vl server listening on http://{args.host}:{args.port}")
    print(f"model: {service.model_id}")
    print(f"repo_path: {service.repo_path}")
    print(f"max image pixels: {service.max_image_pixels}")
    print("allowed image roots:")
    for root in service.allowed_dirs:
        print(f"  - {root}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
