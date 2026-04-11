#!/usr/bin/env python3
"""Local HTTP server for jina-embeddings-v4 (PyTorch).

Endpoints:
  POST /embed/text   body: {"text": "...", "prompt_name": "query|passage"}
  POST /embed/image  body: {"path": "/abs/or/rel/path.jpg"}
  GET  /healthz
  GET  /readyz
  GET  /stats

Response for embed endpoints:
  {"embedding": [float, ...]}
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
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def _maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


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


class JinaTorchService:
    def __init__(
        self,
        model_id: str,
        allowed_dirs: list[str],
        device: str,
        max_image_pixels: int,
        enable_mps_fallback: bool,
    ) -> None:
        if enable_mps_fallback and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        try:
            import torch
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependencies. Install with: "
                "pip install torch transformers peft torchvision pillow requests"
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.allowed_dirs = [os.path.realpath(d) for d in allowed_dirs]
        self.max_image_pixels = max(3136, max_image_pixels)

        self.device = self._resolve_device(device)
        self._patch_autocast_for_mps()

        dtype = (
            self.torch.float16
            if self.device.startswith("cuda") or self.device == "mps"
            else self.torch.float32
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        self.model = self.model.to(self.device)
        self.model.task = "retrieval"
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        requested = requested.strip().lower()
        if requested in {"", "auto"}:
            if self.torch.cuda.is_available():
                return "cuda"
            if self.torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda":
            if not self.torch.cuda.is_available():
                raise RuntimeError("requested device cuda but CUDA is not available")
            return "cuda"
        if requested.startswith("cuda:"):
            if not self.torch.cuda.is_available():
                raise RuntimeError(
                    f"requested device {requested} but CUDA is not available"
                )
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise RuntimeError(
                    "invalid cuda device format (expected cuda or cuda:N)"
                ) from exc
            if index < 0 or index >= self.torch.cuda.device_count():
                raise RuntimeError(
                    f"requested CUDA device index {index} out of range (found {self.torch.cuda.device_count()} devices)"
                )
            return requested
        if requested == "mps":
            if not self.torch.backends.mps.is_available():
                raise RuntimeError("requested device mps but MPS is not available")
            return "mps"
        if requested == "cpu":
            return "cpu"
        raise RuntimeError("invalid device (expected: auto, cuda, cuda:N, mps, or cpu)")

    def _patch_autocast_for_mps(self) -> None:
        if self.device != "mps":
            return

        original_autocast = self.torch.autocast

        def patched_autocast(*args, **kwargs):
            device_type = kwargs.get("device_type")
            if device_type is None and len(args) >= 1:
                device_type = args[0]
            if device_type == "mps":
                kwargs["dtype"] = self.torch.float16
            return original_autocast(*args, **kwargs)

        self.torch.autocast = patched_autocast

    def resolve_allowed_path(self, candidate: str) -> str:
        real = os.path.realpath(candidate)
        for root in self.allowed_dirs:
            if real == root or real.startswith(root + os.sep):
                return real
        raise PermissionError("image path outside allowed directories")

    def embed_text(self, text: str, prompt_name: str) -> list[float]:
        with self.torch.no_grad():
            vec = self.model.encode_text(
                texts=[text],
                task="retrieval",
                prompt_name=prompt_name,
                return_numpy=False,
            )[0]
        return vec.float().cpu().tolist()

    def embed_image(self, path: str) -> list[float]:
        with self.torch.no_grad():
            vec = self.model.encode_image(
                images=[path],
                task="retrieval",
                max_pixels=self.max_image_pixels,
                return_numpy=False,
            )[0]
        return vec.float().cpu().tolist()

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        if not image_bytes:
            raise RuntimeError("missing image bytes")

        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")

        with self.torch.no_grad():
            vec = self.model.encode_image(
                images=[rgb],
                task="retrieval",
                max_pixels=self.max_image_pixels,
                return_numpy=False,
            )[0]
        return vec.float().cpu().tolist()


class Handler(BaseHTTPRequestHandler):
    service: JinaTorchService
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
            payload["device"] = self.service.device
            payload["max_image_pixels"] = self.service.max_image_pixels
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
    p = argparse.ArgumentParser(description="Local jina torch embedding server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9009)
    p.add_argument("--model-id", default="jinaai/jina-embeddings-v4")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps, or cpu")
    p.add_argument(
        "--allow-dir",
        action="append",
        default=[],
        help="Allowed image root directory. May be passed multiple times.",
    )
    p.add_argument(
        "--max-image-pixels",
        type=int,
        default=602112,
        help="Maximum pixels for encode_image (default: 602112)",
    )
    p.add_argument(
        "--disable-mps-fallback",
        action="store_true",
        help="Disable PYTORCH_ENABLE_MPS_FALLBACK=1 behavior",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    allowed = args.allow_dir or [os.path.join(os.getcwd(), "data", "images")]

    service = JinaTorchService(
        args.model_id,
        allowed,
        args.device,
        args.max_image_pixels,
        not args.disable_mps_fallback,
    )
    Handler.service = service

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"jina-torch server listening on http://{args.host}:{args.port}")
    print(f"model: {args.model_id}")
    print(f"device: {service.device}")
    print(f"max image pixels: {service.max_image_pixels}")
    print("allowed image roots:")
    for root in service.allowed_dirs:
        print(f"  - {root}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
