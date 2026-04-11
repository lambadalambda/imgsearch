#!/usr/bin/env python3
"""Minimal local HTTP server for jina-embeddings-v4-mlx-8bit.

Endpoints:
  POST /embed/text   body: {"text": "..."}
  POST /embed/image  body: {"path": "/abs/or/rel/path.jpg"}

Response:
  {"embedding": [float, ...]}
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import resource
import shutil
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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

            if status >= 400:
                self.requests_error += 1
                if path == "/embed/text":
                    self.embed_text_error += 1
                if path == "/embed/image":
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
                if path == "/embed/image":
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


class JinaMLXService:
    def __init__(
        self,
        model_id: str,
        processor_id: str,
        allowed_dirs: list[str],
        max_image_pixels: int,
    ) -> None:
        try:
            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependencies. Install with: "
                "pip install mlx mlx-lm numpy huggingface_hub 'transformers==4.52.0' "
                "pillow requests torch peft torchvision"
            ) from exc

        self.mx = mx
        self.model_id = model_id
        self.processor_id = processor_id
        self.allowed_dirs = [os.path.realpath(d) for d in allowed_dirs]
        self.max_image_pixels = max(256 * 256, max_image_pixels)

        model_dir = snapshot_download(model_id)
        self._ensure_weight_aliases(model_dir)
        load_mlx_model = self._load_model_loader(model_dir)

        self.model = load_mlx_model(model_dir)
        # Upstream MLX model aggressively clears weights after first image call.
        # For a long-running HTTP service we need repeated text/image requests.
        self.model.visual.clear_weights = lambda: None
        self.model._clear_model_weights = lambda: None
        self.processor = AutoProcessor.from_pretrained(
            processor_id,
            trust_remote_code=True,
        )

    def _load_model_loader(self, model_dir: str):
        loader_path = os.path.join(model_dir, "load_model.py")
        spec = importlib.util.spec_from_file_location(
            "jina_mlx_load_model", loader_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load model loader from {loader_path}")
        module = importlib.util.module_from_spec(spec)

        inserted = False
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
            inserted = True
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted and sys.path and sys.path[0] == model_dir:
                sys.path.pop(0)

        if not hasattr(module, "load_mlx_model"):
            raise RuntimeError(f"load_mlx_model not found in {loader_path}")
        return module.load_mlx_model

    def _ensure_weight_aliases(self, model_dir: str) -> None:
        aliases = [
            ("model.safetensors.index.json", "weights.safetensors.index.json"),
            ("model.safetensors", "weights.safetensors"),
        ]
        for source_name, alias_name in aliases:
            source = os.path.join(model_dir, source_name)
            alias = os.path.join(model_dir, alias_name)
            if os.path.exists(alias) or not os.path.exists(source):
                continue
            try:
                os.symlink(source, alias)
            except OSError:
                shutil.copyfile(source, alias)

    def resolve_allowed_path(self, candidate: str) -> str:
        real = os.path.realpath(candidate)
        for root in self.allowed_dirs:
            if real == root or real.startswith(root + os.sep):
                return real
        raise PermissionError("image path outside allowed directories")

    def embed_text(self, text: str) -> list[float]:
        prompt = "<|im_start|>user\n" + text + "<|im_end|>"
        inputs = self.processor(
            text=[prompt],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )

        out = self.model.encode_text(
            input_ids=self.mx.array(inputs["input_ids"]),
            attention_mask=self.mx.array(inputs["attention_mask"]),
            task="retrieval",
        )
        self.mx.eval(out)
        return out.tolist()[0]

    def embed_image(self, path: str) -> list[float]:
        try:
            return self._embed_image_once(path, self.max_image_pixels)
        except RuntimeError as exc:
            if not self._is_memory_error(exc):
                raise

            fallback_pixels = min(self.max_image_pixels, 1024 * 1024)
            if fallback_pixels >= self.max_image_pixels:
                raise

            print(
                f"retry image embedding with reduced pixels={fallback_pixels} path={path} error={exc}",
                flush=True,
            )
            gc.collect()
            return self._embed_image_once(path, fallback_pixels)

    def _is_memory_error(self, exc: Exception) -> bool:
        msg = str(exc)
        return (
            "Unable to allocate" in msg
            or "Command buffer execution failed" in msg
            or "METAL" in msg
        )

    def _prepare_image(self, path: str, max_pixels: int):
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB")

        pixels = image.size[0] * image.size[1]
        if pixels > max_pixels:
            scale = math.sqrt(max_pixels / float(pixels))
            width = max(1, int(image.size[0] * scale))
            height = max(1, int(image.size[1] * scale))
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        return image

    def _embed_image_once(self, path: str, max_pixels: int) -> list[float]:
        image = self._prepare_image(path, max_pixels)
        prompt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>Describe the image."
            "<|im_end|>"
        )
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="np",
            padding=True,
        )

        out = self.model.encode_image(
            input_ids=self.mx.array(inputs["input_ids"]),
            pixel_values=self.mx.array(
                inputs["pixel_values"].reshape(-1, inputs["pixel_values"].shape[-1])
            ),
            image_grid_thw=[tuple(r) for r in inputs["image_grid_thw"]],
            attention_mask=self.mx.array(inputs["attention_mask"]),
            task="retrieval",
        )
        self.mx.eval(out)
        vec = out.tolist()[0]
        del out
        del inputs
        del image
        gc.collect()
        return vec


class Handler(BaseHTTPRequestHandler):
    service: JinaMLXService
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
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}

            if self.path == "/embed/text":
                text = payload.get("text", "")
                if not isinstance(text, str) or not text:
                    status = 400
                    return self._send(status, {"error": "missing text"})
                with Handler.inference_lock:
                    embedding = self.service.embed_text(text)
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
            return self._send(200, Handler.metrics.snapshot())
        return self._send(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep stdout clean for local dev UX.
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local jina mlx embedding server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9009)
    p.add_argument("--model-id", default="jinaai/jina-embeddings-v4-mlx-8bit")
    p.add_argument("--processor-id", default="jinaai/jina-embeddings-v4")
    p.add_argument(
        "--allow-dir",
        action="append",
        default=[],
        help="Allowed image root directory. May be passed multiple times.",
    )
    p.add_argument(
        "--max-image-pixels",
        type=int,
        default=3_145_728,
        help="Maximum pixels per image before downscaling (default: 3,145,728 ~= 2048x1536)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    allowed = args.allow_dir or [os.path.join(os.getcwd(), "data", "images")]
    service = JinaMLXService(
        args.model_id,
        args.processor_id,
        allowed,
        args.max_image_pixels,
    )
    Handler.service = service

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"jina-mlx server listening on http://{args.host}:{args.port}")
    print(f"model: {args.model_id}")
    print("allowed image roots:")
    for root in service.allowed_dirs:
        print(f"  - {root}")
    print(f"max image pixels: {service.max_image_pixels}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
