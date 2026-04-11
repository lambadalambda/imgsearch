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
import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class JinaMLXService:
    def __init__(
        self,
        model_id: str,
        processor_id: str,
        allowed_dirs: list[str],
    ) -> None:
        try:
            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependencies. Install with: pip install mlx mlx-lm numpy "
                "huggingface_hub transformers pillow requests"
            ) from exc

        self.mx = mx
        self.model_id = model_id
        self.processor_id = processor_id
        self.allowed_dirs = [os.path.realpath(d) for d in allowed_dirs]

        model_dir = snapshot_download(model_id)
        load_mlx_model = self._load_model_loader(model_dir)

        self.model = load_mlx_model(model_dir)
        self.processor = AutoProcessor.from_pretrained(processor_id)

    def _load_model_loader(self, model_dir: str):
        loader_path = os.path.join(model_dir, "load_model.py")
        spec = importlib.util.spec_from_file_location(
            "jina_mlx_load_model", loader_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load model loader from {loader_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "load_mlx_model"):
            raise RuntimeError(f"load_mlx_model not found in {loader_path}")
        return module.load_mlx_model

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
        from PIL import Image

        image = Image.open(path).convert("RGB")
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
        return out.tolist()[0]


class Handler(BaseHTTPRequestHandler):
    service: JinaMLXService

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}

            if self.path == "/embed/text":
                text = payload.get("text", "")
                if not isinstance(text, str) or not text:
                    return self._send(400, {"error": "missing text"})
                embedding = self.service.embed_text(text)
                return self._send(200, {"embedding": embedding})

            if self.path == "/embed/image":
                path = payload.get("path", "")
                if not isinstance(path, str) or not path:
                    return self._send(400, {"error": "missing path"})
                safe_path = self.service.resolve_allowed_path(path)
                if not os.path.exists(safe_path):
                    return self._send(404, {"error": "image path not found"})
                embedding = self.service.embed_image(safe_path)
                return self._send(200, {"embedding": embedding})

            return self._send(404, {"error": "not found"})
        except PermissionError as exc:  # pragma: no cover
            return self._send(403, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            return self._send(500, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            return self._send(200, {"status": "ok"})
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    allowed = args.allow_dir or [os.path.join(os.getcwd(), "data", "images")]
    service = JinaMLXService(args.model_id, args.processor_id, allowed)
    Handler.service = service

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"jina-mlx server listening on http://{args.host}:{args.port}")
    print(f"model: {args.model_id}")
    print("allowed image roots:")
    for root in service.allowed_dirs:
        print(f"  - {root}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
