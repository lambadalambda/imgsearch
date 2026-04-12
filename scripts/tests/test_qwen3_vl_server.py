import os
import sys
import tempfile
import unittest

from scripts.qwen3_vl_server import Qwen3VLService


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def _clear_embedder_modules() -> None:
    for name in [
        "src.models.qwen3_vl_embedding",
        "src.models",
        "src",
        "scripts.qwen3_vl_embedding",
    ]:
        sys.modules.pop(name, None)


class Qwen3RepoPathTests(unittest.TestCase):
    def _service(self) -> Qwen3VLService:
        return Qwen3VLService.__new__(Qwen3VLService)

    def test_resolve_repo_path_accepts_src_models_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write(
                os.path.join(temp_dir, "src", "models", "qwen3_vl_embedding.py"),
                "class Qwen3VLEmbedder:\n    pass\n",
            )
            service = self._service()
            self.assertEqual(
                service._resolve_repo_path(temp_dir), os.path.realpath(temp_dir)
            )

    def test_resolve_repo_path_accepts_scripts_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write(
                os.path.join(temp_dir, "scripts", "qwen3_vl_embedding.py"),
                "class Qwen3VLEmbedder:\n    pass\n",
            )
            service = self._service()
            self.assertEqual(
                service._resolve_repo_path(temp_dir), os.path.realpath(temp_dir)
            )


class Qwen3EmbedderImportTests(unittest.TestCase):
    def _service(self) -> Qwen3VLService:
        return Qwen3VLService.__new__(Qwen3VLService)

    def test_load_embedder_class_from_scripts_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write(
                os.path.join(temp_dir, "scripts", "qwen3_vl_embedding.py"),
                "class Qwen3VLEmbedder:\n    pass\n",
            )
            _clear_embedder_modules()
            service = self._service()
            embedder_cls = service._load_embedder_class(temp_dir)
            self.assertEqual(embedder_cls.__name__, "Qwen3VLEmbedder")


if __name__ == "__main__":
    unittest.main()
