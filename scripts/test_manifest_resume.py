"""Resume behaviour for `--manifest` runs.

The deck pipeline rewrites `image_prompts.json` whenever a prompt is edited,
which resets every `status` to Pending. Before the signature sidecar existed,
that made the next `--manifest` run regenerate images that were already
correct — 8 minutes of GPU time per accidental rewrite.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import image_gen


class FakeBackend:
    """Writes a real PNG at the manifest target and records each call."""

    VALID_ASPECT_RATIOS = ("16:9",)

    def __init__(self):
        self.calls = []

    def generate(self, prompt, aspect_ratio, image_size, output_dir,
                 filename=None, model=None):
        # v6.8.0 passes the manifest filename with its extension; every real
        # backend peels a known image extension once, keeping dots in the stem.
        self.calls.append(filename)
        stem = Path(filename).stem if filename else "image"
        target = Path(output_dir) / f"{stem}.png"
        Image.new("RGB", (16, 16), "navy").save(target)
        return str(target)


def _write_manifest(tmp, *, status="Pending", size="1K",
                    prompt="a blueprint schematic"):
    data = {
        "project": "resume-test",
        "generated_at": "2026-10-19",
        "items": [
            {
                "filename": "a.png",
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "image_size": size,
                "status": status,
                "page_role": "local",
                "text_policy": "none",
            }
        ],
    }
    Path(tmp, "image_prompts.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return data


class TestManifestResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.path = os.path.join(self.dir, "image_prompts.json")
        self.backend = FakeBackend()

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *, force=False, size="1K"):
        manifest = image_gen.load_manifest(self.path)
        with contextlib.redirect_stdout(io.StringIO()):
            image_gen._run_manifest(
                manifest, self.path, self.backend,
                initial_concurrency=1,
                image_size=size,
                output_dir=self.dir,
                model=None,
                force=force,
            )
        return json.loads(Path(self.path).read_text(encoding="utf-8"))

    def test_writes_signature_and_sidecar(self):
        _write_manifest(self.dir)
        data = self._run()
        self.assertEqual(self.backend.calls, ["a.png"])
        self.assertEqual(data["items"][0]["status"], "Generated")
        self.assertTrue(data["items"][0].get("prompt_sha"))
        sidecar = Path(self.dir) / image_gen.MANIFEST_STATE_FILENAME
        self.assertTrue(sidecar.is_file())
        self.assertEqual(
            json.loads(sidecar.read_text(encoding="utf-8"))["a.png"]["signature"],
            data["items"][0]["prompt_sha"],
        )

    def test_status_reset_does_not_regenerate(self):
        _write_manifest(self.dir)
        self._run()
        self.assertEqual(len(self.backend.calls), 1)

        # The pipeline rewrites the manifest from its own copy: every status
        # back to Pending and no prompt_sha.
        _write_manifest(self.dir, status="Pending")
        data = self._run()
        self.assertEqual(len(self.backend.calls), 1, "image was regenerated")
        self.assertEqual(data["items"][0]["status"], "Generated")

    def test_changed_prompt_regenerates(self):
        _write_manifest(self.dir)
        self._run()
        _write_manifest(self.dir, status="Pending", prompt="a different diagram")
        self._run()
        self.assertEqual(len(self.backend.calls), 2)

    def test_changed_image_size_regenerates(self):
        _write_manifest(self.dir, size="1K")
        self._run()
        # Same prompt, bigger target: the 1K file is stale, not reusable.
        _write_manifest(self.dir, status="Pending", size="2K")
        self._run(size="2K")
        self.assertEqual(len(self.backend.calls), 2)

    def test_force_regenerates_unchanged_item(self):
        _write_manifest(self.dir)
        self._run()
        data = self._run(force=True)
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(data["items"][0]["status"], "Generated")

    def test_missing_file_is_regenerated(self):
        _write_manifest(self.dir)
        self._run()
        os.remove(os.path.join(self.dir, "a.png"))
        _write_manifest(self.dir, status="Pending")
        self._run()
        self.assertEqual(len(self.backend.calls), 2)

    def test_local_backend_uses_sliding_window_without_inter_batch_sleep(self):
        import threading
        import time
        from unittest import mock

        class LocalPipelineBackend(FakeBackend):
            IS_LOCAL_BACKEND = True
            INTER_BATCH_PAUSE_SECONDS = 0.0

            def __init__(self):
                super().__init__()
                self._lock = threading.Lock()
                self._gate = threading.Event()
                self.active = 0
                self.max_active = 0

            def generate(self, prompt, aspect_ratio, image_size, output_dir,
                         filename=None, model=None):
                with self._lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    # 使用 Event.wait 模拟耗时，避免与被 mock 的 time.sleep 混淆
                    self._gate.wait(0.02 if filename == "a.png" else 0.08)
                    return super().generate(
                        prompt, aspect_ratio, image_size, output_dir,
                        filename=filename, model=model,
                    )
                finally:
                    with self._lock:
                        self.active -= 1

        data = {
            "project": "local-pipeline-test",
            "generated_at": "2026-11-26",
            "items": [
                {
                    "filename": name,
                    "prompt": f"prompt {name}",
                    "aspect_ratio": "16:9",
                    "image_size": "1K",
                    "status": "Pending",
                    "page_role": "local",
                    "text_policy": "none",
                }
                for name in ("a.png", "b.png", "c.png", "d.png")
            ],
        }
        Path(self.path).write_text(json.dumps(data), encoding="utf-8")
        local_backend = LocalPipelineBackend()
        manifest = image_gen.load_manifest(self.path)
        with mock.patch.object(image_gen.time, "sleep") as sleep, \
             contextlib.redirect_stdout(io.StringIO()):
            ok, fail, skipped = image_gen._run_manifest(
                manifest, self.path, local_backend,
                initial_concurrency=2,
                image_size="1K",
                output_dir=self.dir,
                model=None,
            )
        self.assertEqual((ok, fail, skipped), (4, 0, 0))
        self.assertEqual(local_backend.max_active, 2)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
