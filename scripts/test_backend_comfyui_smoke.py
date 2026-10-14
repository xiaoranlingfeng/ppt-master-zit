"""GPU-free smoke tests for the comfyui backend (CI-safe, no ComfyUI needed).

Catches import-time regressions and pure-function bugs in backend_comfyui.py
without touching a real server. Run from the scripts/ directory.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_backends import backend_comfyui as bc  # noqa: E402


class TestResolveDimensions(unittest.TestCase):
    def test_16x9_1k(self):
        self.assertEqual(bc._resolve_dimensions("16:9", "1K"), (1344, 736))

    def test_1x1_1k(self):
        self.assertEqual(bc._resolve_dimensions("1:1", "1K"), (992, 992))

    def test_9x16_512px(self):
        self.assertEqual(bc._resolve_dimensions("9:16", "512px"), (384, 672))

    def test_3x1_2k(self):
        self.assertEqual(bc._resolve_dimensions("3:1", "2K"), (3456, 1152))

    def test_vae_multiple_of_32(self):
        for ratio in ("16:9", "1:1", "9:16", "3:2", "4:3", "21:9"):
            for size in ("512px", "1K", "2K"):
                w, h = bc._resolve_dimensions(ratio, size)
                self.assertGreater(w, 0, (ratio, size))
                self.assertGreater(h, 0, (ratio, size))
                self.assertEqual(w % 32, 0, f"{ratio}@{size} width {w} not /32")
                self.assertEqual(h % 32, 0, f"{ratio}@{size} height {h} not /32")


class TestFindNodes(unittest.TestCase):
    def test_matches_by_class_type(self):
        graph = {
            "1": {"class_type": "KSampler", "inputs": {}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": bc.PROMPT_PLACEHOLDER}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "other"}},
        }
        self.assertEqual(bc._find_nodes(graph, "CLIPTextEncode"), ["2", "3"])
        self.assertEqual(bc._find_nodes(graph, "VAELoader"), [])


class TestPatchGraph(unittest.TestCase):
    def _base(self):
        return {
            "10": {"class_type": "CLIPTextEncode",
                   "inputs": {"text": bc.PROMPT_PLACEHOLDER}},
            "11": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": 1, "height": 1, "batch_size": 99}},
        }

    def test_positive_injection_and_size(self):
        out = bc._patch_graph(self._base(), "hello", 1344, 736, None, "prefix")
        self.assertEqual(out["10"]["inputs"]["text"], "hello")
        self.assertEqual(out["11"]["inputs"]["width"], 1344)
        self.assertEqual(out["11"]["inputs"]["height"], 736)
        self.assertEqual(out["11"]["inputs"]["batch_size"], 1)

    def test_input_not_mutated(self):
        base = self._base()
        bc._patch_graph(base, "changed", 1024, 576, None, "p")
        self.assertEqual(base["10"]["inputs"]["text"], bc.PROMPT_PLACEHOLDER)

    def test_negative_sweeps_non_positive_clip(self):
        g = self._base()
        g["12"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "old negative"}}
        out = bc._patch_graph(g, "pos", 512, 512, "my negative", "p")
        self.assertEqual(out["10"]["inputs"]["text"], "pos")
        self.assertEqual(out["12"]["inputs"]["text"], "my negative")

    def test_missing_positive_placeholder_raises(self):
        g = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "no placeholder here"}}}
        with self.assertRaises(ValueError):
            bc._patch_graph(g, "p", 512, 512, None, "x")


class TestBackendRegistration(unittest.TestCase):
    def test_comfyui_is_a_known_backend(self):
        # image_gen.py must know the backend + its aliases; this import alone
        # guards against the "fork dropped the registry entry" rebase bug.
        import importlib
        ig = importlib.import_module("image_gen")
        registry = ig.BACKEND_REGISTRY
        self.assertIn("comfyui", registry)
        entry = registry["comfyui"]
        self.assertEqual(entry.get("module"), "backend_comfyui")
        self.assertIn("zit", entry.get("aliases", []))
        self.assertIn("comfyui", ig.SUPPORTED_BACKENDS)


if __name__ == "__main__":
    unittest.main()
