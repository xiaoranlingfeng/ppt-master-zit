"""GPU-free smoke tests for the comfyui backend (CI-safe, no ComfyUI needed).

Catches import-time regressions and pure-function bugs in backend_comfyui.py
without touching a real server. Run from the scripts/ directory.
"""
import contextlib
import io
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
    def setUp(self):
        # Sampler and encoder overrides come from the caller's environment;
        # these tests assert template behavior, so start from a clean slate.
        self._saved = {}
        for key in ("COMFY_TEXT_RESOLUTION", "COMFY_STEPS", "COMFY_CFG",
                    "COMFY_SAMPLER", "COMFY_SCHEDULER", "COMFY_SHIFT"):
            self._saved[key] = os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

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

    def _qwen_graph(self):
        return {
            "452": {"class_type": "TextEncodeQwenImage21",
                    "inputs": {"prompt": bc.PROMPT_PLACEHOLDER,
                               "negative_prompt": "", "resolution": 1024}},
            "456": {"class_type": "EmptyLatentImage",
                    "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        }

    def test_encoder_resolution_follows_the_size_tier(self):
        # 1K 16:9 (1344x736) is within 20% of the shipped 1024: leave it alone so
        # the measured default path keeps sending the values it shipped with.
        out = bc._patch_graph(self._qwen_graph(), "p", 1344, 736, None, "x",
                              model="qwen")
        self.assertEqual(out["452"]["inputs"]["resolution"], 1024)
        # 2K is 4x the pixels, so the encoder hint has to move with the latent.
        out = bc._patch_graph(self._qwen_graph(), "p", 2656, 1504, None, "x",
                              model="qwen")
        self.assertEqual(out["452"]["inputs"]["resolution"], 1984)

    def test_shipped_qwen_template_resolution_tracks_tier(self):
        path, _ = bc._resolve_template("qwen")
        graph = bc._load_template(path)
        for w, h, expected in ((1344, 736, 1024), (2656, 1504, 1984)):
            out = bc._patch_graph(graph, "p", w, h, None, "x", model="qwen")
            resolutions = [n["inputs"].get("resolution")
                           for n in out.values()
                           if n.get("class_type") == "TextEncodeQwenImage21"]
            self.assertEqual(resolutions, [expected], f"{w}x{h}")

    def test_encoder_resolution_override_and_other_models_noop(self):
        os.environ["COMFY_TEXT_RESOLUTION"] = "1536"
        out = bc._patch_graph(self._qwen_graph(), "p", 1344, 736, None, "x",
                              model="qwen")
        self.assertEqual(out["452"]["inputs"]["resolution"], 1536)
        # A CLIPTextEncode template has no such input and must stay untouched.
        out = bc._patch_graph(self._base(), "p", 2656, 1504, None, "x", model="moss")
        self.assertNotIn("resolution", out["10"]["inputs"])


class TestSamplerOverrideWarning(unittest.TestCase):
    """A sampler value kept from one model must not silently retune another."""

    _KEYS = ("COMFY_STEPS", "COMFY_CFG", "COMFY_SAMPLER", "COMFY_SCHEDULER",
             "COMFY_SHIFT")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ.pop(k, None)
        bc._OVERRIDE_WARNINGS_EMITTED.clear()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        bc._OVERRIDE_WARNINGS_EMITTED.clear()

    def _moss_graph(self):
        path, _ = bc._resolve_template("moss")
        return bc._load_template(path)

    def _patch(self, **env):
        for k, v in env.items():
            os.environ[k] = v
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = bc._patch_graph(self._moss_graph(), "p", 1328, 752, None, "x",
                                  model="moss")
        return out, buf.getvalue()

    def test_qwen_steps_leaking_into_moss_warns(self):
        out, text = self._patch(COMFY_STEPS="25")
        self.assertIn("COMFY_STEPS=25", text)
        self.assertIn("MOSS-Image SFT", text)
        self.assertIn("50", text)
        # The override is still honored; it is only made visible.
        self.assertEqual(out["16"]["inputs"]["steps"], 25)

    def test_matching_value_is_silent(self):
        out, text = self._patch(COMFY_STEPS="50", COMFY_CFG="3.0",
                                COMFY_SAMPLER="euler",
                                COMFY_SCHEDULER="sgm_uniform")
        self.assertEqual(text, "")
        self.assertEqual(out["16"]["inputs"]["steps"], 50)

    def test_warning_is_emitted_once_per_model_and_key(self):
        _, first = self._patch(COMFY_STEPS="25")
        _, second = self._patch(COMFY_STEPS="25")
        self.assertIn("COMFY_STEPS=25", first)
        self.assertEqual(second, "")


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
        self.assertIn("moss", entry.get("aliases", []))
        self.assertIn("comfyui", ig.SUPPORTED_BACKENDS)


class TestModelSelection(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("COMFY_MODEL", "COMFY_WORKFLOW")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_named_models_resolve_to_shipped_templates(self):
        for name, expected in (("qwen", "qwen_image_2_1_t2i_api.json"),
                               ("moss", "moss_image_sft_api.json"),
                               ("zit", "zit_basic_api.json")):
            path, key = bc._resolve_template(name)
            self.assertEqual(key, name)
            self.assertEqual(path.name, expected)
            self.assertTrue(path.is_file(), f"missing template {path}")

    def test_moss_aliases(self):
        for alias in ("moss-image", "MOSS_IMAGE_SFT", "moss-sft"):
            self.assertEqual(bc._resolve_model_name(alias), "moss")
            self.assertEqual(bc._resolve_template(alias)[1], "moss")

    def test_default_is_qwen(self):
        self.assertEqual(bc._resolve_template(None)[1], "qwen")

    def test_personal_env_pins_qwen_default(self):
        # `.env` carries personal paths and is gitignored, so a clean checkout
        # (and CI) has none. When one exists it must not drift off the default.
        env = bc._SCRIPTS_DIR.parent / ".env"
        if not env.is_file():
            self.skipTest(f"no personal .env at {env}; nothing to pin")
        values = {}
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
        self.assertEqual(values.get("COMFY_MODEL"), "qwen")
        self.assertEqual(values.get("IMAGE_BACKEND"), "comfyui")

    def test_env_example_documents_every_local_model(self):
        # The example config is where a new machine learns the backend exists,
        # so its ComfyUI block must name COMFY_MODEL and all three families.
        text = (bc._SCRIPTS_DIR.parent / ".env.example").read_text(encoding="utf-8")
        start = text.index("Local ComfyUI")
        block = text[start:text.index("# Example:", start)]
        self.assertIn("COMFY_MODEL", block)
        for token in ("qwen", "moss", "zit"):
            self.assertIn(token, block.lower(), f"COMFY_MODEL list omits {token}")

    def test_default_image_timeout_covers_qwen_2k(self):
        # Measured 2K Qwen runs take ~80 s, plus queue time behind a busy GPU.
        self.assertGreaterEqual(bc.DEFAULT_IMAGE_TIMEOUT, 900)

    def test_comfy_model_env(self):
        os.environ["COMFY_MODEL"] = "moss"
        path, key = bc._resolve_template(None)
        self.assertEqual(key, "moss")
        self.assertEqual(path.name, "moss_image_sft_api.json")

    def test_explicit_model_beats_env(self):
        os.environ["COMFY_MODEL"] = "moss"
        self.assertEqual(bc._resolve_template("qwen")[1], "qwen")

    def test_custom_template_path_is_not_a_model(self):
        path, key = bc._resolve_template("C:/somewhere/custom_api.json")
        self.assertIsNone(key)
        self.assertEqual(path.name, "custom_api.json")

    def test_workflow_env_matching_shipped_template_infers_model(self):
        os.environ["COMFY_WORKFLOW"] = str(
            bc._COMFY_TEMPLATE_DIR / "moss_image_sft_api.json")
        path, key = bc._resolve_template(None)
        self.assertEqual(key, "moss")
        self.assertEqual(path.name, "moss_image_sft_api.json")


class TestMossDimensions(unittest.TestCase):
    def test_moss_sizes_are_multiples_of_16_at_least_64(self):
        for ratio, size in (("16:9", "1K"), ("1:1", "1K"), ("9:16", "512px"),
                            ("16:9", "2K"), ("21:9", "1K")):
            w, h = bc._resolve_dimensions(ratio, size, "moss")
            self.assertEqual(w % 16, 0, f"{ratio}@{size} width {w} not /16")
            self.assertEqual(h % 16, 0, f"{ratio}@{size} height {h} not /16")
            self.assertGreaterEqual(min(w, h), 64, f"{ratio}@{size} too small")

    def test_moss_uses_its_own_16px_alignment(self):
        # MOSS's Flux2 VAE only needs 16px alignment, so 16:9 @1K lands on the
        # exact 1.777 ratio (1328x752) rather than qwen's 32px-snapped
        # 1344x736, while still remaining a valid qwen size (multiple of 32).
        qwen = bc._resolve_dimensions("16:9", "1K", "qwen")
        moss = bc._resolve_dimensions("16:9", "1K", "moss")
        self.assertEqual(qwen, (1344, 736))
        self.assertEqual(moss, (1328, 752))
        # MOSS lands closer to the requested 16:9 than the 32px-snapped qwen size
        self.assertLess(abs(moss[0] / moss[1] - 16 / 9),
                        abs(qwen[0] / qwen[1] - 16 / 9))
        self.assertEqual(moss[0] % 16, 0)
        self.assertEqual(moss[1] % 16, 0)

    def test_moss_snaps_to_16_where_qwen_would_not(self):
        # MOSS's Flux2 VAE only needs 16px alignment, so a size that is not a
        # multiple of 32 lands closer to the requested ratio.
        self.assertEqual(bc._resolve_dimensions("10:16", "512px", "moss")[1] % 16, 0)
        w, h = bc._apply_dimension_rule(1030, 1030, "moss")
        self.assertEqual((w, h), (1024, 1024))

    def test_unknown_model_falls_back_to_qwen_rules(self):
        self.assertEqual(bc._resolve_dimensions("16:9", "1K", "custom"),
                         (1344, 736))


class TestMossTemplateAndPromptGuard(unittest.TestCase):
    def _moss_graph(self):
        path, _ = bc._resolve_template("moss")
        return bc._load_template(path)

    def test_moss_template_patches_prompt_and_size(self):
        graph = self._moss_graph()
        out = bc._patch_graph(graph, "a cat on a windowsill", 1024, 576,
                              "my negative", "moss")
        self.assertEqual(out["4"]["inputs"]["text"], "a cat on a windowsill")
        self.assertEqual(out["5"]["inputs"]["text"], "my negative")
        self.assertEqual(out["6"]["inputs"]["width"], 1024)
        self.assertEqual(out["6"]["inputs"]["height"], 576)
        self.assertEqual(out["6"]["inputs"]["batch_size"], 1)
        # fresh seed always injected
        self.assertNotEqual(out["16"]["inputs"]["seed"], 0)

    def test_moss_template_shape(self):
        g = self._moss_graph()
        self.assertEqual(g["31"]["inputs"]["type"], "moss_image")
        self.assertEqual(g["32"]["inputs"]["unet_name"],
                         "moss_image_sft_int8_convrot.safetensors")
        self.assertEqual(g["16"]["inputs"]["cfg"], 3.0)
        self.assertEqual(g["16"]["inputs"]["steps"], 50)
        self.assertEqual(g["16"]["inputs"]["scheduler"], "sgm_uniform")
        # ModelSamplingAuraFlow(flow), NOT ModelSamplingSD3: SD3's implicit
        # SD3 shift stacks on the model's own 5.0 sampling shift and the output
        # comes back as noise (verified against the local server).
        self.assertNotIn("ModelSamplingSD3",
                         {n["class_type"] for n in g.values()})
        sampling = [n for n in g.values()
                    if n["class_type"] == "ModelSamplingAuraFlow"]
        self.assertEqual(len(sampling), 1)
        self.assertEqual(sampling[0]["inputs"]["shift"], 5.0)
        self.assertEqual(sampling[0]["inputs"]["sampling"], "flow")

    def test_shift_override_reaches_moss_node(self):
        saved = os.environ.get("COMFY_SHIFT")
        os.environ["COMFY_SHIFT"] = "4.0"
        try:
            out = bc._patch_graph(self._moss_graph(), "x", 512, 512, None, "m")
            self.assertEqual(out["35"]["inputs"]["shift"], 4.0)
        finally:
            if saved is None:
                os.environ.pop("COMFY_SHIFT", None)
            else:
                os.environ["COMFY_SHIFT"] = saved

    def test_prompt_guard_flags_rgba_wrapper(self):
        warning = bc._moss_prompt_warning(
            "This is an RGBA format image with transparency. A bulb. "
            "The image has an alpha channel and a transparent background.")
        self.assertIsNotNone(warning)
        self.assertIn("moss-image-prompt-enhancer", warning)

    def test_prompt_guard_accepts_caption_and_layout(self):
        caption = ("A calm editorial photograph of a ceramic cup on an ivory "
                   "table, soft side lighting, shallow depth of field.")
        self.assertIsNone(bc._moss_prompt_warning(caption))
        layout = ('{"summary":{"caption":"poster"},"items":[{"label":"text",'
                  '"caption":"title","text":"Hi","bbox":[0,0,0,0]}],'
                  '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}')
        self.assertIsNone(bc._moss_prompt_warning(layout))

    def test_prompt_guard_flags_empty_and_short(self):
        self.assertIsNotNone(bc._moss_prompt_warning(""))
        self.assertIsNotNone(bc._moss_prompt_warning("a cat"))


if __name__ == "__main__":
    unittest.main()
