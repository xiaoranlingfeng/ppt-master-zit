#!/usr/bin/env python3
"""
Local ComfyUI image generation backend (Qwen-Image-2.1 / MOSS-Image SFT /
Z-Image Turbo t2i workflows).

Runs an exported text-to-image workflow against a locally running ComfyUI
server via its HTTP API. No API key is required. Three model families ship:

  qwen (default)  Qwen-Image-2.1 — 7B single-stream DiT + Qwen3-VL 8B int8
                  encoder, native RGBA, cfg 1.
  moss            MOSS-Image SFT — Qwen3.5 9B encoder + Flux2-small VAE,
                  steps 50 / cfg 3 / euler / sgm_uniform / shift 5, opaque
                  RGB. Prompts MUST be authored with the user's
                  moss-image-prompt-enhancer skill (see references/
                  image-generator.md §4.5); the backend warns when the
                  prompt does not look like a MOSS caption / Layout.
  zit             Z-Image Turbo "zit基础" — fast, Apache-2.0, opaque RGB.

Selection order (first hit wins):
  COMFY_MODEL          (optional; qwen | moss | zit, or a template path)
  COMFY_WORKFLOW       (optional; path to an API-format workflow JSON)
  built-in default     qwen_image_2_1_t2i_api.json

Configuration keys:
  COMFY_BASE_URL       (optional; default http://127.0.0.1:8188)
  COMFY_MODEL          (optional; named model family, see above)
  COMFY_WORKFLOW       (optional; path to an API-format workflow JSON.
                        Default: <skill>/templates/comfyui/
                        qwen_image_2_1_t2i_api.json)
  COMFY_NEGATIVE_PROMPT (optional; overrides the template's negative prompt.
                        Ignored while cfg stays at 1 — the Qwen-Image-2.1
                        official path.)
  COMFY_FILENAME_PREFIX (optional; SaveImage prefix, default "ppt_master")
  COMFY_TIMEOUT        (optional; seconds to wait per image, default 900)
  COMFY_TEXT_RESOLUTION (optional; pins the Qwen-Image-2.1 encoder
                        `resolution` hint. Unset re-derives it from the
                        requested size: the shipped template keeps 1K at its
                        own 1024 and follows the tier at 2K / 4K.)

Auto-start keys (launch the local ComfyUI when 127.0.0.1:8188 is down):
  COMFY_AUTOSTART      (optional; "1" enable / "0" disable, default "1")
  COMFY_OPEN_BROWSER   (optional; "1" open the ComfyUI web UI in the default
                        browser after an auto-start so the user can watch the
                        queue, "0" stay headless; default "1")
  COMFY_ROOT           (optional; aki-v3 integration root, e.g.
                        ~/Downloads/ComfyUI-aki-v3. Autodetected
                        from a candidate list when unset.)
  COMFY_LAUNCH_CMD     (optional; full custom launch command with {port}
                        placeholder, overrides COMFY_ROOT autodetection)
  COMFY_START_TIMEOUT  (optional; seconds to wait for server boot, default 300)

Sampler overrides (Qwen-Image-2.1 official path: steps 25 / cfg 1 /
euler / simple; MOSS-Image SFT: steps 50 / cfg 3 / euler / sgm_uniform /
shift 5; Z-Image Turbo: steps 8 / cfg 1 / res_multistep / simple / shift 3):
  COMFY_STEPS          (optional; KSampler steps)
  COMFY_CFG            (optional; KSampler cfg)
  COMFY_SAMPLER        (optional; sampler_name, e.g. euler)
  COMFY_SCHEDULER      (optional; scheduler, e.g. simple)
  COMFY_SHIFT          (optional; ModelSamplingSD3 / ModelSamplingAuraFlow
                        shift — MOSS and Z-Image)

The template's placeholder nodes are patched per request:
  - the positive CLIPTextEncode (text) or TextEncodeQwenImage21 (prompt)
    node carrying "{{POSITIVE_PROMPT}}" receives the prompt;
  - EmptySD3LatentImage / EmptyLatentImage receives width/height computed
    from aspect_ratio and image_size (nearest multiple of 32, per the
    Qwen-Image-2.1 usage note; also satisfies Z-Image's multiple-of-16);
  - KSampler gets a fresh random seed each call.

All HTTP calls use a direct session (trust_env=False) so a system proxy
(HTTP_PROXY/HTTPS_PROXY) never intercepts the localhost ComfyUI traffic.
"""

import json
import math
import random
import shlex
import stat as _stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from console_encoding import configure_utf8_stdio  # noqa: E402

configure_utf8_stdio()

import os

import requests
from requests.adapters import HTTPAdapter

# 本机直连会话：绕过系统代理（避免 HTTP_PROXY/HTTPS_PROXY 劫持 127.0.0.1:8188），
# 并配置连接池复用 TCP 长连接以支撑 manifest 并发提交。
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.proxies = {"http": None, "https": None}
_HTTP_ADAPTER = HTTPAdapter(pool_connections=4, pool_maxsize=8)
_SESSION.mount("http://", _HTTP_ADAPTER)
_SESSION.mount("https://", _HTTP_ADAPTER)

from image_backends.backend_common import (
    MAX_RETRIES,
    is_permanent_error,
    is_rate_limit_error,
    normalize_image_size,
    parse_aspect_ratio,
    resolve_output_path,
    retry_delay,
    save_image_bytes,
)

# 声明本地后端特性，供 image_gen.py 启用滑动窗口流水线调度并跳过云端批次休眠。
IS_LOCAL_BACKEND = True
INTER_BATCH_PAUSE_SECONDS = 0.0

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_COMFY_TEMPLATE_DIR = _SCRIPTS_DIR.parent / "templates" / "comfyui"
DEFAULT_TEMPLATE = _COMFY_TEMPLATE_DIR / "qwen_image_2_1_t2i_api.json"
PROMPT_PLACEHOLDER = "{{POSITIVE_PROMPT}}"
# Qwen-Image-2.1 measures ~80 s per 2K image on a mobile 3080 and the local GPU
# queue can sit behind another client, so the old 600 s cap killed live runs.
DEFAULT_IMAGE_TIMEOUT = 900
_WIN_ERROR_ACCESS_DENIED = 5
_READY_TTL_SECONDS = 15.0
_READY_CACHE: dict[str, float] = {}
_READY_LOCK = threading.Lock()
_AUTOSTART_LOCK = threading.Lock()
_TEMPLATE_CACHE: dict[tuple[Path, int, int], dict] = {}
_TEMPLATE_CACHE_LOCK = threading.Lock()

# Named local model families. "template" is the API-format workflow shipped
# under templates/comfyui/; "prompt_style" tells the deck pipeline which
# prompting manual owns this model's prompts.
MODELS = {
    "qwen": {
        "template": "qwen_image_2_1_t2i_api.json",
        "label": "Qwen-Image-2.1",
        "prompt_style": "qwen",
        "rgba": True,
        "min_dimension": 32,
        "dimension_step": 32,
        "aliases": ("qwen-image", "qwen_image", "qwen-image-2.1", "qwen_image_2_1"),
    },
    "moss": {
        "template": "moss_image_sft_api.json",
        "label": "MOSS-Image SFT",
        # Prompts for this model are authored by the moss-image-prompt-enhancer
        # skill (a sibling skill), never written ad hoc by the deck pipeline.
        "prompt_style": "moss",
        "rgba": False,
        "min_dimension": 64,
        "dimension_step": 16,
        "aliases": ("moss-image", "moss_image", "moss-image-sft", "moss_image_sft",
                    "moss-sft"),
    },
    "zit": {
        "template": "zit_basic_api.json",
        "label": "Z-Image Turbo (zit基础)",
        "prompt_style": "qwen",
        "rgba": False,
        "min_dimension": 32,
        "dimension_step": 32,
        "aliases": ("z-image", "z_image", "z-image-turbo", "zit-basic", "zit基础"),
    },
}
_MODEL_ALIASES = {
    alias: name for name, cfg in MODELS.items() for alias in (*cfg["aliases"], name)
}

# image_size token -> target megapixels (mirrors the other backends' 512px/1K/2K/4K)
IMAGE_SIZE_TO_MEGAPIXELS = {
    "512px": 0.25,
    "1K": 1.0,
    "2K": 4.0,
    "4K": 16.0,
}
# ALL_ASPECT_RATIOS from image_gen.py, as W/H floats
ASPECT_RATIO_MAP = {
    "1:1": 1 / 1, "1:2": 1 / 2, "1:3": 1 / 3, "1:4": 1 / 4, "1:8": 1 / 8,
    "2:1": 2 / 1, "2:3": 2 / 3, "3:1": 3 / 1, "3:2": 3 / 2, "3:4": 3 / 4,
    "4:1": 4 / 1, "4:3": 4 / 3, "4:5": 4 / 5, "5:4": 5 / 4, "8:1": 8 / 1,
    "9:16": 9 / 16, "9:21": 9 / 21, "10:16": 10 / 16,
    "16:9": 16 / 9, "16:10": 16 / 10, "21:9": 21 / 9,
}

# Known aki-v3 integration roots (folder containing python\python.exe and
# ComfyUI\main.py). COMFY_ROOT overrides; COMFY_LAUNCH_CMD overrides both.
_AKI_ROOT_CANDIDATES = (
    Path.home() / "Downloads" / "ComfyUI-aki-v3",
    Path.home() / "ComfyUI-aki-v3",
    Path("D:/ComfyUI-aki-v3"),
    Path("C:/ComfyUI-aki-v3"),
    Path.home() / "ComfyUI",
    Path("C:/ComfyUI"),
    Path("D:/ComfyUI"),
)


def _server_ready(base_url: str, timeout: float = 3.0) -> bool:
    """True when the local ComfyUI HTTP API answers /system_stats."""
    try:
        resp = _SESSION.get(f"{base_url}/system_stats", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _open_browser(base_url: str) -> None:
    """Open the ComfyUI web UI so the user can watch the queue.

    Only called after an auto-start (a server the user launched manually
    already has its own UI). Best-effort: failures never block generation.
    """
    if os.environ.get("COMFY_OPEN_BROWSER", "1").strip().lower() in {"0", "false", "no"}:
        return
    try:
        import webbrowser
        webbrowser.open(base_url)
        print("  [..] Opened the ComfyUI web UI in your default browser.", flush=True)
    except Exception:
        pass


def _resolve_launch_plan(base_url: str) -> tuple[list[str], Path]:
    """Return (command tokens, working directory) for auto-starting ComfyUI."""
    port = urlparse(base_url).port or 8188
    custom = os.environ.get("COMFY_LAUNCH_CMD", "").strip()
    if custom:
        tokens = shlex.split(custom.replace("{port}", str(port)), posix=False)
        return [t.strip('"').strip("'") for t in tokens], Path.cwd()
    root_env = os.environ.get("COMFY_ROOT", "").strip()
    candidates = [Path(root_env)] if root_env else [Path(p) for p in _AKI_ROOT_CANDIDATES]
    for root in candidates:
        python_exe = root / "python" / "python.exe"
        main_py = root / "ComfyUI" / "main.py"
        if python_exe.is_file() and main_py.is_file():
            return (
                [str(python_exe), "-s", str(main_py), "--port", str(port)],
                main_py.parent,
            )
    raise RuntimeError(
        "Cannot auto-start ComfyUI: no aki-v3 integration found. Set COMFY_ROOT "
        "to the integration root (the folder containing 'python\\python.exe' and "
        "'ComfyUI\\main.py') or COMFY_LAUNCH_CMD to a custom launch command "
        "containing a {port} placeholder. Alternatively start ComfyUI manually "
        "(e.g. the aki launcher) and retry."
    )


def _launch_comfyui(base_url: str) -> None:
    """Start the local ComfyUI detached from this process, logging to a file."""
    cmd, cwd = _resolve_launch_plan(base_url)
    log_path = cwd / "comfyui_autostart.log"
    print(f"  [..] ComfyUI not running — auto-starting it (log: {log_path})...", flush=True)
    with open(log_path, "ab") as log:
        popen_kwargs = dict(
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        if sys.platform == "win32":
            detached = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
            # 对齐上游 server_common.popen_detached：仅在 Windows Job Object
            # 拒绝 breakaway（ERROR_ACCESS_DENIED=5）时回退普通后台启动并告警，
            # 避免掩盖可执行文件缺失等真实启动错误。
            try:
                subprocess.Popen(
                    cmd, creationflags=detached | breakaway,
                    **popen_kwargs,
                )
                return
            except OSError as exc:
                if getattr(exc, "winerror", None) != _WIN_ERROR_ACCESS_DENIED:
                    raise
                print(
                    f"  [WARN] Windows Job Object 拒绝进程 breakaway ({exc})，已回退为普通后台启动；"
                    "若当前命令结束后 ComfyUI 随之退出，请手动启动 ComfyUI 后重试。",
                    flush=True,
                )
                subprocess.Popen(cmd, creationflags=detached, **popen_kwargs)
                return
        subprocess.Popen(cmd, **popen_kwargs)


def _mark_server_ready(base_url: str) -> None:
    """记录指定地址最近一次健康探活时间，减少批量出图时的重复探测。"""
    with _READY_LOCK:
        _READY_CACHE[base_url] = time.monotonic()


def _invalidate_ready_cache(base_url: str) -> None:
    """清除指定地址的健康缓存，确保后续请求重新探活或触发自动拉起。"""
    with _READY_LOCK:
        _READY_CACHE.pop(base_url, None)


def _ensure_server_running(base_url: str) -> None:
    """确保 ComfyUI 已就绪；批量并发场景下通过互斥锁避免重复拉起多个实例。"""
    with _READY_LOCK:
        last_ok = _READY_CACHE.get(base_url)
    if last_ok is not None and (time.monotonic() - last_ok) < _READY_TTL_SECONDS:
        return
    if _server_ready(base_url):
        _mark_server_ready(base_url)
        return
    with _AUTOSTART_LOCK:
        # 二次检查，防止并发工作线程同时触发自动拉起
        with _READY_LOCK:
            last_ok = _READY_CACHE.get(base_url)
        if last_ok is not None and (time.monotonic() - last_ok) < _READY_TTL_SECONDS:
            return
        if _server_ready(base_url):
            _mark_server_ready(base_url)
            return
        if os.environ.get("COMFY_AUTOSTART", "1").strip().lower() in {"0", "false", "no"}:
            raise ConnectionError(
                f"ComfyUI is not reachable at {base_url} and COMFY_AUTOSTART=0. "
                "Start the local ComfyUI manually and retry."
            )
        _launch_comfyui(base_url)
        start_timeout = int(os.environ.get("COMFY_START_TIMEOUT", "300"))
        deadline = time.time() + start_timeout
        poll_delay = 0.5
        while time.time() < deadline:
            if _server_ready(base_url, timeout=5.0):
                _mark_server_ready(base_url)
                print("  [OK] ComfyUI is up.", flush=True)
                _open_browser(base_url)
                return
            time.sleep(poll_delay)
            poll_delay = min(2.0, poll_delay * 1.5)
        raise ConnectionError(
            f"ComfyUI did not become ready at {base_url} within {start_timeout}s. "
            "Check comfyui_autostart.log under the integration root, or start it "
            "manually with the aki launcher and retry."
        )


def _apply_dimension_rule(width: int, height: int, model: str | None = None) -> tuple[int, int]:
    """Snap width/height to the model family's minimum and alignment.

    Qwen-Image-2.1 (and the Z-Image Turbo template, which loads the same VAE
    stack) decode at multiples of 32. MOSS-Image only requires multiples of
    16, but its 16px latents combined with 4x patching break down below 64px.
    """
    cfg = MODELS.get(model or "qwen", MODELS["qwen"])
    step = int(cfg.get("dimension_step") or 32)
    minimum = max(int(cfg.get("min_dimension") or 64), step)

    def snap(value: int) -> int:
        value = int(round(value / step)) * step
        return max(minimum, value)

    return snap(width), snap(height)


def _resolve_dimensions(aspect_ratio: str, image_size: str,
                        model: str | None = None) -> tuple[int, int]:
    """按目标宽高比与尺寸档位计算模型对齐后的宽高（支持任意正数 W:H）。"""
    ratio = ASPECT_RATIO_MAP.get(aspect_ratio)
    if ratio is None:
        try:
            ratio = float(parse_aspect_ratio(aspect_ratio))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported aspect ratio '{aspect_ratio}' for comfyui backend. "
                f"Expected positive W:H (e.g. 16:9, 1.85:1); preset ratios: {sorted(ASPECT_RATIO_MAP)}"
            ) from exc
        if ratio < 1 / 16 or ratio > 16:
            raise ValueError(
                f"Aspect ratio '{aspect_ratio}' is out of supported range (1:16 to 16:1) "
                "for comfyui backend."
            )
    normalized_size = normalize_image_size(image_size)
    megapixels = IMAGE_SIZE_TO_MEGAPIXELS.get(normalized_size)
    if megapixels is None:
        raise ValueError(
            f"Unsupported image size '{image_size}' for comfyui backend. "
            f"Supported: {list(IMAGE_SIZE_TO_MEGAPIXELS)}"
        )
    env_mp = os.environ.get("COMFY_MEGAPIXELS", "").strip()
    if env_mp:
        try:
            megapixels = float(env_mp)
        except ValueError:
            raise ValueError(f"COMFY_MEGAPIXELS is not a number: {env_mp!r}")
    total = megapixels * 1_000_000
    width = math.sqrt(total * ratio)
    height = width / ratio
    return _apply_dimension_rule(width, height, model)


# The Qwen-Image-2.1 encoder node carries a `resolution` conditioning hint.
# The shipped template pins it at the 1K tier (1024); leaving it there while the
# latent grows to 2K/4K conditions the text encoder on a quarter of the real
# area. Only re-derive when the tier actually moved, so the measured 1K path
# keeps sending exactly the values it shipped with.
_TEXT_RESOLUTION_TRIGGER_RATIO = 0.2


def _resolve_text_resolution(width: int, height: int, current,
                             model: str | None = None) -> int:
    """Return the encoder `resolution` hint for the requested output size."""
    raw = os.environ.get("COMFY_TEXT_RESOLUTION", "").strip()
    if raw:
        try:
            return max(64, int(raw))
        except ValueError:
            raise ValueError(f"COMFY_TEXT_RESOLUTION is not a number: {raw!r}")
    step = int(MODELS.get(model or "qwen", MODELS["qwen"]).get("dimension_step") or 32)
    equivalent = max(2 * step, int(round(math.sqrt(width * height) / step)) * step)
    if not isinstance(current, int) or current <= 0:
        return equivalent
    if abs(equivalent - current) / current <= _TEXT_RESOLUTION_TRIGGER_RATIO:
        return current
    return equivalent


def _resolve_model_name(model: str | None = None) -> str | None:
    """Map a named model / shipped template path to a MODELS key.

    Precedence: explicit ``model`` argument, COMFY_MODEL, then a COMFY_WORKFLOW
    whose filename matches a shipped template. Returns None for custom
    templates and unknown names.
    """
    for raw in (model, os.environ.get("COMFY_MODEL", "")):
        value = (raw or "").strip()
        if not value:
            continue
        key = _MODEL_ALIASES.get(value.lower())
        if key:
            return key
        for name, cfg in MODELS.items():
            if Path(value).name.lower() == cfg["template"].lower():
                return name
        return None
    workflow = os.environ.get("COMFY_WORKFLOW", "").strip()
    if workflow:
        name = Path(workflow).name.lower()
        for key, cfg in MODELS.items():
            if name == cfg["template"].lower():
                return key
    return None


def _resolve_template(model: str | None = None) -> tuple[Path, str | None]:
    """Return (template path, MODELS key) for the requested/configured model."""
    explicit = (model or "").strip()
    if explicit and explicit.lower() not in _MODEL_ALIASES:
        # A real path or bare filename, not a model name.
        known = _resolve_model_name(explicit)
        if known:
            return _COMFY_TEMPLATE_DIR / MODELS[known]["template"], known
        return Path(explicit).expanduser(), None
    key = _resolve_model_name(explicit or None)
    if key:
        return _COMFY_TEMPLATE_DIR / MODELS[key]["template"], key
    template_env = os.environ.get("COMFY_WORKFLOW", "").strip()
    if template_env:
        return Path(template_env).expanduser(), None
    return DEFAULT_TEMPLATE, "qwen"


def _clone_graph(graph: dict) -> dict:
    """快速深拷贝 ComfyUI API 工作流图，避免 json 往返序列化开销。"""
    cloned: dict = {}
    for nid, node in graph.items():
        if not isinstance(node, dict):
            cloned[nid] = node
            continue
        node_copy = dict(node)
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            inputs_copy = {}
            for k, v in inputs.items():
                if isinstance(v, list):
                    inputs_copy[k] = list(v)
                elif isinstance(v, dict):
                    inputs_copy[k] = dict(v)
                else:
                    inputs_copy[k] = v
            node_copy["inputs"] = inputs_copy
        cloned[nid] = node_copy
    return cloned


def _load_template(template_path: Path) -> dict:
    """加载并按文件修改时间缓存 ComfyUI API 工作流模板，返回独立副本。"""
    key_path = os.path.abspath(template_path)
    try:
        st = os.stat(key_path)
    except OSError as exc:
        raise FileNotFoundError(
            f"ComfyUI workflow template not found: {template_path}\n"
            "Set COMFY_MODEL to qwen / moss / zit, or COMFY_WORKFLOW to an "
            "API-format workflow JSON exported from ComfyUI (preinstalled: "
            "templates/comfyui/qwen_image_2_1_t2i_api.json, "
            "moss_image_sft_api.json, zit_basic_api.json)."
        ) from exc
    if not _stat.S_ISREG(st.st_mode):
        raise FileNotFoundError(
            f"ComfyUI workflow template not found: {template_path}\n"
            "Set COMFY_MODEL to qwen / moss / zit, or COMFY_WORKFLOW to an "
            "API-format workflow JSON exported from ComfyUI (preinstalled: "
            "templates/comfyui/qwen_image_2_1_t2i_api.json, "
            "moss_image_sft_api.json, zit_basic_api.json)."
        )
    cache_key = (key_path, st.st_mtime_ns, st.st_size)
    with _TEMPLATE_CACHE_LOCK:
        cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return _clone_graph(cached)
    graph = json.loads(Path(key_path).read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not graph:
        raise ValueError(f"Workflow template must be a non-empty JSON object: {template_path}")
    with _TEMPLATE_CACHE_LOCK:
        # 清理同一路径的旧版本缓存项，防止内存累积
        stale = [k for k in _TEMPLATE_CACHE if k[0] == key_path]
        for k in stale:
            _TEMPLATE_CACHE.pop(k, None)
        _TEMPLATE_CACHE[cache_key] = graph
    return _clone_graph(graph)


# Latent nodes whose width/height are patched; Flux2 latents (MOSS-Image)
# are included so a MOSS graph receives the requested size.
_SIZE_NODE_TYPES = ("EmptyFlux2LatentImage", "EmptySD3LatentImage", "EmptyLatentImage")


def _find_nodes(graph: dict, class_type: str) -> list[str]:
    return [nid for nid, node in graph.items() if node.get("class_type") == class_type]


_OVERRIDE_WARNINGS_EMITTED: set[tuple[str, str]] = set()

_OVERRIDE_ENV_NAMES = {
    "steps": "COMFY_STEPS",
    "cfg": "COMFY_CFG",
    "sampler_name": "COMFY_SAMPLER",
    "scheduler": "COMFY_SCHEDULER",
    "shift": "COMFY_SHIFT",
}


def _override_matches_template(key: str, raw: str, template_value) -> bool:
    """True when an env override restates the template's own value."""
    if template_value is None:
        return True
    if key in ("steps", "cfg", "shift"):
        try:
            return math.isclose(float(raw), float(template_value), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return str(raw).strip().lower() == str(template_value).strip().lower()


def _warn_override_retunes_model(
    model: str | None, key: str, raw: str, template_value
) -> None:
    """Warn once when a global sampler override retunes a model.

    COMFY_STEPS / COMFY_CFG / COMFY_SAMPLER / COMFY_SCHEDULER / COMFY_SHIFT are
    model-agnostic. A value kept from one model (typically Qwen's 25 / 1 /
    euler / simple) silently rewrites another model's official sampler path,
    so switching COMFY_MODEL produced degraded MOSS or Z-Image runs with no
    sign of why.
    """
    if model is None or _override_matches_template(key, raw, template_value):
        return
    token = (model, key)
    if token in _OVERRIDE_WARNINGS_EMITTED:
        return
    _OVERRIDE_WARNINGS_EMITTED.add(token)
    label = MODELS.get(model, {}).get("label", model)
    env_name = _OVERRIDE_ENV_NAMES.get(key, key.upper())
    print(
        f"  WARNING: {env_name}={raw} overrides the {label} template default "
        f"{template_value} — sampler overrides apply to every local model; "
        f"clear it in .env when switching COMFY_MODEL."
    )


def _patch_graph(
    graph: dict,
    prompt: str,
    width: int,
    height: int,
    negative_prompt: str | None,
    filename_prefix: str,
    *,
    model: str | None = None,
) -> dict:
    """Return a per-run copy of the template with prompt/size/seed injected."""
    graph = _clone_graph(graph)

    # Positive-prompt carrier: a CLIPTextEncode whose text (Z-Image style
    # templates) or a TextEncodeQwenImage21 whose prompt (Qwen-Image-2.1,
    # which encodes positive+negative in one node) equals the placeholder.
    clip_positives = [
        nid for nid, node in graph.items()
        if node.get("class_type") == "CLIPTextEncode"
        and node.get("inputs", {}).get("text") == PROMPT_PLACEHOLDER
    ]
    qwen_positives = [
        nid for nid, node in graph.items()
        if node.get("class_type") == "TextEncodeQwenImage21"
        and node.get("inputs", {}).get("prompt") == PROMPT_PLACEHOLDER
    ]
    if not clip_positives and not qwen_positives:
        raise ValueError(
            "Workflow template is missing the positive-prompt placeholder node "
            f"(a CLIPTextEncode text or TextEncodeQwenImage21 prompt equal to "
            f"{PROMPT_PLACEHOLDER!r}). Use a shipped template under "
            "templates/comfyui/ or add the placeholder manually."
        )
    for nid in clip_positives + qwen_positives:
        node = graph[nid]
        if node.get("class_type") == "CLIPTextEncode":
            node["inputs"]["text"] = prompt
        else:
            node["inputs"]["prompt"] = prompt
            if negative_prompt:
                node["inputs"]["negative_prompt"] = negative_prompt

    # Negative-prompt scope note: the Qwen-Image-2.1 default template carries
    # positive+negative inside one TextEncodeQwenImage21 node and runs cfg=1,
    # where the sampler ignores the negative branch entirely (kept for API
    # completeness). The CLIPTextEncode sweep below only bites for Z-Image
    # Turbo / custom templates that wire a second negative encoder node.
    if negative_prompt:
        negatives = [
            nid for nid in _find_nodes(graph, "CLIPTextEncode")
            if nid not in clip_positives
        ]
        for nid in negatives:
            graph[nid]["inputs"]["text"] = negative_prompt

    size_nodes = [nid for t in _SIZE_NODE_TYPES for nid in _find_nodes(graph, t)]
    if not size_nodes:
        raise ValueError(
            "Workflow template has no empty-latent node (EmptyLatentImage / "
            "EmptySD3LatentImage / EmptyFlux2LatentImage) to receive the "
            "requested size."
        )
    for nid in size_nodes:
        graph[nid]["inputs"]["width"] = width
        graph[nid]["inputs"]["height"] = height
        graph[nid]["inputs"]["batch_size"] = 1

    # Encoder resolution hint follows the size tier (Qwen-Image-2.1 only; the
    # other templates carry no such input).
    for nid in _find_nodes(graph, "TextEncodeQwenImage21"):
        inputs = graph[nid]["inputs"]
        if "resolution" in inputs:
            inputs["resolution"] = _resolve_text_resolution(
                width, height, inputs["resolution"], model)

    # Optional sampler overrides (template defaults: Qwen-Image-2.1 steps 25,
    # cfg 1, euler, simple; MOSS-Image SFT steps 50, cfg 3, euler,
    # sgm_uniform; Z-Image Turbo steps 8, cfg 1, res_multistep, simple).
    _int_overrides = {
        "steps": os.environ.get("COMFY_STEPS", "").strip(),
    }
    _float_overrides = {
        "cfg": os.environ.get("COMFY_CFG", "").strip(),
    }
    _raw_overrides = {
        "sampler_name": os.environ.get("COMFY_SAMPLER", "").strip(),
        "scheduler": os.environ.get("COMFY_SCHEDULER", "").strip(),
    }
    for nid in _find_nodes(graph, "KSampler"):
        inputs = graph[nid]["inputs"]
        inputs["seed"] = random.randint(0, 2 ** 53)
        inputs.pop("control_after_generate", None)
        for key, raw in _int_overrides.items():
            if raw:
                _warn_override_retunes_model(model, key, raw, inputs.get(key))
                inputs[key] = int(raw)
        for key, raw in _float_overrides.items():
            if raw:
                _warn_override_retunes_model(model, key, raw, inputs.get(key))
                inputs[key] = float(raw)
        for key, raw in _raw_overrides.items():
            if raw:
                _warn_override_retunes_model(model, key, raw, inputs.get(key))
                inputs[key] = raw
    shift_raw = os.environ.get("COMFY_SHIFT", "").strip()
    if shift_raw:
        # ModelSamplingSD3 carries MOSS-Image's shift (default 5.0);
        # ModelSamplingAuraFlow carries Z-Image Turbo's (default 3).
        for class_type in ("ModelSamplingSD3", "ModelSamplingAuraFlow"):
            for nid in _find_nodes(graph, class_type):
                _warn_override_retunes_model(
                    model, "shift", shift_raw,
                    graph[nid]["inputs"].get("shift"))
                graph[nid]["inputs"]["shift"] = float(shift_raw)

    for nid in _find_nodes(graph, "SaveImage"):
        graph[nid]["inputs"]["filename_prefix"] = filename_prefix

    return graph


class _ComfyRunFailed(RuntimeError):
    """Server executed the graph and reported a node failure."""


def _ws_enabled() -> bool:
    """判断是否启用 WebSocket 实时事件监听（需安装可选依赖 websocket-client）。"""
    return os.environ.get("COMFY_USE_WS", "1").strip().lower() not in {"0", "false", "no"}


def _try_open_ws(base_url: str, client_id: str):
    """尝试在提交任务前建立 ComfyUI WebSocket 连接，不可用时静默返回 None。"""
    if not _ws_enabled():
        return None
    try:
        import websocket  # type: ignore[import-not-found]
    except ImportError:
        return None
    parsed = urlparse(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/ws?clientId={client_id}"
    try:
        return websocket.create_connection(ws_url, timeout=3.0, http_proxy_host=None)
    except Exception:
        return None


def _check_history_entry(base_url: str, prompt_id: str) -> dict | None:
    """查询单次 /history/{prompt_id}，若已完成返回条目，失败则抛 _ComfyRunFailed。"""
    resp = _SESSION.get(f"{base_url}/history/{prompt_id}", timeout=30)
    if resp.status_code != 200:
        return None
    entry = resp.json().get(prompt_id)
    if not entry:
        return None
    status = entry.get("status") or {}
    if status.get("status_str") == "error" or status.get("completed") is False:
        details = []
        for msg in status.get("messages") or []:
            if isinstance(msg, list) and len(msg) > 1 and isinstance(msg[1], dict):
                details.append(
                    f"node {msg[1].get('node_id')} ({msg[1].get('node_type')}): "
                    f"{msg[1].get('exception_message')}"
                )
        raise _ComfyRunFailed(
            "ComfyUI execution failed"
            + (f" — {'; '.join(details)[:800]}" if details else "")
        )
    if entry.get("outputs"):
        return entry
    return None


def _wait_via_ws(ws, base_url: str, prompt_id: str, deadline: float) -> dict | None:
    """通过 WebSocket 实时等待执行完成；遇到网络异常返回 None 交由 HTTP 轮询接管。"""
    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            ws.settimeout(min(5.0, remaining))
            try:
                raw = ws.recv()
            except TimeoutError:
                continue
            except Exception as exc:
                if "timed out" in str(exc).lower():
                    continue
                return None
            if not isinstance(raw, str):
                # 忽略二进制预览帧
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type")
            data = msg.get("data") or {}
            if data.get("prompt_id") != prompt_id:
                continue
            if msg_type == "execution_error":
                detail = (
                    f"node {data.get('node_id')} ({data.get('node_type')}): "
                    f"{data.get('exception_message')}"
                )
                raise _ComfyRunFailed(f"ComfyUI execution failed — {detail[:800]}")
            if msg_type == "execution_success" or (
                msg_type == "executing" and data.get("node") is None
            ):
                # ComfyUI 触发完成事件后立即拉取一次历史记录，实现近零延迟返回
                for _ in range(5):
                    entry = _check_history_entry(base_url, prompt_id)
                    if entry is not None:
                        return entry
                    time.sleep(0.1)
                return None
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return None


def _resolve_poll_schedule() -> tuple[float, float]:
    """返回 HTTP 轮询的 (初始间隔, 最大间隔)，支持通过 COMFY_POLL_INTERVAL 覆盖。"""
    raw = os.environ.get("COMFY_POLL_INTERVAL", "").strip()
    if raw:
        try:
            val = max(0.1, float(raw))
            return val, val
        except ValueError:
            pass
    return 0.3, 1.2


def _wait_for_completion(
    base_url: str,
    prompt_id: str,
    timeout: int,
    *,
    ws=None,
) -> dict:
    """等待 ComfyUI 任务完成：优先 WebSocket 实时监听，不可用时自动退避 HTTP 轮询。"""
    deadline = time.time() + timeout
    if ws is not None:
        entry = _wait_via_ws(ws, base_url, prompt_id, deadline)
        if entry is not None:
            return entry

    poll_delay, max_delay = _resolve_poll_schedule()
    while time.time() < deadline:
        try:
            entry = _check_history_entry(base_url, prompt_id)
        except requests.RequestException:
            time.sleep(min(2.0, max_delay))
            continue
        if entry is not None:
            return entry
        time.sleep(poll_delay)
        poll_delay = min(max_delay, poll_delay * 1.4)
    raise RuntimeError(
        f"ComfyUI run {prompt_id} did not finish within {timeout}s "
        "(is the GPU busy with another queue?). Raise COMFY_TIMEOUT or clear "
        "the queue and retry."
    )


def _collect_and_save(base_url: str, entry: dict, output_dir: str, filename: str) -> str:
    """下载首个输出图像并按调用方文件名/扩展名直接落盘，避免二次转码。"""
    for out in entry.get("outputs", {}).values():
        for img in out.get("images", []) or []:
            if img.get("type") != "output":
                continue
            resp = _SESSION.get(
                f"{base_url}/view",
                params={
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                },
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"ComfyUI /view failed ({resp.status_code}) for {img['filename']}"
                )
            ext = Path(img["filename"]).suffix.lower() or ".png"
            target_name = filename or f"comfy_{int(time.time())}"
            path = resolve_output_path(
                prompt="",
                output_dir=output_dir or None,
                filename=target_name,
                ext=ext,
            )
            return save_image_bytes(resp.content, path,
                                    content_type=resp.headers.get("content-type"))
    raise RuntimeError("ComfyUI run finished but produced no output images")


def _moss_prompt_warning(prompt: str) -> str | None:
    """Return a warning when a MOSS prompt was not authored for MOSS-Image.

    Prompts for MOSS-Image SFT must come from the sibling
    ``moss-image-prompt-enhancer`` skill: a cohesive English caption
    (``{"caption": "..."}``) or a compiled Layout (one JSON line with
    ``abs_centralized`` / ``yxyx`` boxes). The RGBA wrapper used by the
    Qwen-Image-2.1 path is meaningless here — MOSS returns opaque RGB — and
    tag-soup or empty prompts degrade badly at cfg 3.
    """
    text = (prompt or "").strip()
    if not text:
        return "The prompt is empty; MOSS-Image needs a caption or Layout prompt."
    if "RGBA" in text or "alpha channel" in text:
        return ("The prompt carries the Qwen-Image-2.1 RGBA wrapper, but MOSS-Image "
                "SFT returns opaque RGB. Author the prompt with the "
                "moss-image-prompt-enhancer skill instead.")
    looks_like_layout = (
        text.startswith("{")
        and '"items"' in text
        and '"abs_centralized"' in text
    )
    if looks_like_layout:
        return None
    if len(text) < 40:
        return ("The prompt is very short; MOSS-Image expects a developed caption "
                "from the moss-image-prompt-enhancer skill.")
    return None


def _generate_image(base_url: str, template_path: Path, prompt: str,
                    aspect_ratio: str, image_size: str,
                    output_dir: str = None, filename: str = None,
                    model: str | None = None) -> str:
    width, height = _resolve_dimensions(aspect_ratio, image_size, model)
    graph = _patch_graph(
        _load_template(template_path),
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=os.environ.get("COMFY_NEGATIVE_PROMPT"),
        filename_prefix=os.environ.get("COMFY_FILENAME_PREFIX", "ppt_master"),
        model=model,
    )
    client_id = f"ppt-master-{random.randint(0, 2**31)}"

    print("[comfyui local]")
    print(f"  Server:       {base_url}")
    print(f"  Model:        {(MODELS.get(model) or {}).get('label', 'custom template')}")
    print(f"  Template:     {template_path}")
    print(f"  Prompt:       {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"  Size:         {width}x{height}")
    if model == "moss":
        print("  Prompt style: MOSS caption/Layout — author with the "
              "moss-image-prompt-enhancer skill")
        warning = _moss_prompt_warning(prompt)
        if warning:
            print(f"  [WARN] {warning}", flush=True)
    print()
    _ensure_server_running(base_url)
    # 在 POST /prompt 前建立 WebSocket（若可用），防止极快任务漏掉完成事件
    ws = _try_open_ws(base_url, client_id)
    print("  [..] Submitting to ComfyUI queue...", flush=True)
    start = time.time()
    try:
        resp = _SESSION.post(
            f"{base_url}/prompt",
            json={"prompt": graph, "client_id": client_id},
            timeout=30,
        )
    except requests.ConnectionError as exc:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        _invalidate_ready_cache(base_url)
        raise ConnectionError(
            f"Cannot reach ComfyUI at {base_url}. Start the local ComfyUI "
            "(e.g. the aki launcher or `comfy launch`) or set COMFY_BASE_URL."
        ) from exc
    if resp.status_code != 200:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        raise _ComfyRunFailed(
            f"ComfyUI /prompt rejected the workflow ({resp.status_code}): "
            f"{resp.text[:800]}"
        )
    _mark_server_ready(base_url)
    prompt_id = resp.json()["prompt_id"]

    entry = _wait_for_completion(
        base_url,
        prompt_id,
        int(os.environ.get("COMFY_TIMEOUT", str(DEFAULT_IMAGE_TIMEOUT))),
        ws=ws,
    )
    elapsed = time.time() - start
    print(f"\n  [DONE] ComfyUI finished in {elapsed:.1f}s")
    path = _collect_and_save(base_url, entry, output_dir or "", filename)
    if not path:
        raise RuntimeError("ComfyUI save returned no path")
    return path


def generate(prompt: str,
             aspect_ratio: str = "1:1", image_size: str = "1K",
             output_dir: str = None, filename: str = None,
             model: str = None, max_retries: int = MAX_RETRIES) -> str:
    """Generate an image on the local ComfyUI, with retries.

    ``model`` selects the local model family (qwen / moss / zit) or a custom
    API-format template path; it wins over COMFY_MODEL and COMFY_WORKFLOW.
    """
    if model and model.strip().lower() not in _MODEL_ALIASES:
        print(f"[comfyui] Treating --model '{model}' as a workflow template path.")
    base_url = (os.environ.get("COMFY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    template_path, model_key = _resolve_template(model)
    if model and model_key is None:
        print(f"[comfyui] Custom template: the model is fixed by {template_path.name}.")

    # The local server serializes its own queue; retry transient submission
    # errors only (connection resets), never validation failures.
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return _generate_image(base_url, template_path, prompt,
                                   aspect_ratio, image_size,
                                   output_dir=output_dir, filename=filename,
                                   model=model_key)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            if isinstance(exc, _ComfyRunFailed) or is_permanent_error(exc):
                raise
            last_error = exc
            if attempt >= max_retries:
                break
            limited = is_rate_limit_error(exc)
            delay = retry_delay(attempt, rate_limited=limited)
            print(f"\n  [WARN] {exc} Retrying in {delay}s...")
            time.sleep(delay)
        except ConnectionError as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = retry_delay(attempt, rate_limited=False)
            print(f"\n  [WARN] {exc} Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")


if __name__ == "__main__":
    print(__doc__)
    print("Use via: python3 skills/ppt-master/scripts/image_gen.py \"prompt\" --backend comfyui")
    raise SystemExit(0 if any(arg in {"-h", "--help", "help"} for arg in sys.argv[1:]) else 1)
