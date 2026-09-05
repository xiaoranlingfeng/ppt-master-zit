#!/usr/bin/env python3
"""
Local ComfyUI image generation backend (zit基础 workflow).

Runs the exported "zit基础" workflow (Z-Image Turbo) against a locally
running ComfyUI server via its HTTP API. No API key is required.

Configuration keys:
  COMFY_BASE_URL       (optional; default http://127.0.0.1:8188)
  COMFY_WORKFLOW       (optional; path to an API-format workflow JSON.
                        Default: <skill>/templates/comfyui/zit_basic_api.json)
  COMFY_NEGATIVE_PROMPT (optional; overrides the template's negative prompt)
  COMFY_FILENAME_PREFIX (optional; SaveImage prefix, default "ppt_master")
  COMFY_TIMEOUT        (optional; seconds to wait per image, default 600)

Auto-start keys (launch the local ComfyUI when 127.0.0.1:8188 is down):
  COMFY_AUTOSTART      (optional; "1" enable / "0" disable, default "1")
  COMFY_OPEN_BROWSER   (optional; "1" open the ComfyUI web UI in the default
                        browser after an auto-start so the user can watch the
                        queue, "0" stay headless; default "1")
  COMFY_ROOT           (optional; aki-v3 integration root, e.g.
                        ~\Downloads\ComfyUI-aki-v3. Autodetected
                        from a candidate list when unset.)
  COMFY_LAUNCH_CMD     (optional; full custom launch command with {port}
                        placeholder, overrides COMFY_ROOT autodetection)
  COMFY_START_TIMEOUT  (optional; seconds to wait for server boot, default 300)

Sampler overrides (Z-Image Turbo recommended: steps 8 / cfg 1 /
res_multistep / simple / shift 3):
  COMFY_STEPS          (optional; KSampler steps)
  COMFY_CFG            (optional; KSampler cfg)
  COMFY_SAMPLER        (optional; sampler_name, e.g. res_multistep)
  COMFY_SCHEDULER      (optional; scheduler, e.g. simple)
  COMFY_SHIFT          (optional; ModelSamplingAuraFlow shift)

The template's placeholder nodes are patched per request:
  - the positive CLIPTextEncode node (text == "{{POSITIVE_PROMPT}}")
    receives the prompt;
  - EmptySD3LatentImage receives width/height computed from aspect_ratio
    and image_size (nearest multiple of 8, same policy as ResolutionSelector);
  - KSampler gets a fresh random seed each call.

All HTTP calls use a direct session (trust_env=False) so a system proxy
(HTTP_PROXY/HTTPS_PROXY) never intercepts the localhost ComfyUI traffic.
"""

import json
import math
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from console_encoding import configure_utf8_stdio  # noqa: E402

configure_utf8_stdio()

if __name__ == "__main__":
    print(__doc__)
    print("Use via: python3 skills/ppt-master/scripts/image_gen.py \"prompt\" --backend comfyui")
    raise SystemExit(0 if any(arg in {"-h", "--help", "help"} for arg in sys.argv[1:]) else 1)

import os

import requests

# Direct localhost session: never route ComfyUI traffic through a system proxy
# (HTTP_PROXY/HTTPS_PROXY are commonly set on this machine and would break
# access to 127.0.0.1:8188).
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.proxies = {"http": None, "https": None}

from image_backends.backend_common import (
    MAX_RETRIES,
    is_permanent_error,
    is_rate_limit_error,
    normalize_image_size,
    retry_delay,
    save_image_bytes,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_TEMPLATE = (
    _SCRIPTS_DIR.parent / "templates" / "comfyui" / "zit_basic_api.json"
)
PROMPT_PLACEHOLDER = "{{POSITIVE_PROMPT}}"
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
            # CREATE_BREAKAWAY_FROM_JOB lets the server outlive this agent
            # session (job objects that kill children on close); fall back to
            # plain detached when the job forbids breakaway.
            try:
                subprocess.Popen(
                    cmd, creationflags=detached | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                    **popen_kwargs,
                )
                return
            except OSError:
                subprocess.Popen(cmd, creationflags=detached, **popen_kwargs)
                return
        subprocess.Popen(cmd, **popen_kwargs)


def _ensure_server_running(base_url: str) -> None:
    """Make sure ComfyUI answers; auto-start it when configured to do so."""
    if _server_ready(base_url):
        return
    if os.environ.get("COMFY_AUTOSTART", "1").strip().lower() in {"0", "false", "no"}:
        raise ConnectionError(
            f"ComfyUI is not reachable at {base_url} and COMFY_AUTOSTART=0. "
            "Start the local ComfyUI manually and retry."
        )
    _launch_comfyui(base_url)
    start_timeout = int(os.environ.get("COMFY_START_TIMEOUT", "300"))
    deadline = time.time() + start_timeout
    while time.time() < deadline:
        if _server_ready(base_url, timeout=5.0):
            print("  [OK] ComfyUI is up.", flush=True)
            _open_browser(base_url)
            return
        time.sleep(3)
    raise ConnectionError(
        f"ComfyUI did not become ready at {base_url} within {start_timeout}s. "
        "Check comfyui_autostart.log under the integration root, or start it "
        "manually with the aki launcher and retry."
    )


def _resolve_dimensions(aspect_ratio: str, image_size: str) -> tuple[int, int]:
    """Compute width/height (multiple of 8) for the requested ratio and size."""
    ratio = ASPECT_RATIO_MAP.get(aspect_ratio)
    if ratio is None:
        raise ValueError(
            f"Unsupported aspect ratio '{aspect_ratio}' for comfyui backend. "
            f"Supported: {sorted(ASPECT_RATIO_MAP)}"
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
    width = max(64, int(round(width / 8.0)) * 8)
    height = max(64, int(round(height / 8.0)) * 8)
    return width, height


def _load_template(template_path: Path) -> dict:
    if not template_path.is_file():
        raise FileNotFoundError(
            f"ComfyUI workflow template not found: {template_path}\n"
            "Set COMFY_WORKFLOW to an API-format workflow JSON exported "
            "from ComfyUI (the local 'zit基础' workflow is preinstalled at "
            "templates/comfyui/zit_basic_api.json)."
        )
    graph = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not graph:
        raise ValueError(f"Workflow template must be a non-empty JSON object: {template_path}")
    return graph


def _find_nodes(graph: dict, class_type: str) -> list[str]:
    return [nid for nid, node in graph.items() if node.get("class_type") == class_type]


def _patch_graph(
    graph: dict,
    prompt: str,
    width: int,
    height: int,
    negative_prompt: str | None,
    filename_prefix: str,
) -> dict:
    """Return a per-run copy of the template with prompt/size/seed injected."""
    graph = json.loads(json.dumps(graph))  # deep copy

    positive_nodes = [
        nid for nid, node in graph.items()
        if node.get("class_type") == "CLIPTextEncode"
        and node.get("inputs", {}).get("text") == PROMPT_PLACEHOLDER
    ]
    if not positive_nodes:
        # Fall back to a heuristic: the CLIPTextEncode whose text is shortest
        # but non-empty is usually the positive; a long quality-word list is
        # the negative. Require the placeholder instead of guessing.
        raise ValueError(
            "Workflow template is missing the positive-prompt placeholder node "
            f"(a CLIPTextEncode whose text input equals {PROMPT_PLACEHOLDER!r}). "
            "Use the shipped templates/comfyui/zit_basic_api.json or add the "
            "placeholder manually."
        )
    graph[positive_nodes[0]]["inputs"]["text"] = prompt

    if negative_prompt:
        negatives = [
            nid for nid in _find_nodes(graph, "CLIPTextEncode")
            if nid not in positive_nodes
        ]
        for nid in negatives:
            graph[nid]["inputs"]["text"] = negative_prompt

    for nid in _find_nodes(graph, "EmptySD3LatentImage"):
        graph[nid]["inputs"]["width"] = width
        graph[nid]["inputs"]["height"] = height
        graph[nid]["inputs"]["batch_size"] = 1

    # Optional sampler overrides (Z-Image Turbo defaults: steps 8, cfg 1,
    # res_multistep, simple, shift 3).
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
                inputs[key] = int(raw)
        for key, raw in _float_overrides.items():
            if raw:
                inputs[key] = float(raw)
        for key, raw in _raw_overrides.items():
            if raw:
                inputs[key] = raw
    shift_raw = os.environ.get("COMFY_SHIFT", "").strip()
    if shift_raw:
        for nid in _find_nodes(graph, "ModelSamplingAuraFlow"):
            graph[nid]["inputs"]["shift"] = float(shift_raw)

    for nid in _find_nodes(graph, "SaveImage"):
        graph[nid]["inputs"]["filename_prefix"] = filename_prefix

    return graph


class _ComfyRunFailed(RuntimeError):
    """Server executed the graph and reported a node failure."""


def _wait_for_completion(base_url: str, prompt_id: str, timeout: int) -> dict:
    """Poll /history until the run completes; return its history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = _SESSION.get(f"{base_url}/history/{prompt_id}", timeout=30)
        except requests.RequestException:
            time.sleep(3)
            continue
        if resp.status_code == 200:
            entry = resp.json().get(prompt_id)
            if entry:
                status = (entry.get("status") or {})
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
        time.sleep(2)
    raise RuntimeError(
        f"ComfyUI run {prompt_id} did not finish within {timeout}s "
        "(is the GPU busy with another queue?). Raise COMFY_TIMEOUT or clear "
        "the queue and retry."
    )


def _collect_and_save(base_url: str, entry: dict, output_dir: str, filename: str) -> str:
    """Download the first image output and save it under the manifest name."""
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
            if filename:
                base = os.path.splitext(filename)[0]
            else:
                base = f"comfy_{int(time.time())}"
            path = os.path.join(output_dir, f"{base}{ext}") if output_dir else f"{base}{ext}"
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            return save_image_bytes(resp.content, path,
                                    content_type=resp.headers.get("content-type"))
    raise RuntimeError("ComfyUI run finished but produced no output images")


def _generate_image(base_url: str, template_path: Path, prompt: str,
                    aspect_ratio: str, image_size: str,
                    output_dir: str = None, filename: str = None) -> str:
    width, height = _resolve_dimensions(aspect_ratio, image_size)
    graph = _patch_graph(
        _load_template(template_path),
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=os.environ.get("COMFY_NEGATIVE_PROMPT"),
        filename_prefix=os.environ.get("COMFY_FILENAME_PREFIX", "ppt_master"),
    )
    client_id = f"ppt-master-{random.randint(0, 2**31)}"

    print("[comfyui local]")
    print(f"  Server:       {base_url}")
    print(f"  Template:     {template_path}")
    print(f"  Prompt:       {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"  Size:         {width}x{height}")
    print()
    _ensure_server_running(base_url)
    print("  [..] Submitting to ComfyUI queue...", flush=True)
    start = time.time()
    try:
        resp = _SESSION.post(
            f"{base_url}/prompt",
            json={"prompt": graph, "client_id": client_id},
            timeout=30,
        )
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach ComfyUI at {base_url}. Start the local ComfyUI "
            "(e.g. the aki launcher or `comfy launch`) or set COMFY_BASE_URL."
        ) from exc
    if resp.status_code != 200:
        raise _ComfyRunFailed(
            f"ComfyUI /prompt rejected the workflow ({resp.status_code}): "
            f"{resp.text[:800]}"
        )
    prompt_id = resp.json()["prompt_id"]

    entry = _wait_for_completion(base_url, prompt_id,
                                 int(os.environ.get("COMFY_TIMEOUT", "600")))
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
    """Generate an image on the local ComfyUI (zit基础 workflow), with retries."""
    if model:
        print(f"[comfyui] Ignoring --model '{model}': the model is fixed by "
              "the ComfyUI workflow template.")
    base_url = (os.environ.get("COMFY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    template_env = os.environ.get("COMFY_WORKFLOW", "").strip()
    template_path = Path(template_env).expanduser() if template_env else DEFAULT_TEMPLATE

    # The local server serializes its own queue; retry transient submission
    # errors only (connection resets), never validation failures.
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return _generate_image(base_url, template_path, prompt,
                                   aspect_ratio, image_size,
                                   output_dir=output_dir, filename=filename)
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
