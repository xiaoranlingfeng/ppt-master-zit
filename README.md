# ppt-master-comfyui

> Former names 曾用名: `ppt-master-qwen-image-2-1` · `ppt-master-zit` — renamed 2026-10-13; the fork is defined by its local ComfyUI backend, not by any single model.

> **Upstream lock 上游锁定**: this fork is rebased onto official ppt-master **v6.7.0** (MIT, Hugo He). Fork releases are tagged `v<upstream>-comfyui.<n>` (`v6.7.0-comfyui.1` shipped the MOSS + interview-gate baseline; the current local build is `v6.7.0-comfyui.2`); pin those tags instead of `main` if you need a stable point. A GitHub Actions `smoke` workflow runs the GPU-free unit tests (backend, export regressions, manifest resume) on every push.

[English](#english) · [中文](#中文)

---

<a name="english"></a>

## English

**PPT Master + local ComfyUI image backend.** This is a personal-machine fork of [ppt-master](https://github.com/hugohe3/ppt-master) (v6.7.0, MIT, Copyright © 2025-2026 Hugo He). The entire editable-PPTX generation workflow is identical to upstream; the one difference is that **AI slide imagery defaults to your local ComfyUI running the Qwen-Image-2.1 text-to-image workflow — no API key, no cost, nothing leaves your machine.** Two more local workflows are selectable per deck: MOSS-Image SFT for literal, layout-steerable art, and the legacy "zit基础" (Z-Image Turbo) for speed.

### What differs from upstream

| | Official ppt-master | ppt-master-comfyui |
|---|---|---|
| Image backend | openai / gemini / qwen and other cloud APIs | all preserved, **plus a new local `comfyui` backend, enabled by default** |
| Generation engine | cloud models | local ComfyUI = **Qwen-Image-2.1** (7B DiT int8 + Qwen3-VL 8B int8, native RGBA) by default; selectable **MOSS-Image SFT** (Qwen3.5 9B encoder, opaque RGB) and **"zit基础" Z-Image Turbo** (fast, Apache-2.0) |
| Model choice | fixed by the cloud config | `COMFY_MODEL=qwen|moss|zit`, confirmed once per deck at interview-gate **D5** |
| Cost | per-API billing | free (local GPU; ~15–25 s per 1K image at 25 steps, tested on RTX 3080 Laptop) |
| Privacy | prompts leave the machine | fully local |

New / modified files:

- `scripts/image_backends/backend_comfyui.py` — ComfyUI HTTP API backend: prompt injection (CLIPTextEncode or TextEncodeQwenImage21), resolution computed from `aspect_ratio × image_size` (nearest multiple of 32), fresh random seed per run, node-level failure reporting, `trust_env=False` direct connection (immune to system proxy interception)
- `templates/comfyui/qwen_image_2_1_t2i_api.json` — API-format export of the `image_qwen_image_2_1_t2i` workflow (Qwen-Image-2.1 official path: steps 25 / cfg 1 / euler / simple; native RGBA transparency via prompt wrapping)
- `templates/comfyui/moss_image_sft_api.json` — MOSS-Image SFT workflow (steps 50 / cfg 3 / euler / sgm_uniform / shift 5 through `ModelSamplingAuraFlow`, uncond-skip at layer 22; opaque RGB)
- `templates/comfyui/zit_basic_api.json` — legacy "zit基础" workflow (Z-Image Turbo: steps 8 / cfg 1 / res_multistep / simple / shift 3), selectable via `COMFY_WORKFLOW`
- `scripts/image_gen.py` — registers the `comfyui` backend (aliases: comfy / zit / moss)
- `references/image-generator.md` §4.5 — the local-backend contract: the three-model table, the once-per-deck model question and its D5 carve-out, prompting rules (natural language, 1–4 sentences, mandatory no-text clause), and the MOSS rule that hands prompt authoring to the sibling `moss-image-prompt-enhancer` skill
- `workflows/interview-gate.md` — the per-decision interview gate (D0–D9): outline, page count, per-page content, transitions, per-element animation, image usage + local model, playback and deliverables are each confirmed with the user before the existing gates; D5 carries the model choice and the `1K`/`2K` cost rule, so the model is asked once and never re-opened at image time
- Rendered-output verification gate (`workflows/generate-pptx.md`) — after export, convert to PDF (`soffice`, headless) and read **every** page image (`pymupdf`) before claiming a deck is done; the checklist names the four failure classes the SVG checker cannot see
- PPTX bullet fixes (`scripts/svg_to_pptx/`) — generated prose no longer inherits the slide master's `bodyStyle` bullet: every paragraph writes an explicit `buNone` + zeroed indent, the master's default bullets are cleared, and the scope now covers `object` slots and plain text boxes, not just `body`/`subtitle`
- Manifest resume (`scripts/image_gen.py`) — a prompt+size+model signature sidecar (`.image_gen_state.json`) survives pipeline rewrites of `image_prompts.json`, so editing one prompt no longer re-renders the whole deck; `--force` still regenerates everything
- `scripts/prompt_audit.py` + manifest — maintainer-only, read-only budget lint for the agent-facing Markdown corpus (needs `tiktoken` and a repo-shaped checkout; never loaded during generation)
- `references/qwen-image-2-1-prompting.md` + `templates/qwen_prompt_skeletons.md` — the official-case prompting manual (decision flow, layered skeleton, text-locked / edit / multi-ref / storyboard recipes, alpha verification discipline) and its five fill-in skeletons, distilled from modelscope.cn/posts/436932
- `scripts/test_backend_comfyui_smoke.py`, `test_export_regressions.py`, `test_manifest_resume.py` + `.github/workflows/smoke.yml` — GPU-free unit tests (dimension resolution, placeholder injection, backend registry, sampler-override warnings, inherited bullets, manifest resume) run on every push to `scripts/`
- `.env.example` / `.gitignore` / `README.md` — local config docs; personal `.env` is never committed

Everything else matches upstream v6.7.0 (including glyph-accurate text measurement, native chart/table mirroring, the multilingual per-script calibration of v6.4.0, the Edit-Native round-trip rework of v6.5.0, and the v6.6.0 end-to-end-run fixes), plus the four local export fixes behind the rendered-output gate: explicit `buNone` on generated prose, cleared Master `bodyStyle` bullets, bullet scope extended past `body`/`subtitle`, and the multiline-carrier contract.

### Quick start

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (any distribution or integrated pack) with the Qwen-Image-2.1 models in place:
   - `diffusion_models/qwen image 2.1/qwen_image_2.1_int8_convrot.safetensors` (or bf16)
   - `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
   - `vae/qwen_image_2.1_vae_bf16.safetensors`
   (models from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1); weights are free but the Qwen Research License bars commercial use without a separate agreement)
   Optional, only if you want the other two workflows: MOSS-Image SFT needs
   `moss_image_sft_int8_convrot.safetensors` + a Qwen3.5 9B text encoder +
   `flux2-vae.safetensors`, and Z-Image Turbo needs its own UNET/CLIP/VAE set —
   see §4.5 for the exact node names the shipped templates expect.
2. Drop this folder into your agent's skills directory (e.g. `~/.hermes/skills/`, `~/.agents/skills/`), or use it with any agent that can execute Python.
3. Copy `.env.example` to `.env` and set at least:
   ```ini
   IMAGE_BACKEND=comfyui
   COMFY_MODEL=qwen                     # qwen | moss | zit
   COMFY_ROOT=C:/path/to/ComfyUI-aki-v3   # integration root containing python\python.exe and ComfyUI\main.py
   ```
4. Done. ComfyUI does not even need to be running — the backend detects `127.0.0.1:8188` being down, **auto-starts the integration pack, and opens the ComfyUI web UI in your browser** (set `COMFY_OPEN_BROWSER=0` to stay headless).

Manual smoke test:

```bash
python scripts/image_gen.py "students collaborating in a bright classroom, flat illustration, warm tones, Strictly no text, no letters, no numbers, no watermarks anywhere." \
  --backend comfyui --aspect_ratio 16:9 --image_size 1K -o ./out
```

### Key `.env` switches

| Key | Default | Meaning |
|---|---|---|
| `COMFY_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI address |
| `COMFY_MODEL` | `qwen` | local family: `qwen` / `moss` / `zit` (each brings its own template + sampler) |
| `COMFY_WORKFLOW` | built-in template | point at any other API-format workflow JSON |
| `COMFY_AUTOSTART` / `COMFY_OPEN_BROWSER` | `1` / `1` | auto-launch / open web UI after launch |
| `COMFY_STEPS` `COMFY_CFG` `COMFY_SAMPLER` `COMFY_SCHEDULER` `COMFY_SHIFT` | template values | sampler overrides; they beat the selected model's template, so clear them when the model changes |
| `COMFY_TEXT_RESOLUTION` | follows the size tier | pin the Qwen-Image-2.1 encoder `resolution` hint |
| `COMFY_MEGAPIXELS` | from `image_size` | override the megapixel target directly |
| `COMFY_TIMEOUT` / `COMFY_START_TIMEOUT` | `900` / `300` | per-image / cold-boot wait limits (seconds) |
| `IMAGE_CONCURRENCY` | `2` | local GPU serializes; keep at 1–2 |

### Prompting tips (Qwen-Image-2.1)

These are the Qwen-Image-2.1 rules. For `COMFY_MODEL=moss`, prompts are authored
by the sibling `moss-image-prompt-enhancer` skill instead (§4.5 owns that
contract); for `zit`, drop the RGBA wrapper and keep the no-text clause.

- **House default: every deck asset is 16:9 RGBA** — subject cutouts, section art, and background decorations all carry real alpha and drop onto any slide color without a white box
- Three-slot wrapper on every prompt: `This is an RGBA format image with transparency.` → natural-language description → `The image has an alpha channel and a transparent background.`; middle text in Chinese or English, no tag soup
- Layout is steerable through alpha: name the empty zone ("the center and left area must be completely empty, reserved for slide text") and it comes back transparent
- Three recipes: A subject cutout hugging one third · B edge-frame decoration with emptied center · C atmosphere shape layer (author the shape fully; set layer opacity in SVG/PPTX compositing — the model ignores "max N% opacity" numbers)
- Paint in the deck's own palette family so semi-transparent edges composite invisibly on the slide background
- Keep cfg at 1 (official path); negative prompts have no effect there
- **Every prompt must end with** "画面中不出现任何文字、字母、数字或水印" / "Strictly no text, no letters, no numbers, no watermarks anywhere."
- Visually inspect every generated image; re-roll any that contain text

(§4.5 of `references/image-generator.md` is the authoritative, verified version of these rules.)

### License

MIT — upstream copyright belongs to [Hugo He](https://github.com/hugohe3/ppt-master) (Copyright © 2025-2026); the local-backend additions in this repository are released under the same MIT terms.

---

<a name="中文"></a>

## 中文

**PPT Master + 本机 ComfyUI 配图后端** — 这是 [ppt-master](https://github.com/hugohe3/ppt-master) (v6.7.0, MIT, Copyright © 2025-2026 Hugo He) 的本机定制分支：整套可编辑 PPTX 生成工作流与官方一致，唯一区别是 **AI 配图默认走本机 ComfyUI 的 Qwen-Image-2.1 文生图工作流，不需要任何 API Key**。另外两套本机工作流可按 deck 切换：MOSS-Image SFT 擅长字面构图与版式可控，旧「zit基础」（Z-Image Turbo）主打速度。

### 与上游的差异

| 项 | 官方 ppt-master | 本仓库 ppt-master-comfyui（曾用名 ppt-master-qwen-image-2-1、ppt-master-zit） |
|---|---|---|
| 图片后端 | openai / gemini / qwen 等云端 API | 全部保留，**新增 `comfyui` 本地后端并默认启用** |
| 出图引擎 | 云端模型 | 本机 ComfyUI = 默认 **Qwen-Image-2.1**（7B DiT int8 + Qwen3-VL 8B int8，原生 RGBA）；可切 **MOSS-Image SFT**（Qwen3.5 9B 编码器，不透明 RGB）与 **「zit基础」Z-Image Turbo**（最快，Apache-2.0） |
| 模型选择 | 由云端配置写死 | `COMFY_MODEL=qwen|moss|zit`，每套 deck 在访谈门禁 **D5** 确认一次 |
| 费用 | 按 API 计费 | 免费（本地 GPU，约 15~25 秒/张 @1K、25 步，实测 RTX 3080 Laptop） |
| 隐私 | 提示词出网 | 完全本机 |

新增/修改的文件：

- `scripts/image_backends/backend_comfyui.py` — ComfyUI HTTP API 后端：提示词注入（CLIPTextEncode 或 TextEncodeQwenImage21）、按 aspect_ratio×image_size 计算分辨率（32 的倍数）、每次随机种子、失败节点报错透出、`trust_env=False` 直连（不受系统代理劫持）
- `templates/comfyui/qwen_image_2_1_t2i_api.json` — `image_qwen_image_2_1_t2i` 工作流的 API 格式模板（Qwen-Image-2.1 官方路径：steps 25 / cfg 1 / euler / simple；提示词包裹即可原生出 RGBA 透明图）
- `templates/comfyui/moss_image_sft_api.json` — MOSS-Image SFT 工作流（steps 50 / cfg 3 / euler / sgm_uniform，shift 5 走 `ModelSamplingAuraFlow`，第 22 层 uncond-skip；输出不透明 RGB）
- `templates/comfyui/zit_basic_api.json` — 旧「zit基础」工作流（Z-Image Turbo：steps 8 / cfg 1 / res_multistep / simple / shift 3），经 `COMFY_MODEL=zit` 切换
- `scripts/image_gen.py` — 注册 `comfyui` 后端（别名 comfy / zit / moss）
- `references/image-generator.md` §4.5 — 本地后端契约：三档模型表、每套 deck 一次的模型提问及其 D5 豁免、提示词规范（自然语言、1–4 句、强制无文字条款），以及把 MOSS 提示词交给同级技能 `moss-image-prompt-enhancer` 的规则
- `workflows/interview-gate.md` — 逐项决策门禁（D0–D9）：大纲、页数、每页内容、切换动画、元素动画、配图方式与本地模型、放映节奏、交付物，每项都由用户拍板后再进既有门禁；D5 同时承接模型选择与 `1K`/`2K` 成本规则，出图阶段不再重复提问
- 渲染后验收门（`workflows/generate-pptx.md`）— 导出后先用 `soffice` 转 PDF、再用 `pymupdf` 逐页转图并**每页都看**，才可宣称完成；清单列出 SVG 检查器看不见的四类失败
- PPTX 项目符号修复（`scripts/svg_to_pptx/`）— 生成的正文不再继承母版 `bodyStyle` 项目符号：每个段落显式写 `buNone` + 缩进归零、母版默认符号被清除，作用域从 `body`/`subtitle` 扩到 `object` 槽位与普通文本框
- manifest 断点续跑（`scripts/image_gen.py`）— 用「提示词+尺寸+模型」签名旁车文件（`.image_gen_state.json`）跨过流水线对 `image_prompts.json` 的重写，改一条提示词不再重出整套图；`--force` 仍可强制全部重出
- `scripts/prompt_audit.py` + manifest — 仅供维护者的只读提示词预算审计（需 `tiktoken` 且要求仓库式布局；生成过程从不加载）
- `references/qwen-image-2-1-prompting.md` + `templates/qwen_prompt_skeletons.md` — 官方示例库 40 余案例拆解出的提示词手册（决策流、分层骨架、文字锁定/编辑/多参考/分镜配方、alpha 验证纪律）及五套可填空骨架，源出 modelscope.cn/posts/436932
- `scripts/test_backend_comfyui_smoke.py`、`test_export_regressions.py`、`test_manifest_resume.py` + `.github/workflows/smoke.yml` — 无 GPU 依赖的单元测试（尺寸解析、占位符注入、编码器分辨率跟随档位、后端注册表、采样覆盖告警、继承项目符号、断点续跑），每次推送到 `scripts/` 自动执行
- `.env.example` / `.gitignore` / `README.md` — 本机配置文档；含个人路径的 `.env` 永不入库

其余文件与官方 v6.7.0 一致（含真实字形宽度度量、原生图表/表格镜像导出、v6.4.0 多语种按文字体系的宽度校准、v6.5.0 Edit Native 往返重做、v6.6.0 全流程实测修复等上游能力），另加渲染验收门背后的四项本机导出修复：正文显式 `buNone`、清除母版 `bodyStyle` 项目符号、符号作用域扩到 `body`/`subtitle` 之外、多行载体契约。

### 快速上手

1. 安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（任意发行版/整合包均可），放好 Qwen-Image-2.1 模型：
   - `diffusion_models/qwen image 2.1/qwen_image_2.1_int8_convrot.safetensors`（或 bf16）
   - `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
   - `vae/qwen_image_2.1_vae_bf16.safetensors`
   （模型来自 [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)；权重免费下载，但 Qwen Research License 限制商用，需另行申请授权）
   可选，仅在你需要另外两档模型时：MOSS-Image SFT 需要
   `moss_image_sft_int8_convrot.safetensors` + Qwen3.5 9B 文本编码器 +
   `flux2-vae.safetensors`，Z-Image Turbo 需要它自己的 UNET/CLIP/VAE 组合，
   模板期望的确切节点文件名以 §4.5 为准。
2. 把本仓库放进你的 agent skills 目录（如 `~/.hermes/skills/`、`~/.agents/skills/`），或作为任意能执行 Python 的 agent 的技能文件夹。
3. 复制 `.env.example` 为 `.env`，至少设置：
   ```ini
   IMAGE_BACKEND=comfyui
   COMFY_MODEL=qwen                     # qwen | moss | zit
   COMFY_ROOT=C:/path/to/ComfyUI-aki-v3   # 含 python\python.exe 与 ComfyUI\main.py 的整合包根目录
   ```
4. 完事。ComfyUI 没开也行——后端检测到 `127.0.0.1:8188` 无响应会**自动拉起整合包并在浏览器打开 ComfyUI 界面**（`COMFY_OPEN_BROWSER=0` 可静默）。

手动验证一次出图：

```bash
python scripts/image_gen.py "明亮教室里的学生协作，扁平插画，暖色调，画面中不出现任何文字、字母、数字或水印。" \
  --backend comfyui --aspect_ratio 16:9 --image_size 1K -o ./out
```

### 主要 .env 开关

| 键 | 默认 | 说明 |
|---|---|---|
| `COMFY_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI 地址 |
| `COMFY_MODEL` | `qwen` | 本机模型档位：`qwen` / `moss` / `zit`（各自绑定模板与采样参数） |
| `COMFY_WORKFLOW` | 内置模板 | 换任意 API 格式工作流 JSON |
| `COMFY_AUTOSTART` / `COMFY_OPEN_BROWSER` | `1` / `1` | 自动拉起 / 拉起后开界面 |
| `COMFY_STEPS` `COMFY_CFG` `COMFY_SAMPLER` `COMFY_SCHEDULER` `COMFY_SHIFT` | 模板值 | 采样覆盖；这些值会静默压过所选模型模板自带的调参，换模型时请先清空 |
| `COMFY_TEXT_RESOLUTION` | 跟随出图档位 | 手动钉住 Qwen-Image-2.1 编码器的 `resolution` 条件 |
| `COMFY_MEGAPIXELS` | 由 `image_size` 推导 | 直接指定目标像素量，绕开 1K/2K/4K 档位表 |
| `COMFY_TIMEOUT` / `COMFY_START_TIMEOUT` | `900` / `300` | 出图 / 冷启动等待上限（秒） |
| `IMAGE_CONCURRENCY` | `2` | 本地 GPU 串行，建议 1–2 |

### 提示词要点（Qwen-Image-2.1）

以下是 Qwen-Image-2.1 的规则。`COMFY_MODEL=moss` 时提示词改由同级技能
`moss-image-prompt-enhancer` 编写（契约见 §4.5）；`COMFY_MODEL=zit` 时去掉 RGBA 包裹句，
保留强制无文字条款。

- **默认规范：所有 PPT 配图 / 背景图一律 16:9 RGBA**——主体插图、章节图、背景装饰都带真实 alpha 通道，直接贴任何底色页面，没有白框
- 三槽包裹：`This is an RGBA format image with transparency.` → 自然语言描述 → `The image has an alpha channel and a transparent background.`；中间描述中英皆可，不要标签堆砌
- 布局可通过 alpha 指令控制：明确指定留空区（"画面中央与左侧完全留空透明，预留放文字"），该区域就会真的透明
- 三种配方：A 主体贴边抠图（占三分之一）· B 边缘框饰（中间留空）· C 氛围形状层（模型只画形状，"最大 15% 不透明度"这类数字它不理——透明度在 SVG/PPTX 合成层设置）
- 用演示文稿自己的同色系给 RGBA 素材上色，半透明边缘在页面底色上无痕融合
- cfg 保持 1（官方路径），此时负面词不生效
- **每句必须带**「画面中不出现任何文字、字母、数字或水印」
- 出图后逐张目检，有字即改 prompt 重跑 manifest

（权威版本见 `references/image-generator.md` §4.5，全部结论有实测背书。）

### 许可证

MIT — 版权归上游 [Hugo He](https://github.com/hugohe3/ppt-master)（Copyright © 2025-2026）；本仓库的本地后端增量同样以 MIT 发布。
