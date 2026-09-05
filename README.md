# ppt-master-zit

> [English below 中文在下]

**PPT Master + 本机 ComfyUI 配图后端** — 这是 [ppt-master](https://github.com/hugohe3/ppt-master) (v6.1.0, MIT, Copyright © 2025-2026 Hugo He) 的本机定制分支：整套可编辑 PPTX 生成工作流与官方一致，唯一区别是 **AI 配图默认走本机 ComfyUI 的「zit基础」工作流（Z-Image Turbo），不需要任何 API Key**。

## 与上游的差异

| 项 | 官方 ppt-master | 本仓库 ppt-master-zit |
|---|---|---|
| 图片后端 | openai / gemini / qwen 等云端 API | 全部保留，**新增 `comfyui` 本地后端并默认启用** |
| 出图引擎 | 云端模型 | 本机 ComfyUI「zit基础」= Z-Image Turbo (bf16) + qwen_3_4b |
| 费用 | 按 API 计费 | 免费（本地 GPU，约 10~25 秒/张 @1K，实测 RTX 3080 Laptop） |
| 隐私 | 提示词出网 | 完全本机 |

新增/修改的文件：

- `scripts/image_backends/backend_comfyui.py` — ComfyUI HTTP API 后端：提示词注入、按 aspect_ratio×image_size 计算分辨率（8 的倍数）、每次随机种子、失败节点报错透出、`trust_env=False` 直连（不受系统代理劫持）
- `templates/comfyui/zit_basic_api.json` — 「zit基础」工作流的 API 格式模板（Z-Image Turbo 推荐采样：steps 8 / cfg 1 / res_multistep / simple / shift 3）
- `scripts/image_gen.py` — 注册 `comfyui` 后端（别名 comfy / zit）
- `references/image-generator.md` §4.5 — 本地后端的提示词规范（自然语言、1–4 句、强制无文字条款）
- `.env` / `.env.example` — comfyui 后端配置（含自动启动）

其余文件与官方 v6.1.0 一致（含质量门禁 `--stage early`、原生形状词汇等上游能力）。

## 快速上手

1. 安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（任意发行版/整合包均可），放好 Z-Image Turbo 模型：
   - `diffusion_models/zit/z_image_turbo_bf16.safetensors`
   - `text_encoders/qwen_3_4b.safetensors`
   - `vae/ae.safetensors`
   （模型来自 [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)）
2. 把本仓库放进你的 agent skills 目录（如 `~/.hermes/skills/`、`~/.agents/skills/`），或作为任意能执行 Python 的 agent 的技能文件夹。
3. 复制 `.env.example` 为 `.env`，至少设置：
   ```ini
   IMAGE_BACKEND=comfyui
   COMFY_ROOT=C:/path/to/ComfyUI-aki-v3   # 含 python\python.exe 与 ComfyUI\main.py 的整合包根目录
   ```
4. 完事。ComfyUI 没开也行——后端检测到 `127.0.0.1:8188` 无响应会**自动拉起整合包并在浏览器打开 ComfyUI 界面**（`COMFY_OPEN_BROWSER=0` 可静默）。

手动验证一次出图：

```bash
python scripts/image_gen.py "明亮教室里的学生协作，扁平插画，暖色调，no text, letters, numbers or watermarks." \
  --backend comfyui --aspect_ratio 16:9 --image_size 1K -o ./out
```

### 主要 .env 开关

| 键 | 默认 | 说明 |
|---|---|---|
| `COMFY_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI 地址 |
| `COMFY_WORKFLOW` | 内置模板 | 换任意 API 格式工作流 JSON |
| `COMFY_AUTOSTART` / `COMFY_OPEN_BROWSER` | `1` / `1` | 自动拉起 / 拉起后开界面 |
| `COMFY_STEPS` `COMFY_CFG` `COMFY_SAMPLER` `COMFY_SCHEDULER` `COMFY_SHIFT` | 模板值 | 采样覆盖 |
| `COMFY_TIMEOUT` / `COMFY_START_TIMEOUT` | `900` / `300` | 出图 / 冷启动等待上限（秒） |
| `IMAGE_CONCURRENCY` | `2` | 本地 GPU 串行，建议 1–2 |

## 提示词要点（Z-Image Turbo）

- 自然语言整句（主体→场景→光影→风格），不要标签堆砌；中文英文皆可
- **每句必须带**「画面中不出现任何文字、字母、数字或水印」/ "Strictly no text, no letters, no numbers, no watermarks anywhere."
- 抽象概念章节图慎用中文概念词——模型可能把词直接画成乱码假字，改英文更安全
- 出图后逐张目检，有字即改 prompt 重跑 manifest

## License

MIT — 版权归上游 [Hugo He](https://github.com/hugohe3/ppt-master)（Copyright © 2025-2026）；本仓库的本地后端增量同样以 MIT 发布。

---

**ppt-master-zit** is a personal-machine fork of the excellent [ppt-master](https://github.com/hugohe3/ppt-master) v6.1.0 workflow: same editable-PPTX generation, but a new local `comfyui` image backend (the "zit基础" Z-Image Turbo workflow) replaces cloud API image calls — free, private, and it auto-starts your local ComfyUI when needed. See the Chinese section for the full setup; the upstream SKILL.md, LICENSE and attribution are preserved untouched.
