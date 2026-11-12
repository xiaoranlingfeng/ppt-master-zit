# ppt-master + local ComfyUI backend

Official **ppt-master v6.8.0** (MIT, © Hugo He) plus one added image backend: `IMAGE_BACKEND=comfyui`
renders on your own ComfyUI — Qwen-Image-2.1 by default (native RGBA), with MOSS-Image SFT and
Z-Image Turbo selectable per deck. No API key, no per-image cost, nothing leaves the machine.

## What this branch adds

| File | Purpose |
|---|---|
| `scripts/image_backends/backend_comfyui.py` | ComfyUI HTTP adapter: submits an API-format workflow with the prompt injected, polls history, fetches the PNG, computes size from `aspect_ratio × image_size` |
| `templates/comfyui/qwen_image_2_1_t2i_api.json` | Qwen-Image-2.1 official path (25 / cfg 1 / euler / simple, RGBA) |
| `templates/comfyui/moss_image_sft_api.json` | MOSS-Image SFT (50 / cfg 3 / euler / sgm_uniform / shift 5) |
| `templates/comfyui/zit_basic_api.json` | Z-Image Turbo (8 / 1 / res_multistep / simple) |
| `scripts/image_gen.py` | one registry entry (`comfyui`) + the `COMFY_` env prefix |
| `.env.example`, `references/image-generator.md` §4.5 | how to configure and how to prompt |

Everything else is upstream, untouched. A version bump is four files plus the registry patch.

## Configure

```dotenv
IMAGE_BACKEND=comfyui
COMFY_MODEL=qwen          # qwen | moss | zit
COMFY_BASE_URL=http://127.0.0.1:8188
COMFY_AUTOSTART=1         # launch ComfyUI if the port is down
```

```bash
python scripts/image_gen.py "a watercolor mountain range, empty left half" \
  --backend comfyui --aspect_ratio 16:9 --image_size 1K -o images -f cover
```

## Models to install

- Qwen-Image-2.1: `diffusion_models/qwen image 2.1/qwen_image_2.1_int8_convrot.safetensors`,
  `text_encoders/qwen3vl_8b_int8_convrot.safetensors`, `vae/qwen_image_2.1_vae_bf16.safetensors`
- MOSS-Image SFT: `moss_image_sft_int8_convrot.safetensors`, `qwen3.5_9b_w4a8_gptq.safetensors`, `flux2-vae.safetensors`
- Z-Image Turbo: its own UNET/CLIP/VAE set used by `zit_basic_api.json`

## Licenses

Upstream ppt-master is MIT. Qwen-Image-2.1 weights ship under the Qwen Research License
(non-commercial); MOSS-Image likewise — check before commercial delivery. Z-Image Turbo is Apache-2.0.

## Upstream

- Repository: https://github.com/hugohe3/ppt-master (v6.8.0)
- The full-featured local fork (interview gate, manifest resume, rendered-output gate) lives at
  `xiaoranlingfeng/ppt-master-zit`, branch `main`.