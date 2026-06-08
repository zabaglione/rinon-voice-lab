# Rinon Voice Lab

Local character chat and speech app. Windows is the primary tested platform,
and macOS support is experimental.

日本語版: [README.ja.md](README.ja.md)

Rinon Voice Lab connects:

- LM Studio OpenAI-compatible local chat
- Irodori-TTS VoiceDesign speech generation
- Editable 1P/2P character profiles
- Character portraits and expression variants
- Optional lightweight Web-search notes for LLM prompts
- Optional 2P remote TTS on a second PC

## Screenshots / 画面モード

### 1P Mode / 1Pモード

1Pモードは、1人のキャラクターと会話しながら、LM Studio の応答を
Irodori-TTS で読み上げる基本モードです。キャラ設定、TTS Caption、
Web検索、話速、感情スタイルを同じ画面で調整できます。

![Rinon Voice Lab 1P mode](docs/images/rinon-1p-mode.png)

### 2P Mode / 2Pキャラモード

2Pキャラモードでは、1Pと2Pのキャラクターを同じ画面に表示し、
二人の会話を交互に進められます。2人だけで話すモード、2P音声の別PC生成、
キャラクターごとの設定やTTS Captionにも対応しています。

![Rinon Voice Lab 2P mode](docs/images/rinon-2p-mode.png)

## Support / サポートについて

This is a personal experimental release. Please do not expect support,
maintenance, compatibility guarantees, or help with individual environments.
Use it as a reference implementation or a local experiment.

個人の実験的な公開物です。サポート、継続メンテナンス、環境ごとの動作保証、
個別の導入支援は期待しないでください。参考実装またはローカル実験用として
利用してください。

The app is designed to run from any install folder. It does not require a fixed
drive such as `H:`. By default, Irodori-TTS is installed next to this app:

```text
SomeFolder\
  RinonVoiceLab\
  Irodori-TTS\
```

## Requirements

- Windows 10/11
- macOS 14 or newer on Apple Silicon (experimental)
- Python 3.10 or newer
- Git
- LM Studio with the local server enabled
- A local chat model loaded in LM Studio
- NVIDIA GPU strongly recommended for Irodori-TTS
- `uv` for Irodori-TTS dependency setup

The Rinon Voice Lab wrapper uses only the Python standard library directly.
`requirements.txt` intentionally contains no app-level packages. Irodori-TTS is
installed into its own virtual environment by `tools\install_irodori_tts.ps1`.

macOS cannot use CUDA. On Apple Silicon, Irodori-TTS can use PyTorch MPS when it
is available; otherwise it falls back to CPU. MPS/CPU use `fp32`, because
Irodori-TTS `bf16` inference is CUDA/XPU-only. Voice generation may be much
slower than on an NVIDIA GPU.

## Quick Start (Windows)

1. Clone or download this repository.
2. Start LM Studio and enable the OpenAI-compatible local server.
3. Load a chat model, for example `gemma-4-12b-it`.
4. Double-click `start_chat_uv.bat`.
5. Open `http://127.0.0.1:7862/`.

If Irodori-TTS is not installed yet, `start_chat_uv.bat` runs
`tools\install_irodori_tts.ps1` automatically. The first install can take a long
time because PyTorch and model dependencies are large.

## Quick Start (macOS)

1. Start LM Studio and enable the OpenAI-compatible local server.
2. Load a chat model.
3. Open Terminal in this repository.
4. Run:

```bash
chmod +x start_chat_mac.sh tools/install_irodori_tts.sh
./start_chat_mac.sh
```

5. Open `http://127.0.0.1:7862/`.

If Irodori-TTS is not installed yet, `start_chat_mac.sh` runs
`tools/install_irodori_tts.sh`. On macOS, the installer uses
`uv sync --extra cpu`. That extra falls back to standard PyPI PyTorch wheels on
macOS, so Apple Silicon can use MPS when PyTorch reports it as available.

The macOS script uses Python 3.10 by default because PyTorch wheels may not be
available for newer Python versions such as 3.14. To override it:

```bash
IRODORI_PYTHON_VERSION=3.13 ./start_chat_mac.sh
```

To place Irodori-TTS somewhere else:

```bash
IRODORI_ROOT="$PWD/.deps/Irodori-TTS" ./start_chat_mac.sh
```

## Manual Install

Run this from the app folder:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_irodori_tts.ps1
```

The installer defaults to CUDA 12.8 wheels:

```powershell
uv sync --extra cu128
```

For CPU-only setup:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_irodori_tts.ps1 -TorchExtra cpu
```

CPU mode is mainly for testing. Voice generation can be very slow.

For macOS:

```bash
IRODORI_TORCH_EXTRA=cpu tools/install_irodori_tts.sh
```

## Configuration

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `IRODORI_ROOT` | `..\Irodori-TTS` next to this app | Irodori-TTS checkout and virtual environment |
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio OpenAI-compatible endpoint |
| `LM_STUDIO_MODEL` | `gemma-4-12b-it` | Preferred model name |
| `LM_STUDIO_CONTEXT_LIMIT` | `8200` | Visible context budget |
| `IRODORI_TORCH_EXTRA` | `cu128` | Installer torch extra: `cu128`, `cpu`, `rocm`, or `xpu` |
| `IRODORI_MODEL_DEVICE` | `auto` | Irodori-TTS model device: `auto`, `cuda`, `mps`, `cpu`, or `xpu` |
| `IRODORI_MODEL_PRECISION` | `auto` | Model precision: `auto`, `fp32`, or `bf16` |
| `IRODORI_CODEC_DEVICE` | `auto` | Codec device, usually the same as the model device |
| `IRODORI_CODEC_PRECISION` | `auto` | Codec precision. macOS uses `fp32` |

## Character Data

Characters live under `Character\<character-id>\`.

Each character folder can contain:

- `profile.txt` for hand editing
- `profile.json` for structured save/load
- `reference\` for voice reference audio
- `expressions\<slot>\` for expression images

Use the Options dialog in the app to edit character names, prompts, TTS
captions, reference audio, and expression images.

## Character Image Generation

`tools/generate_character_images.py` can generate character expression images
with the Google Nano Banana API, the OpenAI Image API, or a Stable Diffusion
WebUI-compatible local server launched from Stability Matrix. The script uses
only the Python standard library.

Preview prompts and output paths without making API calls:

```bash
python3.10 -B tools/generate_character_images.py \
  --dry-run \
  --provider nano-banana \
  --character akari \
  --expression neutral \
  --expression happy
```

Generate with Google Nano Banana:

```bash
export GEMINI_API_KEY="..."
python3.10 -B tools/generate_character_images.py \
  --provider nano-banana \
  --google-model gemini-3.1-flash-image \
  --character akari \
  --expression neutral \
  --update-profiles
```

When your Google API environment accepts optional image generation config, add
`--google-send-generation-config` with `--google-aspect-ratio` or
`--google-image-size`. Leave it off if the API rejects `responseModalities` or
`responseFormat`.

For stronger character consistency, generate `neutral` first, then use that
image as the identity reference for the other expressions:

```bash
python3.10 -B tools/generate_character_images.py \
  --provider nano-banana \
  --google-model gemini-3.1-flash-image \
  --character akari \
  --expression neutral \
  --postprocess portrait-crop \
  --update-profiles

python3.10 -B tools/generate_character_images.py \
  --provider nano-banana \
  --google-model gemini-3.1-flash-image \
  --character akari \
  --all-profile-expressions \
  --reference-expression neutral \
  --skip-reference-expression \
  --postprocess portrait-crop \
  --update-profiles
```

`--postprocess portrait-crop` normalizes saved images to a vertical 3:4 portrait,
`768x1024` by default. It also converts generated JPEG bytes to real PNG files
at the existing `.png` output paths. Add `--overwrite` to regenerate existing
files instead of skipping them.

To normalize already generated files without calling an image API:

```bash
python3.10 -B tools/generate_character_images.py \
  --provider nano-banana \
  --character akari \
  --all-profile-expressions \
  --postprocess portrait-crop \
  --postprocess-existing
```

Generate with OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
python3.10 -B tools/generate_character_images.py \
  --provider openai \
  --character akari \
  --expression neutral \
  --update-profiles
```

Generate with a Stable Diffusion WebUI-compatible API:

```bash
python3.10 -B tools/generate_character_images.py \
  --provider sd-webui \
  --sd-webui-url http://127.0.0.1:7860 \
  --character akari \
  --expression neutral \
  --update-profiles
```

## Character Reference Audio

`tools/generate_character_reference_audio.py` creates Irodori VoiceDesign
reference wav files from a character profile. Add `--reference-wav` when you
want to seed Irodori from an existing short voice sample:

```bash
python3.10 -B tools/generate_character_reference_audio.py \
  --character yuto \
  --text-file prompts/reference_voice.txt \
  --reference-wav Character/yuto/reference/yuto_voicevox_ref.wav \
  --update-profiles \
  --overwrite
```

VOICEVOX Engine can be used to create the seed wav. The script records the
engine terms checksum, selected speaker/style, and the selected speaker policy
next to the generated wav. Synthesis intentionally requires
`--accept-voice-license`; list and inspect the available voices first:

```bash
python3.10 -B tools/generate_voicevox_reference_audio.py \
  --start-engine \
  --list-speakers

python3.10 -B tools/generate_voicevox_reference_audio.py \
  --start-engine \
  --speaker-id 13 \
  --show-license
```

After confirming the selected voice policy:

```bash
python3.10 -B tools/generate_voicevox_reference_audio.py \
  --start-engine \
  --character yuto \
  --speaker-id 13 \
  --text-file prompts/reference_voice.txt \
  --accept-voice-license \
  --update-profile \
  --overwrite
```

The unified character creation command can also generate a VOICEVOX seed first
and then pass it to Irodori:

```bash
python3.10 -B tools/create_character.py \
  --id yuto \
  --name Yuto \
  --system-prompt-file prompts/yuto_system.txt \
  --tts-caption-file prompts/yuto_voice.txt \
  --design-prompt-file prompts/yuto_design.txt \
  --voice-text-file prompts/reference_voice.txt \
  --generate-voicevox-reference-audio \
  --voicevox-speaker-id 13 \
  --voicevox-accept-license \
  --voicevox-start-engine \
  --generate-reference-audio \
  --audio-overwrite
```

## Optional 2P Remote TTS

By default, both 1P and 2P voices are generated on the local Irodori-TTS
environment.

In the main toolbar, use `TTS PC` to choose the runtime mode:

- `1 PC`: generate both 1P and 2P voices on this machine.
- `2 PCs`: generate 1P locally and send only 2P voice generation to a second
  machine.

When `2 PCs` is selected, enter the second machine in `2P IP`. An IP-only value
such as `192.168.0.10` is expanded to `http://192.168.0.10:7874`. You can also
enter `192.168.0.10:7874` or a full URL.

On the second Windows machine, start the lightweight remote TTS server:

```powershell
$env:IRODORI_ROOT = "H:\AI\Irodori-TTS"
$env:LUVIA_SERVER_PORT = "7874"
python tools\remote_luvia_tts_server.py
```

The second machine must have Irodori-TTS installed and reachable from the main
machine. The remote server exposes `/health` and `/synthesize`.

On macOS or Linux, start the remote TTS server with:

```bash
IRODORI_ROOT="$PWD/../Irodori-TTS" \
LUVIA_SERVER_PORT=7874 \
IRODORI_MODEL_DEVICE=auto \
python tools/remote_luvia_tts_server.py
```

## External Speak Mode

Rinon Voice Lab can receive short text from Codex, Claude Code, or another
local tool and speak it through the open character UI.

Start Rinon Voice Lab, open `http://127.0.0.1:7862/`, then POST UTF-8 JSON:

```powershell
$body = @{
  text = "リノンから外部スピークのテストだよ。"
  emoji = "🤭"
  caption = "soft cheerful Japanese anime voice, clear pronunciation"
  speakerSlot = "main"
  steps = 8
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  http://127.0.0.1:7862/api/speak `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Common payload keys:

| Key | Purpose |
| --- | --- |
| `text` | Text to speak |
| `emoji` / `emojiStyle` | Irodori style emoji |
| `caption` / `ttsCaption` | VoiceDesign acting caption |
| `speakerSlot` | `main` or `second` |
| `referencePath` | Optional reference wav path |
| `steps` | Irodori generation steps |
| `speechRate` | `normal` or `fast` |

The browser polls `/api/speak-events` and plays new events with the normal
character animation, expression switching, panning, and audio save controls.

## Runtime Files

These local runtime files are ignored by Git and should be removed before
distributing a ZIP copy:

- `logs/`
- `profiles/`
- `saved_audio/`
- `static/generated/`
- Python caches and virtual environments

## Validation

Useful development checks:

```powershell
node --check static\app.js
$env:PYTHONDONTWRITEBYTECODE='1'
..\Irodori-TTS\.venv\Scripts\python.exe -B -m py_compile app.py tools\remote_luvia_tts_server.py
```

macOS:

```bash
node --check static/app.js
PYTHONDONTWRITEBYTECODE=1 python3.10 -B -m py_compile app.py tools/remote_luvia_tts_server.py
```

## License

MIT License. See [LICENSE](LICENSE).
