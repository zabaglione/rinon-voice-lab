from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
import wave
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_ROOT = APP_ROOT / "Character"
DEFAULT_IRODORI_ROOT = APP_ROOT / ".deps" / "Irodori-TTS"
DEFAULT_CHECKPOINT = os.environ.get(
    "IRODORI_CHECKPOINT", "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
)
DEFAULT_TTS_STEPS = int(os.environ.get("IRODORI_TTS_STEPS", "8"))
DEFAULT_RUNTIME = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-character reference wav files with Irodori VoiceDesign."
    )
    parser.add_argument(
        "--character",
        action="append",
        default=[],
        help="Character id to generate. Repeat for multiple ids.",
    )
    parser.add_argument(
        "--all-characters",
        action="store_true",
        help="Generate for every Character/*/profile.json profile.",
    )
    parser.add_argument("--text", default="", help="Reference text to synthesize.")
    parser.add_argument("--text-file", default="", help="Read reference text from a UTF-8 file.")
    parser.add_argument(
        "--irodori-root",
        default=os.environ.get("IRODORI_ROOT", str(DEFAULT_IRODORI_ROOT)),
        help="Path to the Irodori-TTS checkout.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Irodori checkpoint.")
    parser.add_argument("--steps", type=int, default=DEFAULT_TTS_STEPS, help="Sampling steps.")
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Duration scale passed to Irodori.",
    )
    parser.add_argument(
        "--reference-wav",
        default="",
        help="Optional reference wav passed to Irodori speaker conditioning.",
    )
    parser.add_argument(
        "--speaker-cfg-scale",
        type=float,
        default=5.0,
        help="Speaker CFG scale used when --reference-wav is set.",
    )
    parser.add_argument(
        "--speaker-kv-scale",
        default="",
        help="Optional speaker KV scale passed to Irodori.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of candidates to generate per character.",
    )
    parser.add_argument("--seed", default="", help="Optional fixed seed.")
    parser.add_argument("--model-device", default=DEFAULT_RUNTIME)
    parser.add_argument("--model-precision", default=DEFAULT_RUNTIME)
    parser.add_argument("--codec-device", default=DEFAULT_RUNTIME)
    parser.add_argument("--codec-precision", default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--pitch-semitones",
        type=float,
        default=0.0,
        help="Post-process generated wav pitch by semitones. Negative values lower the voice.",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face network checks and downloads instead of offline cache mode.",
    )
    parser.add_argument(
        "--output-template",
        default="{id}_ref.wav",
        help="Output file name template. Available fields: id, index.",
    )
    parser.add_argument(
        "--update-profiles",
        action="store_true",
        help="Update profile.json and profile.txt referencePath values.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing wav files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs only.")
    args = parser.parse_args()
    if not args.all_characters and not args.character:
        parser.error("Use --character ID or --all-characters.")
    if args.steps < 1 or args.steps > 120:
        parser.error("--steps must be between 1 and 120.")
    if args.num_candidates < 1 or args.num_candidates > 8:
        parser.error("--num-candidates must be between 1 and 8.")
    if args.duration_scale <= 0:
        parser.error("--duration-scale must be positive.")
    if args.speaker_cfg_scale < 0:
        parser.error("--speaker-cfg-scale must be zero or greater.")
    if args.pitch_semitones < -12.0 or args.pitch_semitones > 12.0:
        parser.error("--pitch-semitones must be between -12 and 12.")
    if not args.text and not args.text_file:
        parser.error("Use --text or --text-file.")
    return args


def sanitize_id(value: str) -> str:
    return re.sub(r"[^0-9a-z_-]+", "_", value.strip().lower()).strip("_")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_character_profiles() -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for path in sorted(CHARACTER_ROOT.glob("*/profile.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        character_id = sanitize_id(str(data.get("id") or path.parent.name))
        if character_id:
            profiles[character_id] = data
    return profiles


def selected_profiles(args: argparse.Namespace, profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if args.all_characters:
        return [profiles[key] for key in sorted(profiles)]
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw_id in args.character:
        character_id = sanitize_id(raw_id)
        profile = profiles.get(character_id)
        if profile is None:
            missing.append(raw_id)
        else:
            selected.append(profile)
    if missing:
        raise SystemExit(f"Unknown character id: {', '.join(missing)}")
    return selected


def resolve_text(args: argparse.Namespace) -> str:
    if args.text_file:
        text = Path(args.text_file).expanduser().read_text(encoding="utf-8")
    else:
        text = str(args.text)
    text = " ".join(text.split())
    if not text:
        raise SystemExit("Reference text is empty.")
    return text


def resolve_reference_wav(value: str) -> Path | None:
    if not str(value or "").strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = APP_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"Reference wav not found: {path}")
    if path.suffix.lower() != ".wav":
        raise SystemExit(f"Reference audio must be a wav file: {path}")
    validate_wav(path)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_text(profile: dict[str, Any]) -> str:
    expressions = profile.get("expressions")
    if not isinstance(expressions, dict):
        expressions = {}
    lines = [
        "# Rinon Voice Lab character profile",
        "# Edit this file, then reload characters from Options.",
        f"id: {profile.get('id', '')}",
        f"name: {profile.get('name', '')}",
        f"referencePath: {profile.get('referencePath', '')}",
        f"portrait: {profile.get('portrait', '')}",
        "",
        "[systemPrompt]",
        str(profile.get("systemPrompt") or ""),
        "",
        "[ttsCaption]",
        str(profile.get("ttsCaption") or ""),
        "",
        "[expressions]",
    ]
    for key in sorted(expressions):
        values = expressions.get(key) or []
        raw_values = values if isinstance(values, list) else [values]
        lines.append(f"{key}=" + "|".join(str(value) for value in raw_values if str(value or "").strip()))
    lines.append("")
    return "\n".join(lines)


def save_profile(profile: dict[str, Any]) -> None:
    character_id = sanitize_id(str(profile.get("id") or ""))
    if not character_id:
        raise RuntimeError("Cannot save profile without character id.")
    profile_dir = CHARACTER_ROOT / character_id
    atomic_write_text(
        profile_dir / "profile.json",
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(profile_dir / "profile.txt", profile_text(profile))


def is_auto(value: str) -> bool:
    return str(value or "").strip().lower() in {"", "auto", "default"}


def load_irodori_module(irodori_root: Path):
    if not irodori_root.exists():
        raise RuntimeError(f"Irodori root not found: {irodori_root}")
    sys.path.insert(0, str(irodori_root))
    old_cwd = Path.cwd()
    os.chdir(irodori_root)
    try:
        import gradio_app_voicedesign as module

        quiet_irodori_watermark_warnings()
        return module
    finally:
        os.chdir(old_cwd)


def irodori_python_path(irodori_root: Path) -> Path:
    override = os.environ.get("IRODORI_PYTHON", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        return irodori_root / ".venv" / "Scripts" / "python.exe"
    return irodori_root / ".venv" / "bin" / "python"


def maybe_reexec_with_irodori_python(irodori_root: Path) -> None:
    if os.environ.get("RINON_IRODORI_AUDIO_REEXEC") == "1":
        return
    python_path = irodori_python_path(irodori_root)
    if not python_path.exists():
        return
    venv_root = (irodori_root / ".venv").resolve()
    if Path(sys.prefix).resolve() == venv_root:
        return
    os.environ["RINON_IRODORI_AUDIO_REEXEC"] = "1"
    os.execv(str(python_path), [str(python_path), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])


def configure_huggingface_mode(args: argparse.Namespace) -> None:
    if args.allow_downloads:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def quiet_irodori_watermark_warnings() -> None:
    try:
        import irodori_tts.inference_runtime as inference_runtime
    except Exception:
        return

    base_watermarker = inference_runtime.SilentCipherWatermarker
    if getattr(base_watermarker, "__rinon_quiet__", False):
        return

    class QuietSilentCipherWatermarker(base_watermarker):
        __rinon_quiet__ = True

        def __init__(self, *args, **kwargs) -> None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`.*",
                    category=FutureWarning,
                )
                super().__init__(*args, **kwargs)

        def encode_batch(self, audios: list, *, sample_rate: int) -> list:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`.*",
                    category=FutureWarning,
                )
                return super().encode_batch(audios, sample_rate=sample_rate)

    inference_runtime.SilentCipherWatermarker = QuietSilentCipherWatermarker


def runtime_settings(args: argparse.Namespace) -> dict[str, str]:
    from irodori_tts.inference_runtime import (
        default_runtime_device,
        list_available_runtime_precisions,
    )

    model_device = str(args.model_device)
    codec_device = str(args.codec_device)
    if is_auto(model_device) or is_auto(codec_device):
        default_device = default_runtime_device()
        if is_auto(model_device):
            model_device = default_device
        if is_auto(codec_device):
            codec_device = default_device

    def precision_for(device: str, requested: str) -> str:
        if not is_auto(requested):
            return str(requested).strip().lower()
        choices = list_available_runtime_precisions(device)
        device_type = str(device).split(":", 1)[0].lower()
        if device_type in {"cuda", "xpu"} and "bf16" in choices:
            return "bf16"
        return choices[0] if choices else "fp32"

    return {
        "modelDevice": model_device.strip().lower(),
        "modelPrecision": precision_for(model_device, str(args.model_precision)),
        "codecDevice": codec_device.strip().lower(),
        "codecPrecision": precision_for(codec_device, str(args.codec_precision)),
    }


def output_path(character_id: str, args: argparse.Namespace, index: int) -> Path:
    name = args.output_template.format(id=character_id, index=f"{index:03d}")
    if args.num_candidates > 1 and "{index" not in args.output_template:
        path = Path(name)
        name = f"{path.stem}_{index:03d}{path.suffix or '.wav'}"
    if Path(name).suffix.lower() != ".wav":
        raise RuntimeError("--output-template must produce a .wav file.")
    return CHARACTER_ROOT / character_id / "reference" / name


def parse_saved_paths(detail: str) -> list[str]:
    paths = [match.group(2).strip() for match in re.finditer(r"saved\[(\d+)\]:\s*(.+)", detail)]
    if paths:
        return paths
    match = re.search(r"saved:\s*(.+)", detail, re.IGNORECASE)
    return [match.group(1).strip()] if match else []


def validate_wav(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        frames = reader.getnframes()
        sample_width = reader.getsampwidth()
    if channels < 1 or sample_rate < 8000 or frames < 1 or sample_width < 1:
        raise RuntimeError(f"Invalid wav file: {path}")
    return {
        "channels": channels,
        "sampleRate": sample_rate,
        "frames": frames,
        "duration": round(frames / sample_rate, 3),
    }


def atempo_filter(value: float) -> str:
    parts: list[float] = []
    remaining = value
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    parts.append(remaining)
    return ",".join(f"atempo={part:.8f}" for part in parts)


def apply_pitch_shift(path: Path, semitones: float) -> bool:
    if math.isclose(float(semitones), 0.0, abs_tol=0.001):
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("--pitch-semitones requires ffmpeg.")
    stats = validate_wav(path)
    sample_rate = int(stats["sampleRate"])
    channels = int(stats["channels"])
    pitch_factor = 2 ** (float(semitones) / 12.0)
    shifted_rate = max(1, int(round(sample_rate * pitch_factor)))
    tempo_factor = 1.0 / pitch_factor
    audio_filter = f"asetrate={shifted_rate},aresample={sample_rate},{atempo_filter(tempo_factor)}"
    tmp = path.with_name(f".{path.stem}.pitch{path.suffix}")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-af",
            audio_filter,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(path)
    return True


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    profile: dict[str, Any],
    text: str,
    stats: dict[str, int | float],
    elapsed: float,
    pitch_shifted: bool,
    reference_wav: Path | None,
) -> None:
    payload = {
        "provider": "irodori-tts",
        "checkpoint": args.checkpoint,
        "characterId": sanitize_id(str(profile.get("id") or "")),
        "ttsCaption": str(profile.get("ttsCaption") or ""),
        "textSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "textLength": len(text),
        "steps": int(args.steps),
        "durationScale": float(args.duration_scale),
        "numCandidates": int(args.num_candidates),
        "referenceWav": (
            str(reference_wav.relative_to(APP_ROOT)) if reference_wav and reference_wav.is_relative_to(APP_ROOT) else str(reference_wav or "")
        ),
        "referenceWavSha256": file_sha256(reference_wav) if reference_wav else "",
        "speakerCfgScale": float(args.speaker_cfg_scale),
        "speakerKvScale": str(args.speaker_kv_scale),
        "pitchSemitones": float(args.pitch_semitones),
        "pitchShifted": bool(pitch_shifted),
        "audio": stats,
        "elapsed": round(elapsed, 3),
        "createdAt": int(time.time()),
    }
    atomic_write_text(path.with_suffix(path.suffix + ".json"), json.dumps(payload, indent=2) + "\n")


def generate_for_profile(
    module: Any,
    runtime: dict[str, str],
    profile: dict[str, Any],
    text: str,
    reference_wav: Path | None,
    args: argparse.Namespace,
) -> list[Path]:
    character_id = sanitize_id(str(profile.get("id") or ""))
    caption = str(profile.get("ttsCaption") or "").strip()
    if not character_id:
        raise RuntimeError("Profile has no character id.")
    if not caption:
        raise RuntimeError(f"Profile has no ttsCaption: {character_id}")
    targets = [output_path(character_id, args, index) for index in range(1, args.num_candidates + 1)]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        for path in existing:
            print(f"skip existing: {path.relative_to(APP_ROOT)}")
        if args.update_profiles:
            profile["referencePath"] = f"/Character/{character_id}/reference/{targets[0].name}"
            save_profile(profile)
            print(f"profile updated: Character/{character_id}/profile.json")
        return targets
    for path in targets:
        print(f"target: {path.relative_to(APP_ROOT)}")
    if args.dry_run:
        return targets

    old_cwd = Path.cwd()
    os.chdir(Path(args.irodori_root).expanduser().resolve())
    try:
        started = time.time()
        result = module._run_generation(
            args.checkpoint,
            runtime["modelDevice"],
            runtime["modelPrecision"],
            runtime["codecDevice"],
            runtime["codecPrecision"],
            text,
            caption,
            str(reference_wav) if reference_wav else None,
            int(args.steps),
            int(args.num_candidates),
            str(args.seed),
            "",
            float(args.duration_scale),
            "linear",
            -1.0,
            "independent",
            3.0,
            4.0,
            float(args.speaker_cfg_scale),
            "",
            0.0,
            1.0,
            True,
            str(args.speaker_kv_scale),
            "",
            "",
            "",
            "",
            "",
            "",
        )
        elapsed = time.time() - started
    finally:
        os.chdir(old_cwd)

    detail = str(result[-2])
    saved_paths = parse_saved_paths(detail)
    if len(saved_paths) != len(targets):
        raise RuntimeError(f"Expected {len(targets)} wav paths, found {len(saved_paths)}.")
    output_paths: list[Path] = []
    irodori_root = Path(args.irodori_root).expanduser().resolve()
    for saved, target in zip(saved_paths, targets):
        source = (irodori_root / saved).resolve()
        if not source.exists():
            raise RuntimeError(f"Generated wav not found: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        pitch_shifted = apply_pitch_shift(target, float(args.pitch_semitones))
        stats = validate_wav(target)
        write_metadata(target, args, profile, text, stats, elapsed, pitch_shifted, reference_wav)
        print(f"saved: {target.relative_to(APP_ROOT)} ({elapsed:.1f}s)")
        output_paths.append(target)
    if args.update_profiles:
        profile["referencePath"] = f"/Character/{character_id}/reference/{targets[0].name}"
        save_profile(profile)
        print(f"profile updated: Character/{character_id}/profile.json")
    return output_paths


def main() -> int:
    args = parse_args()
    args.irodori_root = str(Path(args.irodori_root).expanduser().resolve())
    configure_huggingface_mode(args)
    profiles = load_character_profiles()
    if not profiles:
        raise SystemExit("No character profiles found.")
    selected = selected_profiles(args, profiles)
    text = resolve_text(args)
    reference_wav = resolve_reference_wav(args.reference_wav)
    if args.dry_run:
        for profile in selected:
            character_id = sanitize_id(str(profile.get("id") or ""))
            for index in range(1, args.num_candidates + 1):
                print(f"target: {output_path(character_id, args, index).relative_to(APP_ROOT)}")
        print("dry run complete")
        return 0
    maybe_reexec_with_irodori_python(Path(args.irodori_root))
    module = load_irodori_module(Path(args.irodori_root))
    runtime = runtime_settings(args)
    for profile in selected:
        generate_for_profile(module, runtime, profile, text, reference_wav, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
