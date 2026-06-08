from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_ROOT = APP_ROOT / "Character"
DEFAULT_EXPRESSIONS = ("neutral", "happy", "surprised", "question", "worried", "soft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a character profile and optionally generate images and reference audio."
    )
    parser.add_argument("--id", required=True, help="Character id.")
    parser.add_argument("--name", required=True, help="Display name.")
    parser.add_argument("--system-prompt", default="", help="Character system prompt.")
    parser.add_argument("--system-prompt-file", default="", help="Read system prompt from a UTF-8 file.")
    parser.add_argument("--tts-caption", default="", help="VoiceDesign caption.")
    parser.add_argument("--tts-caption-file", default="", help="Read VoiceDesign caption from a UTF-8 file.")
    parser.add_argument("--design-prompt", default="", help="Image identity design prompt.")
    parser.add_argument("--design-prompt-file", default="", help="Read image identity design prompt from a UTF-8 file.")
    parser.add_argument(
        "--expression",
        action="append",
        default=[],
        help="Expression key to include. Repeat for multiple expressions.",
    )
    parser.add_argument("--reference-path", default="", help="Initial referencePath value.")
    parser.add_argument("--portrait", default="", help="Initial portrait URL.")
    parser.add_argument("--overwrite-profile", action="store_true", help="Overwrite an existing profile.")
    parser.add_argument("--generate-images", action="store_true", help="Run image generation after writing the profile.")
    parser.add_argument(
        "--image-provider",
        choices=("openai", "sd-webui", "nano-banana", "google"),
        default="nano-banana",
        help="Image backend for generate_character_images.py.",
    )
    parser.add_argument("--image-overwrite", action="store_true", help="Overwrite existing generated images.")
    parser.add_argument(
        "--image-arg",
        action="append",
        default=[],
        help="Extra argument for generate_character_images.py. Repeat for each token.",
    )
    parser.add_argument(
        "--generate-reference-audio",
        action="store_true",
        help="Run reference audio generation after writing the profile.",
    )
    parser.add_argument(
        "--generate-voicevox-reference-audio",
        action="store_true",
        help="Run VOICEVOX reference audio generation after writing the profile.",
    )
    parser.add_argument("--voice-text", default="", help="Reference text for audio generation.")
    parser.add_argument("--voice-text-file", default="", help="Read reference text from a UTF-8 file.")
    parser.add_argument("--audio-overwrite", action="store_true", help="Overwrite existing generated reference audio.")
    parser.add_argument(
        "--audio-arg",
        action="append",
        default=[],
        help="Extra argument for generate_character_reference_audio.py. Repeat for each token.",
    )
    parser.add_argument("--voicevox-speaker-id", type=int, default=None, help="VOICEVOX style id.")
    parser.add_argument("--voicevox-speaker-name", default="", help="VOICEVOX speaker name.")
    parser.add_argument("--voicevox-style-name", default="", help="VOICEVOX style name.")
    parser.add_argument(
        "--voicevox-accept-license",
        action="store_true",
        help="Pass --accept-voice-license to generate_voicevox_reference_audio.py.",
    )
    parser.add_argument(
        "--voicevox-start-engine",
        action="store_true",
        help="Pass --start-engine to generate_voicevox_reference_audio.py.",
    )
    parser.add_argument(
        "--voicevox-arg",
        action="append",
        default=[],
        help="Extra argument for generate_voicevox_reference_audio.py. Repeat for each token.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned work only.")
    args = parser.parse_args()
    if not args.system_prompt and not args.system_prompt_file:
        parser.error("Use --system-prompt or --system-prompt-file.")
    if not args.tts_caption and not args.tts_caption_file:
        parser.error("Use --tts-caption or --tts-caption-file.")
    if not args.design_prompt and not args.design_prompt_file:
        parser.error("Use --design-prompt or --design-prompt-file.")
    if (args.generate_reference_audio or args.generate_voicevox_reference_audio) and not args.voice_text and not args.voice_text_file:
        parser.error("Audio generation requires --voice-text or --voice-text-file.")
    if args.generate_voicevox_reference_audio and args.voicevox_speaker_id is None and not args.voicevox_speaker_name:
        parser.error("--generate-voicevox-reference-audio requires --voicevox-speaker-id or --voicevox-speaker-name.")
    if args.generate_voicevox_reference_audio and not args.voicevox_accept_license and not args.dry_run:
        parser.error("--generate-voicevox-reference-audio requires --voicevox-accept-license.")
    return args


def read_value(value: str, file_value: str) -> str:
    if file_value:
        return Path(file_value).expanduser().read_text(encoding="utf-8").strip()
    return str(value).strip()


def sanitize_id(value: str) -> str:
    return re.sub(r"[^0-9a-z_-]+", "_", value.strip().lower()).strip("_")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


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


def write_profile(profile: dict[str, Any], design_prompt: str, overwrite: bool, dry_run: bool) -> None:
    character_id = str(profile["id"])
    profile_dir = CHARACTER_ROOT / character_id
    profile_path = profile_dir / "profile.json"
    if profile_path.exists() and not overwrite:
        raise SystemExit(f"Profile already exists: Character/{character_id}/profile.json")
    if dry_run:
        print(f"would write: Character/{character_id}/profile.json")
        print(f"would write: Character/{character_id}/profile.txt")
        print(f"would write: Character/{character_id}/image_design.txt")
        return
    profile_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(profile_path, json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(profile_dir / "profile.txt", profile_text(profile))
    atomic_write_text(profile_dir / "image_design.txt", design_prompt.strip() + "\n")
    print(f"profile written: Character/{character_id}/profile.json")


def run_command(command: list[str], dry_run: bool) -> None:
    if dry_run:
        print("would run: " + " ".join(command))
        return
    subprocess.run(command, cwd=APP_ROOT, check=True)


def image_commands(args: argparse.Namespace, character_id: str, expressions: list[str]) -> list[list[str]]:
    base = [
        sys.executable,
        "-B",
        str(APP_ROOT / "tools" / "generate_character_images.py"),
        "--provider",
        args.image_provider,
        "--character",
        character_id,
        "--update-profiles",
        "--postprocess",
        "portrait-crop",
    ]
    if args.image_overwrite:
        base.append("--overwrite")
    base.extend(args.image_arg)
    commands: list[list[str]] = []
    if "neutral" in expressions:
        commands.append([*base, "--expression", "neutral"])
    remaining = [value for value in expressions if value != "neutral"]
    if remaining:
        command = [*base]
        for expression in remaining:
            command.extend(["--expression", expression])
        if args.image_provider in {"nano-banana", "google"} and "neutral" in expressions:
            command.extend(["--reference-expression", "neutral"])
        commands.append(command)
    return commands


def audio_command(args: argparse.Namespace, character_id: str) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(APP_ROOT / "tools" / "generate_character_reference_audio.py"),
        "--character",
        character_id,
        "--update-profiles",
    ]
    if args.audio_overwrite:
        command.append("--overwrite")
    if args.voice_text_file:
        command.extend(["--text-file", args.voice_text_file])
    else:
        command.extend(["--text", args.voice_text])
    if args.generate_voicevox_reference_audio and "--reference-wav" not in args.audio_arg:
        command.extend(
            [
                "--reference-wav",
                str(CHARACTER_ROOT / character_id / "reference" / f"{character_id}_voicevox_ref.wav"),
            ]
        )
    command.extend(args.audio_arg)
    return command


def voicevox_command(args: argparse.Namespace, character_id: str) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(APP_ROOT / "tools" / "generate_voicevox_reference_audio.py"),
        "--character",
        character_id,
    ]
    if not args.generate_reference_audio:
        command.append("--update-profile")
    if args.audio_overwrite:
        command.append("--overwrite")
    if args.voice_text_file:
        command.extend(["--text-file", args.voice_text_file])
    else:
        command.extend(["--text", args.voice_text])
    if args.voicevox_speaker_id is not None:
        command.extend(["--speaker-id", str(args.voicevox_speaker_id)])
    if args.voicevox_speaker_name:
        command.extend(["--speaker-name", args.voicevox_speaker_name])
    if args.voicevox_style_name:
        command.extend(["--style-name", args.voicevox_style_name])
    if args.voicevox_accept_license:
        command.append("--accept-voice-license")
    if args.voicevox_start_engine:
        command.append("--start-engine")
    command.extend(args.voicevox_arg)
    return command


def main() -> int:
    args = parse_args()
    character_id = sanitize_id(args.id)
    if not character_id:
        raise SystemExit("Character id is empty.")
    expressions = args.expression or list(DEFAULT_EXPRESSIONS)
    expressions = [value.strip() for value in expressions if value.strip()]
    if "neutral" not in expressions:
        expressions.insert(0, "neutral")
    system_prompt = read_value(args.system_prompt, args.system_prompt_file)
    tts_caption = read_value(args.tts_caption, args.tts_caption_file)
    design_prompt = read_value(args.design_prompt, args.design_prompt_file)
    profile = {
        "id": character_id,
        "name": args.name.strip(),
        "systemPrompt": system_prompt,
        "ttsCaption": tts_caption,
        "referencePath": args.reference_path.strip(),
        "portrait": args.portrait.strip(),
        "expressions": {expression: [] for expression in expressions},
    }
    write_profile(profile, design_prompt, args.overwrite_profile, args.dry_run)
    if args.generate_images:
        for command in image_commands(args, character_id, expressions):
            run_command(command, args.dry_run)
    if args.generate_voicevox_reference_audio:
        run_command(voicevox_command(args, character_id), args.dry_run)
    if args.generate_reference_audio:
        run_command(audio_command(args, character_id), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
