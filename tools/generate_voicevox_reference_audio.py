from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_ROOT = APP_ROOT / "Character"
DEFAULT_ENGINE_ROOT = Path("/Applications/VOICEVOX.app/Contents/Resources/vv-engine")
DEFAULT_ENGINE_COMMAND = os.environ.get("VOICEVOX_ENGINE_COMMAND", str(DEFAULT_ENGINE_ROOT / "run"))
DEFAULT_ENGINE_URL = os.environ.get("VOICEVOX_ENGINE_URL", "http://127.0.0.1:50021")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference wav files with VOICEVOX Engine."
    )
    parser.add_argument("--text", default="", help="Text to synthesize.")
    parser.add_argument("--text-file", default="", help="Read synthesis text from a UTF-8 file.")
    parser.add_argument("--output", default="", help="Output wav path.")
    parser.add_argument("--character", default="", help="Character id for default output and profile updates.")
    parser.add_argument("--update-profile", action="store_true", help="Update profile referencePath.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing wav file.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work only.")
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL, help="VOICEVOX Engine base URL.")
    parser.add_argument(
        "--engine-command",
        default=DEFAULT_ENGINE_COMMAND,
        help="Command used with --start-engine.",
    )
    parser.add_argument(
        "--start-engine",
        action="store_true",
        help="Start VOICEVOX Engine for this command and stop it afterwards.",
    )
    parser.add_argument("--startup-timeout", type=float, default=90.0, help="Engine startup timeout.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout.")
    parser.add_argument("--speaker-id", type=int, default=None, help="VOICEVOX style id.")
    parser.add_argument("--speaker-name", default="", help="VOICEVOX speaker name.")
    parser.add_argument("--style-name", default="", help="VOICEVOX style name.")
    parser.add_argument("--list-speakers", action="store_true", help="List available speakers and styles.")
    parser.add_argument(
        "--include-policies",
        action="store_true",
        help="Include speaker policies when listing speakers.",
    )
    parser.add_argument(
        "--show-license",
        action="store_true",
        help="Print engine terms metadata and selected speaker policy.",
    )
    parser.add_argument(
        "--accept-voice-license",
        action="store_true",
        help="Acknowledge VOICEVOX terms and the selected voice policy before synthesis.",
    )
    parser.add_argument("--speed-scale", type=float, default=0.92, help="AudioQuery speedScale.")
    parser.add_argument("--pitch-scale", type=float, default=-0.08, help="AudioQuery pitchScale.")
    parser.add_argument(
        "--intonation-scale",
        type=float,
        default=0.85,
        help="AudioQuery intonationScale.",
    )
    parser.add_argument("--volume-scale", type=float, default=1.0, help="AudioQuery volumeScale.")
    parser.add_argument(
        "--pre-phoneme-length",
        type=float,
        default=None,
        help="AudioQuery prePhonemeLength override.",
    )
    parser.add_argument(
        "--post-phoneme-length",
        type=float,
        default=None,
        help="AudioQuery postPhonemeLength override.",
    )
    parser.add_argument(
        "--output-sampling-rate",
        type=int,
        default=48000,
        help="AudioQuery outputSamplingRate.",
    )
    args = parser.parse_args()

    if not args.list_speakers and args.speaker_id is None and not args.speaker_name:
        parser.error("Use --speaker-id or --speaker-name, or use --list-speakers.")
    if not args.list_speakers and not args.show_license and not args.text and not args.text_file:
        parser.error("Use --text or --text-file.")
    if args.update_profile and not args.character:
        parser.error("--update-profile requires --character.")
    if args.output_sampling_rate < 8000 or args.output_sampling_rate > 192000:
        parser.error("--output-sampling-rate must be between 8000 and 192000.")
    for name in ("speed_scale", "volume_scale"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.startup_timeout <= 0 or args.timeout <= 0:
        parser.error("--startup-timeout and --timeout must be positive.")
    return args


def sanitize_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower()).strip("_")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_text(args: argparse.Namespace) -> str:
    if args.text_file:
        text = Path(args.text_file).expanduser().read_text(encoding="utf-8")
    else:
        text = str(args.text)
    text = " ".join(text.split())
    if not text:
        raise SystemExit("Synthesis text is empty.")
    return text


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        path = Path(args.output).expanduser()
        if not path.is_absolute():
            path = APP_ROOT / path
        return path.resolve()
    character_id = sanitize_id(args.character)
    if not character_id:
        raise SystemExit("Use --output or --character.")
    return (CHARACTER_ROOT / character_id / "reference" / f"{character_id}_voicevox_ref.wav").resolve()


def api_url(base_url: str, path: str, params: dict[str, Any] | None = None) -> str:
    base = base_url.rstrip("/")
    url = base + "/" + path.lstrip("/")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return url


def request_json(
    method: str,
    url: str,
    payload: Any | None = None,
    timeout: float = 120.0,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:1600]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to {url}: {exc}") from exc


def request_bytes(
    method: str,
    url: str,
    payload: Any | None = None,
    timeout: float = 120.0,
) -> bytes:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:1600]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to {url}: {exc}") from exc


def engine_version(base_url: str, timeout: float) -> str:
    value = request_json("GET", api_url(base_url, "/version"), timeout=timeout)
    return str(value)


def wait_for_engine(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            engine_version(base_url, min(5.0, timeout))
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"VOICEVOX Engine did not become ready: {last_error}")


def engine_cwd(command: list[str]) -> Path | None:
    if not command:
        return None
    path = Path(command[0]).expanduser()
    if path.exists():
        return path.resolve().parent
    return None


def start_engine(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
    if not args.start_engine:
        return None
    try:
        engine_version(args.engine_url, 2.0)
        return None
    except Exception:
        pass
    command = shlex.split(args.engine_command)
    if not command:
        raise RuntimeError("--engine-command is empty.")
    if Path(command[0]).expanduser().exists():
        command[0] = str(Path(command[0]).expanduser().resolve())
    process = subprocess.Popen(
        command,
        cwd=engine_cwd(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    try:
        wait_for_engine(args.engine_url, args.startup_timeout)
    except Exception:
        stop_engine(process)
        raise
    return process


def stop_engine(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10.0)
    except Exception:
        if process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10.0)


def read_engine_manifest(args: argparse.Namespace) -> dict[str, Any]:
    command = shlex.split(args.engine_command)
    if not command:
        return {}
    root = engine_cwd(command)
    if root is None:
        return {}
    manifest_path = root / "engine_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terms_path = root / str(manifest.get("terms_of_service") or "")
    terms_payload: dict[str, Any] = {}
    if terms_path.exists():
        terms_payload = {
            "path": str(terms_path),
            "sha256": file_sha256(terms_path),
        }
    return {
        "manifestPath": str(manifest_path),
        "manifestSha256": file_sha256(manifest_path),
        "name": manifest.get("name"),
        "brandName": manifest.get("brand_name"),
        "uuid": manifest.get("uuid"),
        "version": manifest.get("version"),
        "url": manifest.get("url"),
        "termsOfService": terms_payload,
    }


def fetch_speakers(args: argparse.Namespace) -> list[dict[str, Any]]:
    return request_json("GET", api_url(args.engine_url, "/speakers"), timeout=args.timeout)


def fetch_speaker_info(args: argparse.Namespace, speaker_uuid: str) -> dict[str, Any]:
    return request_json(
        "GET",
        api_url(
            args.engine_url,
            "/speaker_info",
            {"speaker_uuid": speaker_uuid, "resource_format": "url"},
        ),
        timeout=args.timeout,
    )


def flatten_speakers(speakers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for speaker in speakers:
        styles = speaker.get("styles") if isinstance(speaker.get("styles"), list) else []
        for style in styles:
            rows.append(
                {
                    "speakerName": speaker.get("name"),
                    "speakerUuid": speaker.get("speaker_uuid"),
                    "styleName": style.get("name"),
                    "styleId": style.get("id"),
                    "styleType": style.get("type"),
                    "version": speaker.get("version"),
                }
            )
    return rows


def select_voice(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.speaker_id is not None:
        matches = [row for row in rows if int(row["styleId"]) == int(args.speaker_id)]
    else:
        speaker_name = str(args.speaker_name).strip()
        style_name = str(args.style_name).strip()
        matches = [row for row in rows if str(row["speakerName"]) == speaker_name]
        if style_name:
            matches = [row for row in matches if str(row["styleName"]) == style_name]
    if not matches:
        raise SystemExit("Selected VOICEVOX speaker/style was not found.")
    if len(matches) > 1:
        raise SystemExit("Selected VOICEVOX speaker has multiple styles. Add --style-name or use --speaker-id.")
    return matches[0]


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def list_speakers(args: argparse.Namespace, speakers: list[dict[str, Any]]) -> None:
    rows = flatten_speakers(speakers)
    if args.include_policies:
        by_uuid: dict[str, dict[str, Any]] = {}
        for row in rows:
            speaker_uuid = str(row.get("speakerUuid") or "")
            if speaker_uuid and speaker_uuid not in by_uuid:
                by_uuid[speaker_uuid] = fetch_speaker_info(args, speaker_uuid)
            row["policy"] = by_uuid.get(speaker_uuid, {}).get("policy", "")
    print_json({"speakers": rows})


def license_payload(
    args: argparse.Namespace,
    selected: dict[str, Any],
    speaker_info: dict[str, Any],
    version: str,
) -> dict[str, Any]:
    return {
        "provider": "voicevox-engine",
        "engineUrl": args.engine_url,
        "engineVersion": version,
        "engineManifest": read_engine_manifest(args),
        "speaker": selected,
        "speakerPolicy": speaker_info.get("policy", ""),
        "speakerInfo": speaker_info,
    }


def adjust_audio_query(query: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    query["speedScale"] = float(args.speed_scale)
    query["pitchScale"] = float(args.pitch_scale)
    query["intonationScale"] = float(args.intonation_scale)
    query["volumeScale"] = float(args.volume_scale)
    query["outputSamplingRate"] = int(args.output_sampling_rate)
    query["outputStereo"] = False
    if args.pre_phoneme_length is not None:
        query["prePhonemeLength"] = float(args.pre_phoneme_length)
    if args.post_phoneme_length is not None:
        query["postPhonemeLength"] = float(args.post_phoneme_length)
    return query


def synthesize(args: argparse.Namespace, selected: dict[str, Any], text: str) -> bytes:
    speaker_id = int(selected["styleId"])
    query = request_json(
        "POST",
        api_url(args.engine_url, "/audio_query", {"text": text, "speaker": speaker_id}),
        payload=None,
        timeout=args.timeout,
    )
    query = adjust_audio_query(query, args)
    return request_bytes(
        "POST",
        api_url(args.engine_url, "/synthesis", {"speaker": speaker_id}),
        payload=query,
        timeout=args.timeout,
    )


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
        "sampleWidth": sample_width,
        "duration": round(frames / sample_rate, 3),
    }


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


def update_profile_reference(character_id: str, output_path: Path) -> None:
    profile_dir = CHARACTER_ROOT / character_id
    profile_path = profile_dir / "profile.json"
    if not profile_path.exists():
        raise SystemExit(f"Profile not found: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if output_path.is_relative_to(APP_ROOT):
        reference_path = "/" + str(output_path.relative_to(APP_ROOT))
    else:
        reference_path = str(output_path)
    profile["referencePath"] = reference_path
    atomic_write_text(profile_path, json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(profile_dir / "profile.txt", profile_text(profile))
    print(f"profile updated: Character/{character_id}/profile.json")


def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    selected: dict[str, Any],
    speaker_info: dict[str, Any],
    version: str,
    stats: dict[str, int | float],
    elapsed: float,
    text: str,
) -> None:
    payload = {
        **license_payload(args, selected, speaker_info, version),
        "voiceLicenseAccepted": bool(args.accept_voice_license),
        "characterId": sanitize_id(args.character),
        "outputPath": str(output_path.relative_to(APP_ROOT)) if output_path.is_relative_to(APP_ROOT) else str(output_path),
        "outputSha256": file_sha256(output_path),
        "textSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "textLength": len(text),
        "audio": stats,
        "adjustments": {
            "speedScale": float(args.speed_scale),
            "pitchScale": float(args.pitch_scale),
            "intonationScale": float(args.intonation_scale),
            "volumeScale": float(args.volume_scale),
            "outputSamplingRate": int(args.output_sampling_rate),
            "prePhonemeLength": args.pre_phoneme_length,
            "postPhonemeLength": args.post_phoneme_length,
        },
        "elapsed": round(elapsed, 3),
        "createdAt": int(time.time()),
    }
    atomic_write_text(output_path.with_suffix(output_path.suffix + ".json"), json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    if args.dry_run and not args.list_speakers and not args.show_license:
        output_path = resolve_output_path(args)
        if args.text or args.text_file:
            text = resolve_text(args)
            text_length = len(text)
        else:
            text_length = 0
        print(f"target: {output_path.relative_to(APP_ROOT) if output_path.is_relative_to(APP_ROOT) else output_path}")
        print_json(
            {
                "dryRun": True,
                "speakerId": args.speaker_id,
                "speakerName": args.speaker_name,
                "styleName": args.style_name,
                "characterId": sanitize_id(args.character),
                "textLength": text_length,
                "updateProfile": bool(args.update_profile),
            }
        )
        return 0
    process: subprocess.Popen[bytes] | None = None
    try:
        process = start_engine(args)
        version = engine_version(args.engine_url, args.timeout)
        speakers = fetch_speakers(args)
        if args.list_speakers:
            list_speakers(args, speakers)
            return 0

        rows = flatten_speakers(speakers)
        selected = select_voice(args, rows)
        speaker_info = fetch_speaker_info(args, str(selected["speakerUuid"]))
        if args.show_license:
            print_json(license_payload(args, selected, speaker_info, version))
            return 0

        if not args.accept_voice_license:
            raise SystemExit("Synthesis requires --accept-voice-license.")
        text = resolve_text(args)
        output_path = resolve_output_path(args)
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output already exists: {output_path}")
        print(f"target: {output_path.relative_to(APP_ROOT) if output_path.is_relative_to(APP_ROOT) else output_path}")
        if args.dry_run:
            print("dry run complete")
            return 0

        started = time.time()
        wav_bytes = synthesize(args, selected, text)
        elapsed = time.time() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        stats = validate_wav(output_path)
        write_metadata(output_path, args, selected, speaker_info, version, stats, elapsed, text)
        print(f"saved: {output_path.relative_to(APP_ROOT) if output_path.is_relative_to(APP_ROOT) else output_path} ({elapsed:.1f}s)")
        if args.update_profile:
            update_profile_reference(sanitize_id(args.character), output_path)
        return 0
    finally:
        stop_engine(process)


if __name__ == "__main__":
    raise SystemExit(main())
