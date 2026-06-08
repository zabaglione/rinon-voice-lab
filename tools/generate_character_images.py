from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_ROOT = APP_ROOT / "Character"
OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"
GOOGLE_IMAGE_URL_TEMPLATE = "https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"
DEFAULT_SD_WEBUI_URL = "http://127.0.0.1:7860"
DEFAULT_GOOGLE_IMAGE_MODEL = os.environ.get(
    "GEMINI_IMAGE_MODEL",
    os.environ.get("GOOGLE_IMAGE_MODEL", "gemini-3.1-flash-image"),
)

COMMON_STYLE_PROMPT = (
    "anime visual novel character portrait, adult character, safe for work, "
    "centered bust shot, 3/4 view, clean pale studio background, crisp line art, "
    "soft cel shading, detailed expressive eyes, white futuristic jacket with cyan accents, "
    "black high-collar inner suit, white and cyan headset, polished game character asset, "
    "no text, no logo, no watermark"
)

SD_NEGATIVE_PROMPT = (
    "child, teen, underage, nude, explicit, low quality, blurry, deformed, bad anatomy, "
    "extra fingers, missing fingers, extra arms, cropped head, duplicate face, text, logo, watermark"
)

CHARACTER_DESIGNS = {
    "rinon": (
        "adult Japanese anime woman, short navy-blue bob haircut with long side strands, "
        "bright blue eyes, warm approachable smile, playful little-devil charm"
    ),
    "luvia": (
        "adult Japanese anime woman, long vivid red hair in a high ponytail, red eyes, "
        "confident sharp smile, lively teasing presence"
    ),
    "akari": (
        "adult Japanese anime woman, coral-orange medium hair with a practical side clip, "
        "amber eyes, bright energetic creator personality, upbeat playful confidence"
    ),
    "kaede": (
        "adult Japanese anime woman, soft chestnut long hair, gentle hazel eyes, "
        "warm mature counselor presence, calm reassuring smile"
    ),
    "makoto": (
        "adult Japanese anime man, neat dark brown hair, kind brown eyes, thin rectangular glasses, "
        "warm teacher presence, patient reassuring expression"
    ),
    "mio": (
        "adult Japanese anime woman, soft lavender-gray wavy hair, dreamy violet eyes, "
        "quiet storyteller presence, lyrical and gentle expression"
    ),
    "ren": (
        "adult Japanese anime man, charcoal black hair with silver highlights, sharp gray eyes, "
        "cool critic presence, composed dry sarcasm"
    ),
    "sena": (
        "adult Japanese anime woman, sleek teal-black straight hair, cool blue-green eyes, "
        "intelligent researcher presence, restrained curiosity"
    ),
    "shion": (
        "androgynous adult Japanese anime character, soft silver-violet short hair, calm gray eyes, "
        "quiet archivist presence, reflective delicate expression"
    ),
    "yuto": (
        "adult Japanese anime man, short black hair, calm dark eyes, lightweight engineer glasses, "
        "analytical engineer presence, composed helpful expression"
    ),
}

EXPRESSION_PROMPTS = {
    "angry": "controlled angry expression, narrowed eyes, strong posture",
    "breathless": "slightly flushed after rushing, catching breath, still composed",
    "broadcast": "clear presenter expression, speaking to an audience",
    "cough": "polite small cough gesture, one hand near mouth",
    "determined": "determined expression, focused eyes, confident posture",
    "echo": "distant reflective expression, soft voice moment",
    "exasperated": "mildly exasperated expression, raised eyebrow, dry reaction",
    "fast": "energetic fast-talking expression, lively eyes",
    "gasp": "small surprised gasp, widened eyes, safe expression",
    "happy": "bright happy smile, friendly open expression",
    "humming": "gentle humming expression, relaxed closed-mouth smile",
    "laughing": "natural laugh, cheerful open expression",
    "muffled": "playful closed-mouth reaction, cheeks slightly puffed, safe expression",
    "narration": "calm narrator expression, thoughtful eyes, composed posture",
    "neutral": "neutral attentive expression, slight natural smile",
    "pause": "quiet pause, reflective eyes, relaxed mouth",
    "phone": "speaking through a small headset microphone, attentive expression",
    "pleading": "soft pleading expression, gentle eyes, respectful posture",
    "question": "curious questioning expression, slight head tilt",
    "sad": "subtle sad expression, downcast gentle eyes",
    "serious": "serious focused expression, calm intensity",
    "shy": "shy smile, slight blush, averted gaze",
    "sigh": "soft sigh, tired but gentle expression",
    "sleepy": "sleepy expression, relaxed eyelids, calm posture",
    "smug": "smug confident expression, small knowing smile",
    "sniff": "small sniffle expression, vulnerable but composed",
    "soft": "soft gentle expression, warm eyes",
    "strong": "strong confident expression, steady gaze",
    "surprised": "surprised expression, widened eyes, natural reaction",
    "swallow": "brief thoughtful pause with a small drink, polite composed expression",
    "teasing": "playful teasing smile, mischievous eyes",
    "tender": "tender caring expression, warm gentle smile",
    "throat": "politely clearing throat, composed expression",
    "thoughtful": "thoughtful expression, hand near chin, focused eyes",
    "worried": "worried expression, gentle concern, soft eyes",
    "yawn": "small sleepy yawn, relaxed safe expression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Rinon Voice Lab character expression images."
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "sd-webui", "nano-banana", "google"),
        default="openai",
        help=(
            "Image backend. Use nano-banana/google for Gemini image models, or "
            "sd-webui for a Stability Matrix Stable Diffusion WebUI package."
        ),
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
    parser.add_argument(
        "--expression",
        action="append",
        default=[],
        help="Expression key to generate. Repeat for multiple expressions. Defaults to neutral.",
    )
    parser.add_argument(
        "--all-profile-expressions",
        action="store_true",
        help="Generate every expression listed in each selected character profile.",
    )
    parser.add_argument(
        "--variant-count",
        type=int,
        default=1,
        help="Number of files to generate per character/expression.",
    )
    parser.add_argument("--size", default="1024x1024", help="Image size, for example 1024x1024.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned prompts without API calls.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument(
        "--update-profiles",
        action="store_true",
        help="Replace generated expression paths in profile.json and profile.txt.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=("replace", "append"),
        default="replace",
        help="How --update-profiles handles existing expression paths.",
    )
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5"),
        help="OpenAI Image API model.",
    )
    parser.add_argument(
        "--openai-quality",
        choices=("low", "medium", "high", "auto"),
        default=os.environ.get("OPENAI_IMAGE_QUALITY", "medium"),
        help="OpenAI image quality.",
    )
    parser.add_argument(
        "--google-model",
        default=DEFAULT_GOOGLE_IMAGE_MODEL,
        help="Google Gemini image model.",
    )
    parser.add_argument(
        "--google-api-version",
        default=os.environ.get("GEMINI_API_VERSION", "v1"),
        help="Google Gemini API version.",
    )
    parser.add_argument(
        "--google-aspect-ratio",
        default=os.environ.get("GEMINI_IMAGE_ASPECT_RATIO", ""),
        help="Google image aspect ratio. Defaults to the ratio inferred from --size.",
    )
    parser.add_argument(
        "--google-image-size",
        choices=("1K", "2K", "4K"),
        default=os.environ.get("GEMINI_IMAGE_SIZE", "1K"),
        help="Google image size for Gemini 3 image models.",
    )
    parser.add_argument(
        "--sd-webui-url",
        default=os.environ.get("SD_WEBUI_URL", DEFAULT_SD_WEBUI_URL),
        help="Stable Diffusion WebUI base URL.",
    )
    parser.add_argument("--sd-steps", type=int, default=28, help="Stable Diffusion sampling steps.")
    parser.add_argument("--sd-cfg-scale", type=float, default=7.0, help="Stable Diffusion CFG scale.")
    parser.add_argument("--sd-sampler", default="DPM++ 2M Karras", help="Stable Diffusion sampler.")
    parser.add_argument("--sd-seed", type=int, default=-1, help="Stable Diffusion seed.")
    parser.add_argument(
        "--sd-negative-prompt",
        default=SD_NEGATIVE_PROMPT,
        help="Stable Diffusion negative prompt.",
    )
    args = parser.parse_args()
    if not args.all_characters and not args.character:
        parser.error("Use --character ID or --all-characters.")
    if args.variant_count < 1:
        parser.error("--variant-count must be 1 or greater.")
    try:
        parse_size(args.size)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 64 or height < 64:
        raise argparse.ArgumentTypeError("size is too small")
    return width, height


def google_aspect_ratio_from_size(value: str) -> str:
    width, height = parse_size(value)
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return left


def sanitize_id(value: str) -> str:
    return re.sub(r"[^0-9a-z_-]+", "_", value.strip().lower()).strip("_")


def sanitize_expression(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip()).strip("_") or "neutral"


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


def selected_expressions(args: argparse.Namespace, profile: dict[str, Any]) -> list[str]:
    if args.all_profile_expressions:
        expressions = profile.get("expressions")
        if isinstance(expressions, dict) and expressions:
            return sorted(sanitize_expression(key) for key in expressions)
    if args.expression:
        return [sanitize_expression(value) for value in args.expression]
    return ["neutral"]


def character_design(character_id: str, profile: dict[str, Any]) -> str:
    design = CHARACTER_DESIGNS.get(character_id)
    if design:
        return design
    return f"adult Japanese anime character with profile id {character_id}, distinct memorable design"


def expression_prompt(expression: str) -> str:
    return EXPRESSION_PROMPTS.get(
        expression,
        f"{expression.replace('_', ' ')} expression, natural facial acting",
    )


def build_prompt(profile: dict[str, Any], expression: str) -> str:
    character_id = sanitize_id(str(profile.get("id") or "character"))
    return ", ".join(
        [
            COMMON_STYLE_PROMPT,
            character_design(character_id, profile),
            expression_prompt(expression),
        ]
    )


def output_path(character_id: str, expression: str, variant_count: int, variant_index: int) -> Path:
    out_dir = CHARACTER_ROOT / character_id / "expressions" / expression
    suffix = "" if variant_count == 1 else f"_{variant_index:02d}"
    return out_dir / f"{character_id}_{expression}_generated{suffix}.png"


def character_url(character_id: str, expression: str, path: Path) -> str:
    return f"/Character/{character_id}/expressions/{expression}/{path.name}"


def request_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:1600]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc
    return json.loads(raw.decode("utf-8"))


def decode_image_base64(value: str) -> bytes:
    raw = value.strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def generate_openai(prompt: str, args: argparse.Namespace) -> bytes:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    payload: dict[str, Any] = {
        "model": args.openai_model,
        "prompt": prompt,
        "size": args.size,
        "quality": args.openai_quality,
        "n": 1,
    }
    result = request_json(
        OPENAI_IMAGE_URL,
        payload,
        {"Authorization": f"Bearer {api_key}"},
        args.timeout,
    )
    images = result.get("data")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"OpenAI response did not contain image data: {result}")
    b64_json = images[0].get("b64_json") if isinstance(images[0], dict) else None
    if not b64_json:
        raise RuntimeError(f"OpenAI response did not contain b64_json: {result}")
    return decode_image_base64(str(b64_json))


def google_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")).strip()


def normalize_google_model(value: str) -> str:
    return str(value or "").strip().removeprefix("models/")


def google_image_generation_config(args: argparse.Namespace) -> dict[str, Any]:
    aspect_ratio = str(args.google_aspect_ratio or "").strip() or google_aspect_ratio_from_size(args.size)
    image_config: dict[str, Any] = {"aspectRatio": aspect_ratio}
    if not normalize_google_model(args.google_model).startswith("gemini-2.5-"):
        image_config["imageSize"] = args.google_image_size
    return {
        "responseModalities": ["Image"],
        "responseFormat": {"image": image_config},
    }


def extract_google_image_bytes(result: dict[str, Any]) -> bytes:
    text_parts: list[str] = []
    finish_reasons: list[str] = []
    for candidate in result.get("candidates") or []:
        if isinstance(candidate, dict):
            finish_reason = candidate.get("finishReason")
            if finish_reason:
                finish_reasons.append(str(finish_reason))
            content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if text:
                    text_parts.append(str(text))
                inline_data = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline_data, dict) and inline_data.get("data"):
                    return decode_image_base64(str(inline_data["data"]))
    detail = {
        "finishReasons": finish_reasons,
        "text": text_parts[:3],
        "promptFeedback": result.get("promptFeedback"),
    }
    raise RuntimeError(f"Google response did not contain inline image data: {detail}")


def generate_google(prompt: str, args: argparse.Namespace) -> bytes:
    api_key = google_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    model = normalize_google_model(args.google_model)
    api_version = str(args.google_api_version).strip().strip("/") or "v1"
    url = GOOGLE_IMAGE_URL_TEMPLATE.format(api_version=api_version, model=model)
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": google_image_generation_config(args),
    }
    result = request_json(url, payload, {"x-goog-api-key": api_key}, args.timeout)
    return extract_google_image_bytes(result)


def generate_sd_webui(prompt: str, args: argparse.Namespace) -> bytes:
    width, height = parse_size(args.size)
    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": args.sd_negative_prompt,
        "width": width,
        "height": height,
        "steps": args.sd_steps,
        "cfg_scale": args.sd_cfg_scale,
        "sampler_name": args.sd_sampler,
        "seed": args.sd_seed,
        "batch_size": 1,
        "n_iter": 1,
        "save_images": False,
    }
    url = args.sd_webui_url.rstrip("/") + "/sdapi/v1/txt2img"
    result = request_json(url, payload, {}, args.timeout)
    images = result.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"Stable Diffusion WebUI response did not contain images: {result}")
    return decode_image_base64(str(images[0]))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


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


def update_profile_paths(
    profile: dict[str, Any],
    expression: str,
    urls: list[str],
    mode: str,
) -> None:
    expressions = profile.setdefault("expressions", {})
    if not isinstance(expressions, dict):
        expressions = {}
        profile["expressions"] = expressions
    if mode == "append":
        existing = expressions.get(expression) or []
        if not isinstance(existing, list):
            existing = [existing]
        merged = [str(value) for value in existing if str(value or "").strip()]
        for url in urls:
            if url not in merged:
                merged.append(url)
        expressions[expression] = merged
    else:
        expressions[expression] = urls
    if expression == "neutral" and urls:
        profile["portrait"] = urls[0]


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    atomic_write_text(
        metadata_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def generate_image(prompt: str, args: argparse.Namespace) -> bytes:
    if args.provider == "openai":
        return generate_openai(prompt, args)
    if args.provider in {"nano-banana", "google"}:
        return generate_google(prompt, args)
    return generate_sd_webui(prompt, args)


def provider_model(args: argparse.Namespace) -> str | None:
    if args.provider == "openai":
        return args.openai_model
    if args.provider in {"nano-banana", "google"}:
        return normalize_google_model(args.google_model)
    return None


def main() -> int:
    args = parse_args()
    profiles = load_character_profiles()
    if not profiles:
        raise SystemExit("No character profiles found.")
    profiles_to_save: dict[str, dict[str, Any]] = {}

    for profile in selected_profiles(args, profiles):
        character_id = sanitize_id(str(profile.get("id") or ""))
        if not character_id:
            print("Skipping profile without character id.", file=sys.stderr)
            continue
        for expression in selected_expressions(args, profile):
            prompt = build_prompt(profile, expression)
            generated_urls: list[str] = []
            for variant_index in range(1, args.variant_count + 1):
                path = output_path(character_id, expression, args.variant_count, variant_index)
                url = character_url(character_id, expression, path)
                if path.exists() and not args.overwrite:
                    print(f"skip existing: {path.relative_to(APP_ROOT)}")
                    generated_urls.append(url)
                    continue
                print(f"target: {path.relative_to(APP_ROOT)}")
                print(f"prompt: {prompt}")
                if args.dry_run:
                    generated_urls.append(url)
                    continue
                started = time.time()
                image_bytes = generate_image(prompt, args)
                atomic_write_bytes(path, image_bytes)
                write_metadata(
                    path,
                    {
                        "provider": args.provider,
                        "model": provider_model(args),
                        "characterId": character_id,
                        "expression": expression,
                        "prompt": prompt,
                        "size": args.size,
                        "createdAt": int(time.time()),
                    },
                )
                elapsed = time.time() - started
                print(f"saved: {path.relative_to(APP_ROOT)} ({elapsed:.1f}s)")
                generated_urls.append(url)
            if args.update_profiles and generated_urls and not args.dry_run:
                update_profile_paths(profile, expression, generated_urls, args.profile_mode)
                profiles_to_save[character_id] = profile

    for profile in profiles_to_save.values():
        save_profile(profile)
        print(f"profile updated: Character/{profile['id']}/profile.json")

    if args.dry_run:
        print("dry run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
