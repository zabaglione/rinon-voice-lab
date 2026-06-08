from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
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
    "vertical 3:4 portrait composition, upper-chest bust-up framing, head and shoulders visible, "
    "face centered, three-quarter angle, no full body, no waist-up, no landscape composition, "
    "clean pale studio background, crisp line art, soft cel shading, detailed expressive eyes, "
    "white futuristic jacket with cyan accents, black high-collar inner suit, "
    "white and cyan headset, polished game character asset, "
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
        "--exclude-expression",
        action="append",
        default=[],
        help="Expression key to skip. Repeat for multiple expressions.",
    )
    parser.add_argument(
        "--reference-expression",
        default="",
        help=(
            "Use an existing generated image for this expression as an identity reference. "
            "Supported by nano-banana/google."
        ),
    )
    parser.add_argument(
        "--skip-reference-expression",
        action="store_true",
        help="Do not generate the expression named by --reference-expression.",
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
        "--postprocess",
        choices=("none", "portrait-crop"),
        default=os.environ.get("CHARACTER_IMAGE_POSTPROCESS", "none"),
        help="Optional image postprocess after generation.",
    )
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="Postprocess existing generated files without calling an image API.",
    )
    parser.add_argument(
        "--portrait-aspect",
        default=os.environ.get("CHARACTER_IMAGE_PORTRAIT_ASPECT", "3:4"),
        help="Portrait crop aspect as WIDTH:HEIGHT.",
    )
    parser.add_argument(
        "--portrait-size",
        default=os.environ.get("CHARACTER_IMAGE_PORTRAIT_SIZE", "768x1024"),
        help="Portrait output size as WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--portrait-crop-y",
        type=float,
        default=float(os.environ.get("CHARACTER_IMAGE_PORTRAIT_CROP_Y", "0.35")),
        help="Vertical crop anchor from 0.0 top to 1.0 bottom.",
    )
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
        "--google-send-generation-config",
        action="store_true",
        help="Send optional Google generationConfig fields for aspect ratio and image size.",
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
    try:
        parse_size(args.portrait_size)
        parse_aspect(args.portrait_aspect)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.skip_reference_expression and not args.reference_expression:
        parser.error("--skip-reference-expression requires --reference-expression.")
    if args.reference_expression and args.provider not in {"nano-banana", "google"}:
        parser.error("--reference-expression currently requires --provider nano-banana or google.")
    if args.postprocess_existing and args.postprocess == "none":
        parser.error("--postprocess-existing requires --postprocess portrait-crop.")
    if not 0 <= args.portrait_crop_y <= 1:
        parser.error("--portrait-crop-y must be between 0.0 and 1.0.")
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


def parse_aspect(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+):(\d+)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("aspect must look like WIDTH:HEIGHT")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("aspect values must be positive")
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
            selected = sorted(sanitize_expression(key) for key in expressions)
        else:
            selected = ["neutral"]
    elif args.expression:
        selected = [sanitize_expression(value) for value in args.expression]
    else:
        selected = ["neutral"]
    excluded = {sanitize_expression(value) for value in args.exclude_expression}
    if args.skip_reference_expression and args.reference_expression:
        excluded.add(sanitize_expression(args.reference_expression))
    return [expression for expression in selected if expression not in excluded]


def character_design(character_id: str, profile: dict[str, Any]) -> str:
    design = CHARACTER_DESIGNS.get(character_id)
    if design:
        return design
    design_path = CHARACTER_ROOT / character_id / "image_design.txt"
    if design_path.exists():
        text = design_path.read_text(encoding="utf-8").strip()
        if text:
            return " ".join(text.split())
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


def build_reference_prompt(profile: dict[str, Any], expression: str, reference_expression: str) -> str:
    character_id = sanitize_id(str(profile.get("id") or "character"))
    return " ".join(
        [
            "Use the attached image as the exact identity reference for this character.",
            (
                "Keep the same face shape, hairstyle, hair color, eye color, outfit, headset, "
                "camera angle, vertical 3:4 portrait crop, upper-chest bust-up framing, "
                "background, line art, shading, and lighting."
            ),
            (
                "Change only the facial expression and small natural acting details needed for "
                f"{expression_prompt(expression)}."
            ),
            (
                "Do not redesign the character, do not change age, gender presentation, body type, "
                "hair, clothing, accessories, pose framing, or art style."
            ),
            f"Reference expression is {reference_expression}.",
            COMMON_STYLE_PROMPT + ".",
            character_design(character_id, profile) + ".",
        ]
    )


def output_path(character_id: str, expression: str, variant_count: int, variant_index: int) -> Path:
    out_dir = CHARACTER_ROOT / character_id / "expressions" / expression
    suffix = "" if variant_count == 1 else f"_{variant_index:02d}"
    return out_dir / f"{character_id}_{expression}_generated{suffix}.png"


def detect_image_extension(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def character_url(character_id: str, expression: str, path: Path) -> str:
    return f"/Character/{character_id}/expressions/{expression}/{path.name}"


def local_path_for_character_url(url: str) -> Path | None:
    text = str(url or "").strip()
    if not text.startswith("/Character/"):
        return None
    rel = Path(text.removeprefix("/Character/"))
    return (CHARACTER_ROOT / rel).resolve()


def first_profile_expression_path(
    character_id: str,
    profile: dict[str, Any],
    expression: str,
) -> Path | None:
    expressions = profile.get("expressions")
    if not isinstance(expressions, dict):
        return None
    values = expressions.get(expression) or []
    raw_values = values if isinstance(values, list) else [values]
    for value in raw_values:
        text = str(value or "").strip()
        if not text.startswith(f"/Character/{character_id}/"):
            continue
        path = local_path_for_character_url(text)
        if path and path.exists():
            return path
    return None


def reference_image_path(
    character_id: str,
    profile: dict[str, Any],
    reference_expression: str,
) -> Path:
    expression = sanitize_expression(reference_expression)
    profile_path = first_profile_expression_path(character_id, profile, expression)
    if profile_path:
        return profile_path
    generated_path = output_path(character_id, expression, 1, 1)
    if generated_path.exists():
        return generated_path
    raise RuntimeError(
        "Reference image not found. Generate the reference expression first: "
        f"Character/{character_id}/expressions/{expression}/{character_id}_{expression}_generated.png"
    )


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


def encode_inline_image(path: Path) -> dict[str, str]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return {
        "mime_type": mime_type,
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


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


def google_content_parts(prompt: str, reference_path: Path | None) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    if reference_path:
        parts.append({"inline_data": encode_inline_image(reference_path)})
    return parts


def generate_google(prompt: str, args: argparse.Namespace, reference_path: Path | None = None) -> bytes:
    api_key = google_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    model = normalize_google_model(args.google_model)
    api_version = str(args.google_api_version).strip().strip("/") or "v1"
    url = GOOGLE_IMAGE_URL_TEMPLATE.format(api_version=api_version, model=model)
    payload: dict[str, Any] = {"contents": [{"parts": google_content_parts(prompt, reference_path)}]}
    if args.google_send_generation_config:
        payload["generationConfig"] = google_image_generation_config(args)
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


def sips_path() -> str:
    path = shutil.which("sips")
    if not path:
        raise RuntimeError(
            "--postprocess portrait-crop requires Pillow or macOS sips. "
            "Install Pillow or run on macOS."
        )
    return path


def sips_image_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [sips_path(), "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    width_match = re.search(r"pixelWidth:\s*(\d+)", result.stdout)
    height_match = re.search(r"pixelHeight:\s*(\d+)", result.stdout)
    if not width_match or not height_match:
        raise RuntimeError(f"Could not read image size for {path}")
    return int(width_match.group(1)), int(height_match.group(1))


def portrait_crop_geometry(
    width: int,
    height: int,
    aspect: tuple[int, int],
    crop_y: float,
) -> tuple[int, int, int, int]:
    aspect_width, aspect_height = aspect
    target_ratio = aspect_width / aspect_height
    current_ratio = width / height
    if current_ratio > target_ratio:
        crop_height = height
        crop_width = max(1, round(height * target_ratio))
        offset_x = max(0, round((width - crop_width) / 2))
        offset_y = 0
    else:
        crop_width = width
        crop_height = max(1, round(width / target_ratio))
        offset_x = 0
        offset_y = max(0, round((height - crop_height) * crop_y))
    return crop_width, crop_height, offset_x, offset_y


def postprocess_portrait_crop_with_pillow(path: Path, args: argparse.Namespace) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    target_width, target_height = parse_size(args.portrait_size)
    aspect = parse_aspect(args.portrait_aspect)
    with Image.open(path) as image:
        image = image.convert("RGB")
        crop_width, crop_height, offset_x, offset_y = portrait_crop_geometry(
            image.width,
            image.height,
            aspect,
            args.portrait_crop_y,
        )
        cropped = image.crop((offset_x, offset_y, offset_x + crop_width, offset_y + crop_height))
        resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.postprocess.png")
        resized.save(tmp_path, format="PNG")
    tmp_path.replace(path)
    return True


def postprocess_portrait_crop_with_sips(path: Path, args: argparse.Namespace) -> None:
    target_width, target_height = parse_size(args.portrait_size)
    aspect = parse_aspect(args.portrait_aspect)
    width, height = sips_image_size(path)
    crop_width, crop_height, offset_x, offset_y = portrait_crop_geometry(
        width,
        height,
        aspect,
        args.portrait_crop_y,
    )
    crop_path = path.with_name(f".{path.name}.{os.getpid()}.crop.png")
    out_path = path.with_name(f".{path.name}.{os.getpid()}.out.png")
    try:
        subprocess.run(
            [
                sips_path(),
                "-s",
                "format",
                "png",
                "--cropToHeightWidth",
                str(crop_height),
                str(crop_width),
                "--cropOffset",
                str(offset_y),
                str(offset_x),
                str(path),
                "--out",
                str(crop_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sips_path(),
                "--resampleHeightWidth",
                str(target_height),
                str(target_width),
                str(crop_path),
                "--out",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        out_path.replace(path)
    finally:
        for tmp_path in (crop_path, out_path):
            if tmp_path.exists():
                tmp_path.unlink()


def postprocess_image_file(path: Path, args: argparse.Namespace) -> None:
    if args.postprocess == "none":
        return
    if args.postprocess != "portrait-crop":
        raise RuntimeError(f"Unsupported postprocess mode: {args.postprocess}")
    if postprocess_portrait_crop_with_pillow(path, args):
        return
    postprocess_portrait_crop_with_sips(path, args)


def write_generated_image(path: Path, image_bytes: bytes, args: argparse.Namespace) -> None:
    if args.postprocess == "none":
        atomic_write_bytes(path, image_bytes)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    input_path = path.with_name(
        f".{path.name}.{os.getpid()}.input{detect_image_extension(image_bytes)}"
    )
    try:
        input_path.write_bytes(image_bytes)
        postprocess_image_file(input_path, args)
        input_path.replace(path)
    finally:
        if input_path.exists():
            input_path.unlink()


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


def generate_image(prompt: str, args: argparse.Namespace, reference_path: Path | None = None) -> bytes:
    if reference_path and args.provider not in {"nano-banana", "google"}:
        raise RuntimeError("--reference-expression is currently supported only by nano-banana/google.")
    if args.provider == "openai":
        return generate_openai(prompt, args)
    if args.provider in {"nano-banana", "google"}:
        return generate_google(prompt, args, reference_path)
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
            reference_expression = sanitize_expression(args.reference_expression) if args.reference_expression else ""
            ref_path = None
            if reference_expression and expression != reference_expression:
                ref_path = reference_image_path(character_id, profile, reference_expression)
                prompt = build_reference_prompt(profile, expression, reference_expression)
            else:
                prompt = build_prompt(profile, expression)
            generated_urls: list[str] = []
            for variant_index in range(1, args.variant_count + 1):
                path = output_path(character_id, expression, args.variant_count, variant_index)
                url = character_url(character_id, expression, path)
                if args.postprocess_existing:
                    if not path.exists():
                        print(f"skip missing: {path.relative_to(APP_ROOT)}")
                        continue
                    print(f"postprocess: {path.relative_to(APP_ROOT)}")
                    if not args.dry_run:
                        postprocess_image_file(path, args)
                    generated_urls.append(url)
                    continue
                if path.exists() and not args.overwrite:
                    print(f"skip existing: {path.relative_to(APP_ROOT)}")
                    generated_urls.append(url)
                    continue
                print(f"target: {path.relative_to(APP_ROOT)}")
                if ref_path:
                    print(f"reference: {ref_path.relative_to(APP_ROOT)}")
                print(f"prompt: {prompt}")
                if args.dry_run:
                    generated_urls.append(url)
                    continue
                started = time.time()
                image_bytes = generate_image(prompt, args, ref_path)
                write_generated_image(path, image_bytes, args)
                write_metadata(
                    path,
                    {
                        "provider": args.provider,
                        "model": provider_model(args),
                        "characterId": character_id,
                        "expression": expression,
                        "prompt": prompt,
                        "referenceExpression": reference_expression or None,
                        "referencePath": str(ref_path.relative_to(APP_ROOT)) if ref_path else None,
                        "size": args.size,
                        "postprocess": args.postprocess,
                        "portraitAspect": args.portrait_aspect if args.postprocess != "none" else None,
                        "portraitSize": args.portrait_size if args.postprocess != "none" else None,
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
