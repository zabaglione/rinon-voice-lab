from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import generate_character_images as image_tools


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERN = "*/expressions/*/*_generated.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize generated Rinon Voice Lab character images."
    )
    parser.add_argument(
        "--root",
        default="Character",
        help="Root directory to scan. Defaults to Character.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern under --root.",
    )
    parser.add_argument(
        "--portrait-aspect",
        default="3:4",
        help="Crop aspect as WIDTH:HEIGHT.",
    )
    parser.add_argument(
        "--portrait-size",
        default="768x1024",
        help="Output size as WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--portrait-crop-y",
        type=float,
        default=0.35,
        help="Vertical crop anchor from 0.0 top to 1.0 bottom.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing images.",
    )
    args = parser.parse_args()
    try:
        image_tools.parse_aspect(args.portrait_aspect)
        image_tools.parse_size(args.portrait_size)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if not 0 <= args.portrait_crop_y <= 1:
        parser.error("--portrait-crop-y must be between 0.0 and 1.0.")
    return args


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(APP_ROOT))
    except ValueError:
        return str(path)


def generated_image_paths(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def update_sidecar(path: Path, before: tuple[int, int], after: tuple[int, int], args: argparse.Namespace) -> None:
    metadata_path = sidecar_path(path)
    payload: dict = {}
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload["normalizedImage"] = {
        "mode": "portrait-crop",
        "portraitAspect": args.portrait_aspect,
        "portraitSize": args.portrait_size,
        "portraitCropY": args.portrait_crop_y,
        "beforeSize": {"width": before[0], "height": before[1]},
        "afterSize": {"width": after[0], "height": after[1]},
        "normalizedAt": int(time.time()),
    }
    image_tools.atomic_write_text(
        metadata_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def normalize_one(path: Path, args: argparse.Namespace) -> tuple[tuple[int, int], tuple[int, int]]:
    before = image_tools.sips_image_size(path)
    tool_args = SimpleNamespace(
        postprocess="portrait-crop",
        portrait_size=args.portrait_size,
        portrait_aspect=args.portrait_aspect,
        portrait_crop_y=args.portrait_crop_y,
    )
    if not args.dry_run:
        image_tools.postprocess_image_file(path, tool_args)
        after = image_tools.sips_image_size(path)
        update_sidecar(path, before, after, args)
    else:
        target_width, target_height = image_tools.parse_size(args.portrait_size)
        after = (target_width, target_height)
    return before, after


def main() -> int:
    args = parse_args()
    root = (APP_ROOT / args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")
    paths = generated_image_paths(root, args.pattern)
    if not paths:
        print("No generated images found.")
        return 0

    changed = 0
    skipped = 0
    for path in paths:
        before, after = normalize_one(path, args)
        if before == after:
            skipped += 1
            action = "ok"
        else:
            changed += 1
            action = "normalize" if not args.dry_run else "would normalize"
        print(f"{action}: {relative(path)} {before[0]}x{before[1]} -> {after[0]}x{after[1]}")

    mode = "dry-run " if args.dry_run else ""
    print(f"{mode}complete: {changed} changed, {skipped} already normalized, {len(paths)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
