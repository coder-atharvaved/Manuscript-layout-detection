"""CLI: python inference.py --input ./data/sample --output ./results"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2

from src.io import collect_images, ensure_output_dir, write_json
from src.pipeline import detect_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect header, footer, main_text, side_text, and filler regions on manuscript pages."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or folder of manuscript pages (relative paths supported).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for annotated images and JSON metadata.",
    )
    return parser


def run(input_path: str, output_path: str) -> int:
    images = collect_images(input_path)
    out_dir = ensure_output_dir(output_path)
    failures = 0
    for image_path in images:
        try:
            result, overlay = detect_image(image_path)
            stem = Path(image_path).stem
            overlay_path = out_dir / f"{stem}_annotated.png"
            json_path = out_dir / f"{stem}.json"
            cv2.imwrite(str(overlay_path), overlay)
            write_json(result.to_dict(), json_path)
            print(f"OK  {image_path} -> {overlay_path.name}, {json_path.name} ({len(result.regions)} regions)")
        except Exception as exc:  # batch must continue on a bad page
            failures += 1
            print(f"ERR {image_path}: {exc}", file=sys.stderr)
            traceback.print_exc()
    print(f"Done. {len(images) - failures}/{len(images)} succeeded. Output: {out_dir}")
    return 1 if failures else 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run(args.input, args.output))


if __name__ == "__main__":
    main()
