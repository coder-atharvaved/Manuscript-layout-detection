# Manuscript Layout Region Detection

Python CLI that finds **header**, **footer**, **main_text**, **side_text**, and **filler** regions on historical manuscript pages (palm-leaf and paper). Original images are never modified.

## Why this approach

The assignment does not include a labeled training set. A two-day intern submission that trains YOLO on two sample photos would overfit and fail on unseen scripts. This pipeline is a **classical document-layout detector**:

1. Isolate the leaf/paper from scanner background.
2. Enhance faded ink (CLAHE + adaptive threshold).
3. Group ink blobs into text lines.
4. Separate Latin/catalog/pencil **filler** from manuscript script using position + glyph density (no OCR required).
5. Classify remaining lines by geometry relative to the main text core.

The same JSON schema can later wrap a neural detector without changing the CLI.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ recommended. CPU only.

## Run

From the project root, relative paths:

```bash
python inference.py --input ./data/sample --output ./results
```

Single file:

```bash
python inference.py --input ./data/sample/palm_leaf_sample.png --output ./results
```

A bad page logs an error and the batch continues.

## Outputs

For each input `name.png` / `name.jpg`:

| File | Contents |
|---|---|
| `results/name_annotated.png` | Color boxes, labels, confidence, legend |
| `results/name.json` | Page box + regions |

JSON shape:

```json
{
  "image": "palm_leaf_sample.png",
  "width": 4000,
  "height": 1200,
  "page_box": [x, y, w, h],
  "regions": [
    {
      "id": 1,
      "label": "main_text",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.88
    }
  ]
}
```

Boxes are `[x1, y1, x2, y2]` in **original pixel coordinates**, clipped to the image (and to the page for body classes). `filler` may sit in photo margins outside the leaf (digitization credits).

## Region rules (short)

| Label | How it is decided |
|---|---|
| `filler` | Latin-like / sparse banners at the photo top/bottom, or text outside the leaf; English catalog lines on the assignment samples |
| `main_text` | Largest manuscript-script block on the page |
| `header` | Manuscript script above the core, not a full-width body line |
| `footer` | Manuscript script below the core |
| `side_text` | Narrow manuscript strip on the left or right margin |

Confidence starts from geometry (~0.4–0.55) and rises when position and script cues agree (capped at 0.99).

## Robustness

- Uneven lighting / faded ink: CLAHE + adaptive threshold
- Scanner border / palm-leaf shape: corner-sampled background mask
- Skew: axis-aligned boxes (no warp), so coordinates stay on the original photo
- Batch folders of mixed aspect ratios

Heavy OCR (EasyOCR) is **not** a dependency so the CLI stays reproducible on a laptop.

## Layout

```
inference.py          CLI
src/page.py           page/leaf isolation
src/preprocess.py     enhancement + binarization
src/textmap.py        components and lines
src/script.py         Latin vs manuscript cues
src/regions.py        5-class grouping
src/pipeline.py       orchestration
src/visualize.py      overlays
data/sample/          assignment sample pages
```

## Tests

```bash
python -m pytest tests -q
```

## Limits

Without labels, header vs first body line and true marginalia vs a short body line can be ambiguous. Decorative non-ink filler is detected only when it forms a contrast blob. If you later get annotated pages, swap `src/regions.py` for a trained head and keep `inference.py` unchanged.
