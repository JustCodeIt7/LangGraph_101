from __future__ import annotations

import asyncio
import importlib.metadata
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

# A one-Python-file Space cannot use requirements.txt. Install only missing or
# mismatched model dependencies before importing Torch/Transformers/Gradio.
# GLM-OCR (`glm_ocr`) and PP-DocLayoutV3 (`pp_doclayout_v3`) are native
# Transformers 5.x architectures, so no `trust_remote_code` package pile-up
# (einops/addict/easydict) is required anymore.
_RUNTIME_REQUIREMENTS = {
    'gradio': 'gradio==6.17.3',
    'torch': 'torch==2.10.0',
    'torchvision': 'torchvision==0.25.0',
    'transformers': 'transformers==5.9.0',
    'accelerate': 'accelerate>=1.10,<2',
    'safetensors': 'safetensors>=0.5',
    'Pillow': 'Pillow==12.1.1',
    'PyMuPDF': 'PyMuPDF==1.27.2.2',
    'psutil': 'psutil==7.2.2',
}

_missing: list[str] = []
for _distribution, _requirement in _RUNTIME_REQUIREMENTS.items():
    try:
        _version = importlib.metadata.version(_distribution)
    except importlib.metadata.PackageNotFoundError:
        _missing.append(_requirement)
        continue
    if '==' in _requirement and _version != _requirement.split('==', 1)[1]:
        _missing.append(_requirement)

if _missing:
    print('Installing Space runtime dependencies:', ' '.join(_missing), flush=True)
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            '--quiet',
            '--upgrade',
            '--no-cache-dir',
            *_missing,
        ],
        check=True,
    )

import fitz
import spaces
import torch
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from gradio import Server
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from transformers import (
    AutoImageProcessor,
    AutoModelForImageTextToText,
    AutoModelForObjectDetection,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)

MODEL_ID = 'unsloth/GLM-OCR'
LAYOUT_MODEL_ID = 'PaddlePaddle/PP-DocLayoutV3_safetensors'

# GLM-OCR is prompt-routed: one task prompt per detected region.
PROMPT_TEXT = 'Text Recognition:'
PROMPT_TABLE = 'Table Recognition:'
PROMPT_FORMULA = 'Formula Recognition:'
DEFAULT_PROMPT = PROMPT_TEXT
TASK_PROMPTS = {'text': PROMPT_TEXT, 'table': PROMPT_TABLE, 'formula': PROMPT_FORMULA}

DEFAULT_MAX_TOKENS = 8192
MAX_NEW_TOKENS_LIMIT = 32768
DEFAULT_PDF_DPI = 200
DEFAULT_TIMEOUT = 1200
DEFAULT_BATCH_SIZE = 1  # pages per ZeroGPU call
MAX_PAGES_PER_REQUEST = 8
DEFAULT_REGION_BATCH_SIZE = 4  # regions per generate() call
MAX_REGION_BATCH_SIZE = 16
DEFAULT_LAYOUT_THRESHOLD = 0.3
DEFAULT_REPETITION_PENALTY = 1.1

# `layout` = PP-DocLayoutV3 crops + parallel GLM-OCR recognition (recommended).
# `page` = single whole-page GLM-OCR pass, no bounding boxes.
IMAGE_MODE_LAYOUT = 'layout'
IMAGE_MODE_PAGE = 'page'
_MODE_ALIASES = {
    'layout': IMAGE_MODE_LAYOUT,
    'gundam': IMAGE_MODE_LAYOUT,  # legacy index.html value
    'region': IMAGE_MODE_LAYOUT,
    'page': IMAGE_MODE_PAGE,
    'base': IMAGE_MODE_PAGE,  # legacy index.html value
    'full': IMAGE_MODE_PAGE,
}

# Mirrors preprocessor_config.json (`shortest_edge` is an area, not a side).
MIN_REGION_PIXELS = 12_544  # 112 x 112: keeps tiny crops legible
MAX_REGION_PIXELS = 4_194_304  # bounds ZeroGPU memory for full pages
REGION_PADDING_RATIO = 0.008
REGION_PADDING_MIN = 4

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {'.pdf'}
THUMBNAIL_MAX_WIDTH = 1024
SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_PATH = SCRIPT_DIR / 'index.html'
SESSION_ROOT = Path(tempfile.gettempdir()) / 'glm_ocr_space_sessions'
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

# PP-DocLayoutV3 emits 25 layout classes; route each to a GLM-OCR task.
# "abandon" = detected and drawn, never recognized, excluded from Markdown.
# "image"   = cropped and exported as a figure, never recognized.
LAYOUT_TASKS: dict[str, str] = {
    'doc_title': 'text',
    'paragraph_title': 'text',
    'text': 'text',
    'vertical_text': 'text',
    'abstract': 'text',
    'content': 'text',
    'reference': 'text',
    'reference_content': 'text',
    'footnote': 'text',
    'vision_footnote': 'text',
    'figure_title': 'text',
    'aside_text': 'text',
    'seal': 'text',
    'algorithm': 'text',
    'table': 'table',
    'inline_formula': 'formula',
    'display_formula': 'formula',
    'formula_number': 'formula',
    'formula': 'formula',
    'image': 'image',
    'chart': 'image',
    'header_image': 'abandon',
    'footer_image': 'abandon',
    'header': 'abandon',
    'footer': 'abandon',
    'number': 'abandon',
}
FALLBACK_TASK = 'text'
FIGURE_LABELS = {'image', 'chart', 'header_image', 'footer_image'}

LABEL_COLORS = [
    (239, 68, 68),
    (37, 99, 235),
    (5, 150, 105),
    (147, 51, 234),
    (234, 88, 12),
    (219, 39, 119),
    (13, 148, 136),
]

# Stable semantic colors across every page and request, keyed on the
# PP-DocLayoutV3 label set (legacy Unlimited-OCR keys kept for compatibility).
LABEL_COLOR_BY_TYPE: dict[str, tuple[int, int, int]] = {
    'text': (5, 150, 105),
    'vertical_text': (5, 150, 105),
    'abstract': (5, 150, 105),
    'reference_content': (5, 150, 105),
    'doc_title': (239, 68, 68),
    'paragraph_title': (239, 68, 68),
    'title': (239, 68, 68),
    'sub_title': (239, 68, 68),
    'content': (37, 99, 235),
    'reference': (37, 99, 235),
    'list': (37, 99, 235),
    'figure_title': (13, 148, 136),
    'vision_footnote': (13, 148, 136),
    'image_caption': (13, 148, 136),
    'caption': (13, 148, 136),
    'aside_text': (13, 148, 136),
    'image': (147, 51, 234),
    'chart': (147, 51, 234),
    'figure': (147, 51, 234),
    'picture': (147, 51, 234),
    'seal': (147, 51, 234),
    'header_image': (147, 51, 234),
    'footer_image': (147, 51, 234),
    'table': (234, 88, 12),
    'display_formula': (219, 39, 119),
    'inline_formula': (219, 39, 119),
    'formula_number': (219, 39, 119),
    'formula': (219, 39, 119),
    'equation': (219, 39, 119),
    'algorithm': (219, 39, 119),
    'header': (100, 116, 139),
    'footer': (100, 116, 139),
    'number': (100, 116, 139),
    'page_number': (100, 116, 139),
    'footnote': (148, 163, 184),
    'page_footnote': (148, 163, 184),
}


def normalize_mode(mode: str | None) -> str:
    return _MODE_ALIASES.get(str(mode or '').strip().casefold(), IMAGE_MODE_LAYOUT)


def normalize_prompt(prompt: str | None) -> str:
    """Return a user prompt override, or "" to use automatic task routing."""

    value = re.sub(r'^(?:<image>\s*)+', '', (prompt or '').strip())
    if not value or value.casefold() in {
        'document parsing.',
        'multi page parsing.',
        'auto',
        'automatic',
        PROMPT_TEXT.casefold(),
    }:
        return ''
    return value


def clean_markdown(raw_output: str) -> str:
    """Normalize a GLM-OCR response for the Markdown preview."""

    text = (raw_output or '').strip()
    # GLM-OCR occasionally returns literal "\n" escapes through JSON hops.
    text = text.replace('\\r\\n', '\n').replace('\\n', '\n')
    # Drop a single outer fence so headings/tables render in the Preview tab.
    text = re.sub(r'^\s*```(?:markdown|md|text|html|latex)?\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*```\s*$', '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _grounding_label_key(label: Any) -> str:
    return re.sub(r'\s+', ' ', str(label or 'box').strip()).casefold() or 'box'


def _clip_box(box: Iterable[float], width: int, height: int) -> list[int] | None:
    values = [float(value) for value in box][:4]
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _to_list(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _normalized_box(box: Sequence[int], width: int, height: int) -> list[int]:
    return [
        max(0, min(1000, round(box[0] * 1000 / max(1, width)))),
        max(0, min(1000, round(box[1] * 1000 / max(1, height)))),
        max(0, min(1000, round(box[2] * 1000 / max(1, width)))),
        max(0, min(1000, round(box[3] * 1000 / max(1, height)))),
    ]


print(f'Loading {MODEL_ID} processor from Hugging Face Hub...', flush=True)
# Left padding is required for batched decoder-only generation.
PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, padding_side='left')
if getattr(PROCESSOR, 'tokenizer', None) is not None:
    PROCESSOR.tokenizer.padding_side = 'left'

print(f'Loading {MODEL_ID} with ZeroGPU CUDA emulation...', flush=True)
MODEL = (
    AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,  # `torch_dtype` was renamed in Transformers 5
        low_cpu_mem_usage=True,
        attn_implementation='sdpa',
    )
    .eval()
    .cuda()
)

print(f'Loading {LAYOUT_MODEL_ID} layout detector...', flush=True)
LAYOUT_PROCESSOR = AutoImageProcessor.from_pretrained(LAYOUT_MODEL_ID)
LAYOUT_MODEL = AutoModelForObjectDetection.from_pretrained(LAYOUT_MODEL_ID, dtype=torch.float32).eval().cuda()
LAYOUT_ID2LABEL = {int(key): str(value) for key, value in (getattr(LAYOUT_MODEL.config, 'id2label', None) or {}).items()}
torch.set_grad_enabled(False)
print('GLM-OCR + PP-DocLayoutV3 pipeline ready.', flush=True)


class GenerationCancelled(RuntimeError):
    pass


class CancelCriteria(StoppingCriteria):
    """Stop generate() cooperatively when the UI cancels a request."""

    def __init__(self, cancel: threading.Event) -> None:
        self._cancel = cancel

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        return torch.full(
            (input_ids.shape[0],),
            bool(self._cancel.is_set()),
            dtype=torch.bool,
            device=input_ids.device,
        )


class SinkTextStreamer(TextStreamer):
    """Forward decoded deltas to the SSE queue instead of stdout."""

    def __init__(self, tokenizer: Any, sink: Any) -> None:
        self._sink = sink
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if text and self._sink is not None:
            self._sink(text)


# ---------------------------------------------------------------------------
# Stage 1: layout analysis (PP-DocLayoutV3)
# ---------------------------------------------------------------------------


def detect_layout(image: Image.Image, threshold: float) -> list[dict[str, Any]]:
    """Detect layout regions in reading order with optional polygon outlines."""

    inputs = LAYOUT_PROCESSOR(images=image, return_tensors='pt').to(LAYOUT_MODEL.device)
    outputs = LAYOUT_MODEL(**inputs)
    processed = LAYOUT_PROCESSOR.post_process_object_detection(
        outputs,
        threshold=float(threshold),
        target_sizes=[(image.height, image.width)],
    )[0]

    scores = _to_list(processed.get('scores')) or []
    labels = _to_list(processed.get('labels')) or []
    boxes = _to_list(processed.get('boxes')) or []
    polygons = _to_list(processed.get('polygon_points')) or [None] * len(boxes)

    detections: list[dict[str, Any]] = []
    for order, (score, label_id, box) in enumerate(zip(scores, labels, boxes)):
        pixel_box = _clip_box(box, image.width, image.height)
        if pixel_box is None:
            continue
        label = LAYOUT_ID2LABEL.get(int(label_id), str(label_id)).strip().casefold()
        polygon = _to_list(polygons[order]) if order < len(polygons) else None
        detections.append(
            {
                'order': order,
                'label': label,
                'task': LAYOUT_TASKS.get(label, FALLBACK_TASK),
                'score': round(float(score), 4),
                'pixel_box': pixel_box,
                'polygon': polygon,
            }
        )
    # PP-DocLayoutV3 predicts reading order directly; only fall back to a
    # top-left sweep when the detector returned nothing usable to order by.
    return detections


def grounding_from_detections(detections: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    """Group detections by label into the {boxes, pixel_boxes} UI contract."""

    grouped: dict[str, dict[str, Any]] = {}
    grounding: list[dict[str, Any]] = []
    for item in detections:
        key = _grounding_label_key(item['label'])
        target = grouped.get(key)
        if target is None:
            target = {
                'label': item['label'],
                'task': item['task'],
                'boxes': [],
                'pixel_boxes': [],
                'polygons': [],
                'scores': [],
                'orders': [],
            }
            grouped[key] = target
            grounding.append(target)
        pixel_box = item['pixel_box']
        target['boxes'].append(_normalized_box(pixel_box, width, height))
        target['pixel_boxes'].append(pixel_box)
        target['polygons'].append(item.get('polygon'))
        target['scores'].append(item['score'])
        target['orders'].append(item['order'])
    return grounding


def _grounding_color_map(
    grounding: list[dict[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    for item in grounding:
        key = _grounding_label_key(item.get('label'))
        if key in colors:
            continue
        semantic_key = (
            key
            if key in LABEL_COLOR_BY_TYPE
            else next(
                (known for known in LABEL_COLOR_BY_TYPE if key.startswith(f'{known}_')),
                None,
            )
        )
        if semantic_key:
            colors[key] = LABEL_COLOR_BY_TYPE[semantic_key]
        else:
            checksum = sum((index + 1) * ord(char) for index, char in enumerate(key))
            colors[key] = LABEL_COLORS[checksum % len(LABEL_COLORS)]
    return colors


def draw_grounding(image: Image.Image, grounding: list[dict[str, Any]]) -> Image.Image:
    """Draw layout regions with a translucent fill, plus polygons when present."""

    base = image.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    line_width = max(3, min(8, round(min(base.size) / 450)))
    colors = _grounding_color_map(grounding)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    shapes: list[tuple[tuple[int, int, int], str, tuple[int, int, int, int], Any]] = []
    for item in grounding:
        color = colors[_grounding_label_key(item.get('label'))]
        label = str(item.get('label') or 'box')[:32]
        polygons = item.get('polygons') or []
        for index, box in enumerate(item.get('pixel_boxes') or []):
            if len(box) != 4:
                continue
            polygon = polygons[index] if index < len(polygons) else None
            points = None
            if isinstance(polygon, (list, tuple)) and len(polygon) >= 3:
                try:
                    points = [(int(point[0]), int(point[1])) for point in polygon if len(point) >= 2]
                except (TypeError, ValueError):
                    points = None
            shapes.append((color, label, tuple(int(value) for value in box), points))

    for color, _, box, points in shapes:
        if points:
            overlay_draw.polygon(points, fill=(*color, 48))
        else:
            overlay_draw.rectangle(box, fill=(*color, 48))
    for color, _, (x1, y1, x2, y2), points in shapes:
        if points:
            overlay_draw.line([*points, points[0]], fill=(255, 255, 255, 245), width=line_width + 2)
            overlay_draw.line([*points, points[0]], fill=(*color, 255), width=line_width)
            continue
        overlay_draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255, 245), width=line_width + 2)
        if x2 - x1 > 2 and y2 - y1 > 2:
            overlay_draw.rectangle(
                (x1 + 1, y1 + 1, x2 - 1, y2 - 1),
                outline=(*color, 255),
                width=line_width,
            )

    output = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(output)
    for color, label, (x1, y1, _, _), _ in shapes:
        if not label:
            continue
        anchor = (x1 + 3, max(0, y1 - 18))
        draw.rectangle(draw.textbbox(anchor, label, font=font), fill=color)
        draw.text(anchor, label, fill='white', font=font)
    return output


# ---------------------------------------------------------------------------
# Stage 2: recognition (GLM-OCR)
# ---------------------------------------------------------------------------


def _fit_pixels(image: Image.Image) -> Image.Image:
    """Respect the processor's min/max pixel budget without cropping content."""

    area = max(1, image.width * image.height)
    scale = 1.0
    if area < MIN_REGION_PIXELS:
        scale = (MIN_REGION_PIXELS / area) ** 0.5
    elif area > MAX_REGION_PIXELS:
        scale = (MAX_REGION_PIXELS / area) ** 0.5
    if abs(scale - 1.0) < 0.02:
        return image
    size = (max(28, round(image.width * scale)), max(28, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    if resized is not image:
        image.close()
    return resized


def crop_region(image: Image.Image, box: Sequence[int]) -> Image.Image:
    x1, y1, x2, y2 = box
    pad_x = max(REGION_PADDING_MIN, round((x2 - x1) * REGION_PADDING_RATIO))
    pad_y = max(REGION_PADDING_MIN, round((y2 - y1) * REGION_PADDING_RATIO))
    crop = image.crop(
        (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image.width, x2 + pad_x),
            min(image.height, y2 + pad_y),
        )
    ).convert('RGB')
    return _fit_pixels(crop)


def generate_batch(
    images: list[Image.Image],
    prompts: list[str],
    max_new_tokens: int,
    repetition_penalty: float,
    cancel: threading.Event,
    sink: Any = None,
) -> list[str]:
    """Run one GLM-OCR generate() call over a batch of crops or pages."""

    if cancel.is_set():
        raise GenerationCancelled('Generation stopped')

    texts = [
        PROCESSOR.apply_chat_template(
            [
                {
                    'role': 'user',
                    'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    inputs = PROCESSOR(text=texts, images=images, return_tensors='pt', padding=True)
    inputs.pop('token_type_ids', None)
    inputs = inputs.to(MODEL.device)

    # TextStreamer only supports a single sequence; batches stream per region.
    streamer = SinkTextStreamer(PROCESSOR.tokenizer, sink) if sink is not None and len(images) == 1 else None
    generated = MODEL.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,  # greedy decoding is recommended
        repetition_penalty=float(repetition_penalty),
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([CancelCriteria(cancel)]),
    )
    if cancel.is_set():
        raise GenerationCancelled('Generation stopped')
    prompt_length = inputs['input_ids'].shape[1]
    return [text.strip() for text in PROCESSOR.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)]


def region_markdown(label: str, task: str, text: str, asset: str | None) -> str:
    """Render one recognized region into the merged Markdown document."""

    if task == 'abandon':
        return ''
    if task == 'image':
        return f'![{label}]({asset})' if asset else ''
    body = clean_markdown(text)
    if not body:
        return ''
    if label == 'doc_title':
        return f'# {body.lstrip("# ").strip()}'
    if label == 'paragraph_title':
        return f'## {body.lstrip("# ").strip()}'
    if label == 'abstract':
        return f'**Abstract.** {body}'
    if label in {'figure_title', 'vision_footnote'}:
        return f'*{body}*'
    if task == 'formula':
        if '$' in body or r'\[' in body or r'\begin' in body:
            return body
        return f'$$\n{body}\n$$'
    return body


# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------


def normalize_page_range(page_from: int | float | None, page_to: int | float | None, total: int) -> tuple[int, int]:
    if total <= 0:
        raise ValueError('PDF has no pages')
    first = max(1, min(int(page_from or 1), total))
    last = int(page_to or 0)
    last = total if last <= 0 else min(last, total)
    if first > last:
        raise ValueError(f'Invalid page range: {first} > {last}')
    return first, last


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _save_page_result(
    session_dir: Path,
    page: dict[str, Any],
    raw_output: str,
    markdown: str,
    grounding: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    latency: float,
    warning: str | None = None,
) -> dict[str, Any]:
    page_index = int(page['index'])
    prefix = f'page_{page_index + 1:04d}'
    (session_dir / f'{prefix}_raw.txt').write_text(raw_output, encoding='utf-8')
    (session_dir / f'{prefix}.md').write_text(markdown, encoding='utf-8')
    _write_json(session_dir / f'{prefix}_grounding.json', grounding)
    _write_json(session_dir / f'{prefix}_regions.json', regions)
    return {
        'page_index': page_index,
        'page_number': page['page_number'],
        'source_page_number': page.get('source_page_number', page['page_number']),
        'source_name': page.get('source_name', ''),
        'width': page['width'],
        'height': page['height'],
        'raw_output': raw_output,
        'markdown': markdown,
        'grounding': grounding,
        'regions': regions,
        'latency_seconds': round(latency, 3),
        'warning': warning,
    }


def _ensure_grounding_overlays(session_dir: Path) -> None:
    """Create downloadable overlays lazily instead of pausing between batches."""

    for grounding_path in sorted(session_dir.glob('page_*_grounding.json')):
        prefix = grounding_path.name.removesuffix('_grounding.json')
        source_path = session_dir / f'{prefix}.png'
        overlay_path = session_dir / f'{prefix}_grounding.png'
        if not source_path.is_file():
            continue
        if overlay_path.is_file() and overlay_path.stat().st_mtime_ns >= max(source_path.stat().st_mtime_ns, grounding_path.stat().st_mtime_ns):
            continue
        grounding = json.loads(grounding_path.read_text(encoding='utf-8'))
        if not grounding:
            continue
        with Image.open(source_path) as opened:
            image = opened.convert('RGB')
        overlay = draw_grounding(image, grounding)
        try:
            overlay.save(overlay_path, format='PNG')
        finally:
            overlay.close()
            image.close()


def _create_results_zip(session_dir: Path) -> Path:
    _ensure_grounding_overlays(session_dir)
    zip_path = session_dir / 'glm_ocr_streaming_results.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        for path in sorted(session_dir.iterdir()):
            if path == zip_path or not path.is_file() or path.name.startswith('source_') or path.name.endswith('_thumb.jpg'):
                continue
            compression = zipfile.ZIP_STORED if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'} else zipfile.ZIP_DEFLATED
            archive.write(path, path.name, compress_type=compression)
    return zip_path


def _materialize_source_pages(
    session_dir: Path,
    source_path: Path,
    source_name: str,
    extension: str,
    start_index: int,
    pdf_dpi: int,
    page_from: int,
    page_to: int,
) -> list[dict[str, Any]]:
    """Render/save one source incrementally, keeping at most one decoded page."""

    pages: list[dict[str, Any]] = []

    def save_page(source_page_number: int, image: Image.Image) -> None:
        page_index = start_index + len(pages)
        filename = f'page_{page_index + 1:04d}.png'
        thumbnail_filename = f'page_{page_index + 1:04d}_thumb.jpg'
        try:
            image.save(session_dir / filename, format='PNG')
            thumbnail = image.copy()
            try:
                thumbnail.thumbnail(
                    (THUMBNAIL_MAX_WIDTH, max(1, image.height)),
                    Image.Resampling.LANCZOS,
                )
                thumbnail.save(
                    session_dir / thumbnail_filename,
                    format='JPEG',
                    quality=92,
                    subsampling=0,
                    optimize=True,
                )
            finally:
                thumbnail.close()
            pages.append(
                {
                    'index': page_index,
                    'page_number': page_index + 1,
                    'source_page_number': source_page_number,
                    'source_name': source_name,
                    'filename': filename,
                    'width': image.width,
                    'height': image.height,
                    'image_url': f'/api/session/{session_dir.name}/page/{page_index}/image',
                    'thumbnail_url': f'/api/session/{session_dir.name}/page/{page_index}/thumbnail',
                    'thumbnail_filename': thumbnail_filename,
                }
            )
        finally:
            image.close()

    if extension == '.pdf':
        with fitz.open(source_path) as document:
            first, last = normalize_page_range(page_from, page_to, len(document))
            matrix = fitz.Matrix(float(pdf_dpi) / 72.0, float(pdf_dpi) / 72.0)
            for source_page_number in range(first, last + 1):
                pixmap = document[source_page_number - 1].get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
                save_page(source_page_number, image)
    else:
        with Image.open(source_path) as opened:
            save_page(1, opened.convert('RGB'))
    return pages


class StreamSettings(BaseModel):
    session_id: str
    request_id: str = Field(min_length=1, max_length=100)
    page_indices: list[int] | None = None
    mode: Literal['layout', 'page'] = IMAGE_MODE_LAYOUT
    prompt: str = ''
    model_name: str = MODEL_ID
    layout_model_name: str = LAYOUT_MODEL_ID
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=16, le=MAX_NEW_TOKENS_LIMIT)
    request_timeout: int = Field(default=DEFAULT_TIMEOUT, ge=10, le=7200)
    region_batch_size: int = Field(default=DEFAULT_REGION_BATCH_SIZE, ge=1, le=MAX_REGION_BATCH_SIZE)
    layout_threshold: float = Field(default=DEFAULT_LAYOUT_THRESHOLD, ge=0.05, le=0.95)
    repetition_penalty: float = Field(default=DEFAULT_REPETITION_PENALTY, ge=1.0, le=2.0)


def _session_dir(session_id: str) -> Path:
    try:
        normalized = str(uuid.UUID(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Invalid session ID') from exc
    path = SESSION_ROOT / normalized
    if not path.is_dir():
        raise HTTPException(status_code=404, detail='Session not found')
    return path


def _read_metadata(session_dir: Path) -> dict[str, Any]:
    path = session_dir / 'document.json'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='Session metadata not found')
    return json.loads(path.read_text(encoding='utf-8'))


app = Server(debug=True, title='GLM-OCR Hugging Face Space')
demo = app
_MODEL_LOCK = threading.Lock()
_ACTIVE_REQUESTS: dict[str, threading.Event] = {}
_SAFE_REQUEST_ID = re.compile(r'^[A-Za-z0-9_-]{1,100}$')


def _run_page_worker(
    session_dir: Path,
    pages: list[dict[str, Any]],
    settings: StreamSettings,
    cancel: threading.Event,
    emit: Any,
) -> None:
    """Layout -> region recognition -> merged Markdown, one page at a time."""

    with _MODEL_LOCK:
        if cancel.is_set():
            emit('cancelled', None)
            emit('done', None)
            return
        override_prompt = normalize_prompt(settings.prompt)
        try:
            for page in pages:
                started = time.perf_counter()
                page_index = int(page['index'])
                emit('page_started', {'page_index': page_index, 'mode': settings.mode})
                with Image.open(session_dir / page['filename']) as opened:
                    image = opened.convert('RGB')
                try:
                    detections: list[dict[str, Any]] = []
                    warning: str | None = None
                    if settings.mode == IMAGE_MODE_LAYOUT:
                        detections = detect_layout(image, settings.layout_threshold)
                        if not detections:
                            warning = 'PP-DocLayoutV3 found no regions above the threshold; fell back to a whole-page pass.'
                    grounding = grounding_from_detections(detections, image.width, image.height)
                    emit(
                        'layout',
                        {
                            'page_index': page_index,
                            'regions': len(detections),
                            'grounding': grounding,
                        },
                    )

                    prefix = f'page_{page_index + 1:04d}'
                    regions: list[dict[str, Any]] = []
                    pending: list[dict[str, Any]] = []

                    if detections:
                        for position, detection in enumerate(detections):
                            task = detection['task']
                            asset = None
                            if task == 'image':
                                asset = f'{prefix}_region_{position + 1:03d}.png'
                                crop = crop_region(image, detection['pixel_box'])
                                try:
                                    crop.save(session_dir / asset, format='PNG')
                                finally:
                                    crop.close()
                            record = {
                                'region_index': position,
                                'order': detection['order'],
                                'label': detection['label'],
                                'task': task,
                                'score': detection['score'],
                                'pixel_box': detection['pixel_box'],
                                'prompt': ('' if task in {'image', 'abandon'} else override_prompt or TASK_PROMPTS.get(task, PROMPT_TEXT)),
                                'asset': asset,
                                'text': '',
                            }
                            regions.append(record)
                            if record['prompt']:
                                pending.append(record)
                    else:
                        page_image = _fit_pixels(image.copy())
                        regions.append(
                            {
                                'region_index': 0,
                                'order': 0,
                                'label': 'page',
                                'task': 'text',
                                'score': 1.0,
                                'pixel_box': [0, 0, image.width, image.height],
                                'prompt': override_prompt or PROMPT_TEXT,
                                'asset': None,
                                'text': '',
                                '_image': page_image,
                            }
                        )
                        pending.append(regions[-1])

                    size = 1 if not detections else settings.region_batch_size
                    for offset in range(0, len(pending), size):
                        chunk = pending[offset : offset + size]
                        images = [record.pop('_image', None) or crop_region(image, record['pixel_box']) for record in chunk]
                        try:
                            outputs = generate_batch(
                                images,
                                [record['prompt'] for record in chunk],
                                settings.max_tokens,
                                settings.repetition_penalty,
                                cancel,
                                sink=lambda text, index=chunk[0]['region_index']: emit(
                                    'chunk',
                                    {
                                        'page_index': page_index,
                                        'region_index': index,
                                        'delta': text,
                                    },
                                ),
                            )
                        finally:
                            for item in images:
                                item.close()
                        for record, output in zip(chunk, outputs, strict=True):
                            record['text'] = output
                            emit(
                                'region',
                                {
                                    'page_index': page_index,
                                    'region_index': record['region_index'],
                                    'label': record['label'],
                                    'task': record['task'],
                                    'pixel_box': record['pixel_box'],
                                    'text': output,
                                },
                            )

                    blocks = [
                        region_markdown(
                            record['label'],
                            record['task'],
                            record['text'],
                            record.get('asset'),
                        )
                        for record in sorted(regions, key=lambda item: item['order'])
                    ]
                    markdown = re.sub(r'\n{3,}', '\n\n', '\n\n'.join(block for block in blocks if block)).strip()
                    raw_output = '\n\n'.join(
                        f'<|{record["label"]}|>{record["text"]}' for record in sorted(regions, key=lambda item: item['order']) if record['text']
                    )
                    for record in regions:
                        record.pop('_image', None)
                    emit(
                        'page_output',
                        {
                            'page': page,
                            'raw_output': raw_output,
                            'markdown': markdown,
                            'grounding': grounding,
                            'regions': regions,
                            'latency': time.perf_counter() - started,
                            'warning': warning,
                        },
                    )
                finally:
                    image.close()
        except GenerationCancelled:
            emit('cancelled', None)
        except BaseException as exc:
            error_traceback = traceback.format_exc()
            print(
                f'[OCR ERROR] request={settings.request_id} pages={settings.page_indices}: {exc}\n{error_traceback}',
                flush=True,
            )
            emit(
                'error',
                {
                    'message': str(exc) or exc.__class__.__name__,
                    'exception_type': exc.__class__.__name__,
                },
            )
        finally:
            emit('done', None)


@app.api(
    name='run_ocr_batch',
    stream_every=0.1,
    concurrency_limit=1,
    concurrency_id='glm-ocr-gpu',
)
@spaces.GPU(duration=120)
def run_ocr_batch(
    session_id: str,
    page_indices: list[int],
    mode: str = IMAGE_MODE_LAYOUT,
    prompt: str = '',
    max_tokens: int = DEFAULT_MAX_TOKENS,
    ngram_enabled: bool = True,
    request_id: str = 'space-request',
    batch_number: int = 1,
    total_batches: int = 1,
    request_timeout: int = DEFAULT_TIMEOUT,
    region_batch_size: int = DEFAULT_REGION_BATCH_SIZE,
    layout_threshold: float = DEFAULT_LAYOUT_THRESHOLD,
) -> Iterator[dict[str, Any]]:
    """Run one ZeroGPU batch and stream UI events through Gradio's SSE queue."""

    resolved_mode = normalize_mode(mode)
    print(
        f'[OCR START] request={request_id} batch={batch_number}/{total_batches} '
        f'session={session_id} pages={page_indices} mode={resolved_mode} '
        f'max_tokens={max_tokens}',
        flush=True,
    )
    session_dir = _session_dir(session_id)
    pages = _read_metadata(session_dir).get('pages') or []
    if not _SAFE_REQUEST_ID.fullmatch(request_id):
        yield {'event': 'error', 'data': {'message': 'Invalid request ID'}}
        return
    if not page_indices or len(page_indices) > MAX_PAGES_PER_REQUEST or any(index < 0 or index >= len(pages) for index in page_indices):
        yield {'event': 'error', 'data': {'message': 'Page index out of range'}}
        return

    batch = [pages[index] for index in page_indices]
    settings = StreamSettings(
        session_id=session_id,
        request_id=request_id,
        page_indices=page_indices,
        mode=resolved_mode,
        prompt=prompt,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        region_batch_size=region_batch_size,
        layout_threshold=layout_threshold,
        # The legacy "n-gram guard" toggle now drives GLM-OCR's recommended
        # repetition penalty instead of no_repeat_ngram_size.
        repetition_penalty=DEFAULT_REPETITION_PENALTY if ngram_enabled else 1.0,
    )
    cancel = threading.Event()
    _ACTIVE_REQUESTS[request_id] = cancel
    yield {
        'event': 'batch_started',
        'data': {
            'batch': batch_number,
            'batches': total_batches,
            'page_indices': page_indices,
            'mode': resolved_mode,
            'prompt': normalize_prompt(prompt) or 'auto (task-routed)',
        },
    }

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    worker = threading.Thread(
        target=_run_page_worker,
        args=(
            session_dir,
            batch,
            settings,
            cancel,
            lambda kind, value: events.put((kind, value)),
        ),
        daemon=True,
        name=f'glm-ocr-{request_id}',
    )
    characters = 0
    mapped_pages = 0
    started = time.perf_counter()
    results_path = session_dir / 'results.json'
    if results_path.is_file():
        try:
            results_index = json.loads(results_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            results_index = {}
    else:
        results_index = {}

    worker.start()
    try:
        while worker.is_alive() or not events.empty():
            if time.perf_counter() - started >= request_timeout:
                cancel.set()
                yield {
                    'event': 'error',
                    'data': {'message': f'Inference timed out after {request_timeout}s'},
                }
                return
            try:
                kind, value = events.get(timeout=0.05)
            except queue.Empty:
                continue

            if kind == 'chunk':
                delta = str(value.get('delta') or '')
                characters += len(delta)
                yield {
                    'event': 'delta',
                    'data': {
                        'batch': batch_number,
                        'page_index': value.get('page_index'),
                        'region_index': value.get('region_index'),
                        'delta': delta,
                        'characters': characters,
                    },
                }
            elif kind == 'region':
                characters += len(str(value.get('text') or ''))
                yield {'event': 'region_result', 'data': value}
            elif kind in {'page_started', 'layout'}:
                yield {'event': kind, 'data': value}
            elif kind == 'page_output':
                result = _save_page_result(
                    session_dir,
                    value['page'],
                    value['raw_output'],
                    value['markdown'],
                    value['grounding'],
                    value['regions'],
                    value['latency'],
                    value.get('warning'),
                )
                persisted = dict(result)
                persisted.pop('raw_output', None)
                results_index[str(result['page_index'])] = persisted
                client_result = dict(result)
                client_result.pop('raw_output', None)
                mapped_pages += 1
                yield {'event': 'page_result', 'data': client_result}
            elif kind == 'error':
                message = value.get('message') or 'Unknown inference error'
                yield {
                    'event': 'error',
                    'data': {
                        'message': (f'Hugging Face inference failed ({value.get("exception_type", "Error")}): {message}'),
                        'request_id': request_id,
                        'batch': batch_number,
                    },
                }
                return
            elif kind == 'cancelled':
                yield {'event': 'cancelled', 'data': {'message': 'Generation stopped'}}
                return

        _write_json(results_path, results_index)
        latency = time.perf_counter() - started
        yield {
            'event': 'batch_done',
            'data': {
                'batch': batch_number,
                'batches': total_batches,
                'latency_seconds': round(latency, 3),
                'mapped_pages': mapped_pages,
            },
        }
        print(
            f'[OCR DONE] request={request_id} batch={batch_number}/{total_batches} characters={characters} mapped_pages={mapped_pages} latency={latency:.3f}s',
            flush=True,
        )
    finally:
        cancel.set()
        if worker.is_alive():
            worker.join(timeout=5)
        _ACTIVE_REQUESTS.pop(request_id, None)


# Direct FastAPI inference is intentionally absent: every CUDA call must run
# inside the Gradio API function decorated with spaces.GPU above.


@app.get('/', response_class=HTMLResponse)
async def homepage() -> HTMLResponse:
    if not INDEX_PATH.is_file():
        raise HTTPException(status_code=500, detail='index.html is missing')
    defaults = json.dumps(
        {
            'apiBase': '',
            'modelName': MODEL_ID,
            'layoutModelName': LAYOUT_MODEL_ID,
            'maxTokens': DEFAULT_MAX_TOKENS,
            'pdfDpi': DEFAULT_PDF_DPI,
            'timeout': DEFAULT_TIMEOUT,
            'batchSize': DEFAULT_BATCH_SIZE,
            'regionBatchSize': DEFAULT_REGION_BATCH_SIZE,
            'layoutThreshold': DEFAULT_LAYOUT_THRESHOLD,
            'modes': [IMAGE_MODE_LAYOUT, IMAGE_MODE_PAGE],
            'taskPrompts': TASK_PROMPTS,
            'backend': 'huggingface',
        }
    )
    return HTMLResponse(INDEX_PATH.read_text(encoding='utf-8').replace('__APP_DEFAULTS__', defaults))


@app.get('/api/hf/status')
async def hf_status() -> dict[str, Any]:
    return {
        'backend': 'huggingface',
        'ready': True,
        'model': MODEL_ID,
        'layout_model': LAYOUT_MODEL_ID,
        'layout_labels': sorted(set(LAYOUT_ID2LABEL.values())),
        'modes': [IMAGE_MODE_LAYOUT, IMAGE_MODE_PAGE],
        'device': 'ZeroGPU (allocated per queued API call)',
    }


@app.post('/api/document')
async def upload_document(
    files: list[UploadFile] = File(...),
    pdf_dpi: int = Form(DEFAULT_PDF_DPI),
    page_from: int = Form(1),
    page_to: int = Form(0),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail='No files uploaded')
    if not 72 <= pdf_dpi <= 600:
        raise HTTPException(status_code=400, detail='PDF DPI must be between 72 and 600')

    session_id = str(uuid.uuid4())
    session_dir = SESSION_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    pages: list[dict[str, Any]] = []
    try:
        for source_index, upload in enumerate(files):
            source_name = Path(upload.filename or f'upload_{source_index + 1}').name
            extension = Path(source_name).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                raise ValueError(f'Unsupported file type: {extension or "unknown"}')
            source_path = session_dir / f'source_{source_index + 1:03d}{extension}'
            with source_path.open('wb') as destination:
                while chunk := await upload.read(1024 * 1024):
                    destination.write(chunk)
            pages.extend(
                await asyncio.to_thread(
                    _materialize_source_pages,
                    session_dir,
                    source_path,
                    source_name,
                    extension,
                    len(pages),
                    pdf_dpi,
                    page_from,
                    page_to,
                )
            )
        metadata = {'session_id': session_id, 'pages': pages}
        _write_json(session_dir / 'document.json', metadata)
        return metadata
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/session/{session_id}/page/{page_index}/image')
async def page_image(session_id: str, page_index: int) -> FileResponse:
    session_dir = _session_dir(session_id)
    pages = _read_metadata(session_dir).get('pages') or []
    if not 0 <= page_index < len(pages):
        raise HTTPException(status_code=404, detail='Page not found')
    return FileResponse(session_dir / pages[page_index]['filename'], media_type='image/png')


@app.get('/api/session/{session_id}/page/{page_index}/thumbnail')
async def page_thumbnail(session_id: str, page_index: int) -> FileResponse:
    session_dir = _session_dir(session_id)
    pages = _read_metadata(session_dir).get('pages') or []
    if not 0 <= page_index < len(pages):
        raise HTTPException(status_code=404, detail='Page not found')
    return FileResponse(session_dir / pages[page_index]['thumbnail_filename'], media_type='image/jpeg')


@app.post('/api/cancel/{request_id}')
async def cancel(request_id: str) -> dict[str, bool]:
    event = _ACTIVE_REQUESTS.get(request_id)
    if event:
        event.set()
    return {'cancelled': bool(event)}


@app.get('/api/session/{session_id}/download')
async def download(session_id: str) -> FileResponse:
    session_dir = _session_dir(session_id)
    archive = await asyncio.to_thread(_create_results_zip, session_dir)
    return FileResponse(
        archive,
        media_type='application/zip',
        filename='glm_ocr_streaming_results.zip',
    )


@app.delete('/api/session/{session_id}')
async def delete_session(session_id: str) -> dict[str, bool]:
    shutil.rmtree(_session_dir(session_id))
    return {'deleted': True}


if __name__ == '__main__':
    app.launch(
        server_name='0.0.0.0',
        show_error=True,
        max_file_size='1gb',
        allowed_paths=[str(SESSION_ROOT)],
    )
