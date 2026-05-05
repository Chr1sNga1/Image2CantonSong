#!/usr/bin/env python3
"""
Batch evaluate multimodal lyric generation models.

This script tests a batch of images with:
1. Qwen/Qwen2.5-VL-3B-Instruct
2. OpenGVLab/InternVL2-4B
3. OpenGVLab/InternVL2-4B + RAG

For each generated lyric result, it computes four metrics:
1. Image-lyrics alignment (CLIP)
2. Image-lyrics emotion similarity
3. Lyrics format score
4. Cantonese lyrics quality

It does NOT compute genre_alignment.

Example:
  python batch_eval_mm_models.py \
    --input-dir /userhome/cs5/u3665806/Image2CantonSong/Images \
    --output-dir outputs/batch_eval \
    --models qwen intern intern_rag \
    --line-count 8

If using llm-online emotion model:
  HF Token can be passed by:
    --hf-token hf_xxx

The script outputs:
  outputs/batch_eval/results_qwen.csv
  outputs/batch_eval/results_intern.csv
  outputs/batch_eval/results_intern_rag.csv
  outputs/batch_eval/results_all_models.csv
"""

from __future__ import annotations

import inspect
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent

sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from paths import EVAL, IMAGES, PROJECT_ROOT as PATHS_PROJECT_ROOT  # noqa: E402
from modules.mm_direct_gen import (  # noqa: E402
    generate_from_image,
    unload_mm_models,
    build_lyrics_format_instruction,
)


# ---------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------

MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "use_rag": False,
        "label": "qwen",
    },
    "intern": {
        "model_id": "OpenGVLab/InternVL2-4B",
        "use_rag": False,
        "label": "intern",
    },
    "intern_rag": {
        "model_id": "OpenGVLab/InternVL2-4B",
        "use_rag": True,
        "label": "intern_rag",
    },
}

DEFAULT_STYLE_PROMPT = (
    "female Cantonese Melancholic Classical airy vocal "
    "Piano bright vocal Pop Nostalgic Violin"
)

DEFAULT_MOOD_TEXT = "Melancholic"


# ---------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------

def load_module(module_name: str, module_path: Path) -> object:
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_image_text_similarity_module() -> object:
    return load_module(
        "clip_image_text_alignment",
        EVAL / "image_lyrics_alignment" / "clip_image_text_alignment.py",
    )


def load_image_lyrics_emotion_similarity_module() -> object:
    return load_module(
        "image_lyrics_emotion_similarity",
        EVAL / "image_lyrics_emotion" / "image_lyrics_emotion_similarity.py",
    )


def load_lyrics_format_module() -> object:
    return load_module(
        "lyrics_format_transformer_score",
        EVAL / "lyrics_format" / "lyrics_format_transformer_score.py",
    )


def load_lyrics_quality_module() -> object:
    """
    Support both possible folder names:
    - Evaluation/lyrics_quality/lyrics_quality_evaluation.py
    - Evaluation/Lyrics_quality/Lyrics_quality_evaluation.py
    """

    candidate_paths = [
        EVAL / "lyrics_quality" / "lyrics_quality_evaluation.py",
        EVAL / "Lyrics_quality" / "Lyrics_quality_evaluation.py",
    ]

    for path in candidate_paths:
        if path.exists():
            return load_module("lyrics_quality_evaluation", path)

    raise FileNotFoundError(
        "Lyrics quality module not found. Tried:\n"
        + "\n".join(str(p) for p in candidate_paths)
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def list_images(input_dir: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )


def bundle_to_dict(bundle: Any) -> dict[str, Any]:
    if hasattr(bundle, "model_dump"):
        return bundle.model_dump()

    if hasattr(bundle, "__dict__"):
        return dict(bundle.__dict__)

    raise TypeError(f"Unsupported bundle type: {type(bundle)}")


def strip_lyrics_section_tags(lyrics_text: str) -> str:
    section_tags = {
        "[verse]",
        "[chorus]",
        "[bridge]",
        "[outro]",
        "[end]",
    }

    lines = []

    for line in lyrics_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.lower() in section_tags:
            continue

        lines.append(stripped)

    return "\n".join(lines)


def apply_hf_env(hf_token: str = "", hf_llm_model: str = "") -> None:
    if hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token.strip()
        os.environ["HUGGINGFACE_API_TOKEN"] = hf_token.strip()

    if hf_llm_model.strip():
        # Do not use provider suffix such as :novita here.
        model_id = hf_llm_model.strip()
        if ":" in model_id:
            model_id = model_id.split(":", 1)[0].strip()

        os.environ["HUGGINGFACE_LLM_MODEL"] = model_id


def resolve_text_emotion_model_key(emotion_module: object, requested: str) -> str:
    """
    Accept either the internal key or the displayed model name.

    Example:
    - llm-online
    - zai-org/GLM-5
    - zai-org/GLM-5 (online)
    """

    keys = emotion_module.get_text_emotion_model_keys()

    if requested in keys:
        return requested

    requested_norm = requested.lower().replace("(online)", "").strip()

    for key in keys:
        display = emotion_module.get_text_emotion_model_display_name(key)
        display_norm = display.lower().replace("(online)", "").strip()

        if requested_norm == display_norm:
            return key

        if requested_norm in display_norm or display_norm in requested_norm:
            return key

    raise ValueError(
        f"Cannot resolve text emotion model: {requested}. "
        f"Available keys: {keys}"
    )


def safe_float(value: Any) -> float | str:
    if value is None:
        return ""

    try:
        return float(value)
    except Exception:
        return ""


def get_first(d: dict, *keys: str):
    """Return the first existing value from a dict."""
    for key in keys:
        if key in d:
            return d[key]
    return None


def extract_emotion_similarity_value(result: Any):
    """Extract emotion similarity score from different possible return formats."""

    if result is None:
        return None

    if isinstance(result, (int, float)):
        return result

    if isinstance(result, dict):
        value = get_first(
            result,
            "similarity",
            "emotion_similarity",
            "image_lyrics_emotion_similarity",
            "cosine_similarity",
            "score",
        )

        if value is not None:
            return value

        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            return get_first(
                metrics,
                "similarity",
                "emotion_similarity",
                "cosine_similarity",
                "score",
            )

    return None


def append_error(row: dict[str, Any], key: str, exc: BaseException) -> None:
    row[key] = "".join(
        traceback.format_exception_only(type(exc), exc)
    ).strip()


# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------

def compute_clip_alignment(
    similarity_module: object,
    image_bytes: bytes,
    lyrics_text: str,
) -> tuple[float | None, str]:
    try:
        score = similarity_module.score_image_text_similarity(
            image_bytes=image_bytes,
            json_input={"lyrics_text": lyrics_text},
        )
        return float(score), ""
    except Exception as exc:
        return None, traceback.format_exc()


def _call_with_supported_kwargs(func, **kwargs):
    """Call a function with only the keyword arguments it supports."""
    sig = inspect.signature(func)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in sig.parameters
    }
    return func(**supported)


def compute_emotion_similarity(
    emotion_module: object,
    image_path: Path,
    lyrics_text: str,
    text_emotion_model: str,
    top_k_image: int,
    top_k_text: int,
    image_model_type: str,
    embedding_model_name: str,
    embedding_device: str | None,
) -> tuple[dict[str, Any] | None, str]:
    try:
        image = Image.open(image_path).convert("RGB")

        if hasattr(emotion_module, "evaluate_emotion_similarity_with_image"):
            result = _call_with_supported_kwargs(
                emotion_module.evaluate_emotion_similarity_with_image,
                image=image,
                image_path=str(image_path),
                lyrics_text=lyrics_text,
                text=lyrics_text,
                top_k_image=top_k_image,
                top_k_text=top_k_text,
                image_model_type=image_model_type,
                text_emotion_model=text_emotion_model,
                embedding_model_name=embedding_model_name,
                embedding_device=embedding_device,
                verbose=False,
            )
            return result, ""

        if hasattr(emotion_module, "evaluate_emotion_similarity"):
            result = _call_with_supported_kwargs(
                emotion_module.evaluate_emotion_similarity,
                image=image,
                image_path=str(image_path),
                lyrics_text=lyrics_text,
                text=lyrics_text,
                top_k_image=top_k_image,
                top_k_text=top_k_text,
                image_model_type=image_model_type,
                text_emotion_model=text_emotion_model,
                embedding_model_name=embedding_model_name,
                embedding_device=embedding_device,
                verbose=False,
            )
            return result, ""

        available = [
            name for name in dir(emotion_module)
            if "emotion" in name.lower() or "similarity" in name.lower()
        ]
        raise AttributeError(
            "No supported emotion similarity evaluation function found. "
            f"Available candidates: {available}"
        )

    except Exception:
        return None, traceback.format_exc()
    

def compute_lyrics_format(
    lyrics_format_module: object,
    lyrics_text: str,
    line_count: int,
    model_name: str,
    device: str | None,
    rule_weight: float,
    transformer_weight: float,
    sequence_weight: float,
) -> tuple[dict[str, Any] | None, str]:
    try:
        _, reference_lyrics, _ = build_lyrics_format_instruction(int(line_count))

        result = lyrics_format_module.score_lyrics_format_hybrid(
            json_input={"lyrics_text": lyrics_text},
            reference_lyrics=reference_lyrics,
            model_name=model_name,
            device=device,
            rule_weight=rule_weight,
            transformer_weight=transformer_weight,
            sequence_weight=sequence_weight,
            return_details=True,
        )
        return result, ""
    except Exception:
        return None, traceback.format_exc()


def compute_lyrics_quality(
    quality_module: object,
    lyrics_text: str,
) -> tuple[dict[str, Any] | None, str]:
    try:
        cleaned = strip_lyrics_section_tags(lyrics_text)

        if not cleaned.strip():
            raise ValueError("Lyrics text is empty after removing section tags.")

        result = quality_module.evaluate_cantonese_lyrics(cleaned)
        return result, ""
    except Exception:
        return None, traceback.format_exc()


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

MODEL_CSV_COLUMNS = [
    "image_name",
    "clip_alignment",
    "emotion_similarity",
    "lyrics_format_score",
    "quality_overall",
]

ALL_CSV_COLUMNS = [
    "model_key",
    "image_name",
    "clip_alignment",
    "emotion_similarity",
    "lyrics_format_score",
    "quality_overall",
]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            safe_row = {col: row.get(col, "") for col in columns}
            writer.writerow(safe_row)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch test Qwen, InternVL, and InternVL+RAG on image-to-lyrics generation and 4 metrics."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=IMAGES,
        help="Directory containing input images.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEMO_DIR / "outputs" / "batch_eval",
        help="Directory to save CSV and generated JSON outputs.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=["qwen", "intern", "intern_rag"],
        help="Model configs to test.",
    )

    parser.add_argument(
        "--line-count",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Requested lyric line count.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens for multimodal generation.",
    )

    parser.add_argument(
        "--run-on-cpu",
        action="store_true",
        help="Run multimodal generation on CPU.",
    )

    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_API_TOKEN", ""),
        help="HF token for private model access and online emotion model.",
    )

    parser.add_argument(
        "--hf-llm-model",
        default=os.environ.get("HUGGINGFACE_LLM_MODEL", "zai-org/GLM-5"),
        help="HF online chat model for llm-online text emotion predictor. Do not include provider suffix.",
    )

    parser.add_argument(
        "--text-emotion-model",
        default="llm-online",
        help="Text emotion model key or display name. Example: llm-online or zai-org/GLM-5.",
    )

    parser.add_argument(
        "--top-k-image",
        type=int,
        default=25,
        help="Top-k image emotion labels.",
    )

    parser.add_argument(
        "--top-k-text",
        type=int,
        default=25,
        help="Top-k text emotion labels.",
    )

    parser.add_argument(
        "--image-model-type",
        default="25cat",
        help="Image emotion model type.",
    )

    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Embedding model for emotion similarity.",
    )

    parser.add_argument(
        "--embedding-device",
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Embedding device for emotion similarity.",
    )

    parser.add_argument(
        "--lyrics-format-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformer model for lyrics format evaluator.",
    )

    parser.add_argument(
        "--lyrics-format-device",
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Device for lyrics format evaluator.",
    )

    parser.add_argument(
        "--rag-csv-path",
        default=str(PROJECT_ROOT / "cantopop_corpus_final_583_yue.csv"),
        help="CSV corpus path for InternVL + RAG.",
    )

    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=3,
        help="Top-k RAG examples for InternVL + RAG.",
    )

    parser.add_argument(
        "--style-prompt",
        default=DEFAULT_STYLE_PROMPT,
        help="Fixed genre_prompt style used for fair comparison.",
    )

    parser.add_argument(
        "--mood-text",
        default=DEFAULT_MOOD_TEXT,
        help="Mood text override used for generation prompt.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip generation if the per-image JSON already exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    apply_hf_env(args.hf_token, args.hf_llm_model)

    images = list_images(args.input_dir)

    if not images:
        raise ValueError(f"No supported images found in: {args.input_dir}")

    print(f"Input images: {len(images)}")
    print(f"Models: {args.models}")
    print(f"Output dir: {args.output_dir}")

    # Load evaluators once.
    print("Loading evaluation modules...")
    similarity_module = load_image_text_similarity_module()
    emotion_module = load_image_lyrics_emotion_similarity_module()
    lyrics_format_module = load_lyrics_format_module()
    quality_module = load_lyrics_quality_module()

    text_emotion_model_key = resolve_text_emotion_model_key(
        emotion_module,
        args.text_emotion_model,
    )

    print(f"Text emotion model: {text_emotion_model_key}")
    print(f"HUGGINGFACE_LLM_MODEL: {os.environ.get('HUGGINGFACE_LLM_MODEL', '')}")

    all_rows: list[dict[str, Any]] = []

    for model_key in args.models:
        config = MODEL_CONFIGS[model_key]
        model_rows: list[dict[str, Any]] = []

        model_output_dir = args.output_dir / model_key
        json_output_dir = model_output_dir / "json"
        json_output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Testing model config: {model_key}")
        print(f"model_id: {config['model_id']}")
        print(f"use_rag: {config['use_rag']}")
        print("=" * 80)

        for idx, image_path in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] {model_key} | {image_path.name}")

            row: dict[str, Any] = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "model_key": model_key,
                "model_id": config["model_id"],
                "use_rag": config["use_rag"],
                "status": "started",
            }

            image_bytes = image_path.read_bytes()
            output_json_path = json_output_dir / f"{image_path.stem}.json"
            row["json_output_path"] = str(output_json_path)

            # -----------------------------
            # 1. Generate lyrics
            # -----------------------------
            try:
                t0 = time.time()

                if args.resume and output_json_path.exists():
                    payload = json.loads(output_json_path.read_text(encoding="utf-8"))
                    title = payload.get("title", "")
                    lyrics_text = payload.get("lyrics_text", "")
                    genre_prompt = payload.get("genre_prompt", "")
                    print("  Resumed from existing JSON.")
                else:
                    bundle = generate_from_image(
                        image_bytes=image_bytes,
                        model_id=config["model_id"],
                        style=args.style_prompt,
                        line_count=args.line_count,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        run_on_cpu=args.run_on_cpu,
                        hf_token=args.hf_token or None,
                        use_rag=bool(config["use_rag"]),
                        rag_csv_path=args.rag_csv_path,
                        rag_top_k=int(args.rag_top_k),
                        genre_prompt_mode="preset",
                        mood_text_override=args.mood_text,
                    )

                    payload = bundle_to_dict(bundle)

                    title = payload.get("title", "")
                    lyrics_text = payload.get("lyrics_text", "")
                    genre_prompt = payload.get("genre_prompt", "")

                    output_json_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                row["generation_time_s"] = round(time.time() - t0, 3)
                row["title"] = title
                row["lyrics_text"] = lyrics_text
                row["genre_prompt"] = genre_prompt
                row["status"] = "generated"

            except Exception as exc:
                row["status"] = "generation_failed"
                row["generation_error"] = traceback.format_exc()
                model_rows.append(row)
                all_rows.append(row)

                print(f"  Generation failed: {exc}")

                try:
                    unload_mm_models()
                except Exception:
                    pass

                continue

            finally:
                try:
                    unload_mm_models()
                except Exception:
                    pass

            # -----------------------------
            # 2. Compute metrics
            # -----------------------------
            metric_t0 = time.time()

            # Metric 1: CLIP image-lyrics alignment
            clip_score, clip_err = compute_clip_alignment(
                similarity_module,
                image_bytes,
                lyrics_text,
            )
            row["clip_alignment"] = safe_float(clip_score)
            row["clip_error"] = clip_err

            # Metric 2: emotion similarity
            emotion_result, emotion_err = compute_emotion_similarity(
                emotion_module=emotion_module,
                image_path=image_path,
                lyrics_text=lyrics_text,
                text_emotion_model=text_emotion_model_key,
                top_k_image=args.top_k_image,
                top_k_text=args.top_k_text,
                image_model_type=args.image_model_type,
                embedding_model_name=args.embedding_model_name,
                embedding_device=args.embedding_device,
            )
            row["emotion_error"] = emotion_err
            
            if emotion_err:
                print(f"[emotion error] {image_path.name} | {model_key}")
                print(emotion_err)
            else:
                print(f"[emotion result] {image_path.name} | {model_key}: {emotion_result}")

            if emotion_result:
                row["emotion_similarity"] = safe_float(
                    extract_emotion_similarity_value(emotion_result)
                )

            # Metric 3: lyrics format
            format_result, format_err = compute_lyrics_format(
                lyrics_format_module=lyrics_format_module,
                lyrics_text=lyrics_text,
                line_count=args.line_count,
                model_name=args.lyrics_format_model,
                device=args.lyrics_format_device,
                rule_weight=0.50,
                transformer_weight=0.30,
                sequence_weight=0.20,
            )
            row["lyrics_format_error"] = format_err

            if format_result:
                row["lyrics_format_score"] = safe_float(
                    format_result.get("lyrics_format_score")
                )
                row["lyrics_format_grade"] = format_result.get("grade", "")

                metrics = format_result.get("metrics", {}) or {}
                row["lyrics_format_rule_score"] = safe_float(
                    metrics.get("rule_format_score")
                )
                row["lyrics_format_transformer_score"] = safe_float(
                    metrics.get("transformer_format_similarity_score")
                )
                row["lyrics_format_sequence_score"] = safe_float(
                    metrics.get("sequence_structure_score")
                )

                warnings_list = format_result.get("warnings", [])
                row["lyrics_format_warnings"] = json.dumps(
                    warnings_list,
                    ensure_ascii=False,
                )

            # Metric 4: Cantonese lyrics quality
            quality_result, quality_err = compute_lyrics_quality(
                quality_module,
                lyrics_text,
            )
            row["quality_error"] = quality_err

            if quality_result:
                row["quality_overall"] = safe_float(
                    quality_result.get("overall")
                )
                row["quality_grade"] = quality_result.get("grade", "")

                scores = quality_result.get("scores", {}) or {}

                row["quality_tonal"] = safe_float(scores.get("tonal"))
                row["quality_rhyme"] = safe_float(scores.get("rhyme"))
                row["quality_lexical"] = safe_float(scores.get("lexical"))
                row["quality_structure"] = safe_float(scores.get("structure"))
                row["quality_coherence"] = safe_float(scores.get("coherence"))
                row["quality_natural"] = safe_float(scores.get("natural"))

                row["quality_suggestions"] = json.dumps(
                    quality_result.get("suggestions", []),
                    ensure_ascii=False,
                )

            row["metric_time_s"] = round(time.time() - metric_t0, 3)

            if any(
                row.get(k)
                for k in [
                    "clip_error",
                    "emotion_error",
                    "lyrics_format_error",
                    "quality_error",
                ]
            ):
                row["status"] = "metric_partial_error"
            else:
                row["status"] = "ok"

            model_rows.append(row)
            all_rows.append(row)

            print(
                f"  status={row['status']} | "
                f"clip={row.get('clip_alignment', '')} | "
                f"emotion={row.get('emotion_similarity', '')} | "
                f"format={row.get('lyrics_format_score', '')} | "
                f"quality={row.get('quality_overall', '')}"
            )

        model_csv = args.output_dir / f"results_{model_key}.csv"
        write_csv(model_csv, model_rows, MODEL_CSV_COLUMNS)
        print(f"Saved model CSV: {model_csv}")

    all_csv = args.output_dir / "results_all_models.csv"
    write_csv(all_csv, all_rows, ALL_CSV_COLUMNS)
    print(f"\nSaved combined CSV: {all_csv}")


if __name__ == "__main__":
    main()