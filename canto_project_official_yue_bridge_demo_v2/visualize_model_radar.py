#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize selected model result CSVs on one radar chart.

Expected per-model CSV format:
image_name,clip_alignment,emotion_similarity,lyrics_format_score,quality_overall

Example:
python visualize_model_radar.py \
  --input-dir outputs/batch_eval \
  --models intern_rag intern qwen \
  --output outputs/batch_eval/model_radar.png \
  --summary-output outputs/batch_eval_50/model_metric_summary.csv (Optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRIC_COLS = [
    "clip_alignment",
    "emotion_similarity",
    "lyrics_format_score",
    "quality_overall",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected model result CSVs on one radar chart."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing results_intern.csv, results_intern_rag.csv, etc.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["intern_rag", "intern"],
        help="Model keys to visualize. Example: intern_rag intern qwen",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output radar chart path, e.g. model_radar.png",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to save aggregated mean scores as CSV",
    )

    return parser.parse_args()


def load_model_csvs(input_dir: Path, models: list[str]) -> pd.DataFrame:
    all_dfs = []

    for model_key in models:
        csv_path = input_dir / f"results_{model_key}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Model CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = {"image_name", *METRIC_COLS}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_path} missing required columns: {sorted(missing)}"
            )

        df = df.copy()
        df["model_key"] = model_key
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize lyrics_format_score to 0-1 if it is stored as 0-100
    if df["lyrics_format_score"].dropna().max() > 1.5:
        df["lyrics_format_score"] = df["lyrics_format_score"] / 100.0

    return df


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    mean_summary = (
        df.groupby("model_key", as_index=False)[METRIC_COLS]
        .mean()
        .sort_values("model_key")
        .reset_index(drop=True)
    )

    valid_counts = (
        df.groupby("model_key")[METRIC_COLS]
        .count()
        .add_suffix("_valid_count")
        .reset_index()
    )

    total_counts = (
        df.groupby("model_key")
        .size()
        .reset_index(name="total_records")
    )

    summary = (
        mean_summary
        .merge(valid_counts, on="model_key", how="left")
        .merge(total_counts, on="model_key", how="left")
    )

    return summary


def plot_radar(summary: pd.DataFrame, output_path: Path):
    labels = [
        "CLIP Alignment",
        "Emotion Similarity",
        "Lyrics Format",
        "Lyrics Quality",
    ]

    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)

    for _, row in summary.iterrows():
        values = [row[col] for col in METRIC_COLS]

        # If one metric is all missing for a model, fill with 0 only for plotting.
        values = [0 if pd.isna(v) else v for v in values]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=row["model_key"])
        ax.fill(angles, values, alpha=0.20)

    ax.set_title("Model Comparison Radar Chart", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    df = load_model_csvs(args.input_dir, args.models)
    df = normalize_metrics(df)
    summary = aggregate_by_model(df)

    print("Aggregated mean scores by model:")
    print(summary)

    print("\nValid metric counts:")
    count_cols = [
        "model_key",
        "total_records",
        "clip_alignment_valid_count",
        "emotion_similarity_valid_count",
        "lyrics_format_score_valid_count",
        "quality_overall_valid_count",
    ]
    print(summary[count_cols])

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
        print(f"Saved summary CSV to: {args.summary_output}")

    plot_radar(summary, args.output)
    print(f"Saved radar chart to: {args.output}")


if __name__ == "__main__":
    main()