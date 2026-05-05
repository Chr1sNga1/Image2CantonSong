#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize all model results on one radar chart.

Input CSV format (expected):
model_key,image_name,clip_alignment,emotion_similarity,lyrics_format_score,quality_overall

Example:
python visualize_model_radar.py \
  --input outputs/batch_eval/results_all_models.csv \
  --output outputs/batch_eval/model_radar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot all models on one radar chart."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to results_all_models.csv",
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


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric
    metric_cols = [
        "clip_alignment",
        "emotion_similarity",
        "lyrics_format_score",
        "quality_overall",
    ]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize lyrics_format_score to 0-1
    if "lyrics_format_score" in df.columns:
        if df["lyrics_format_score"].dropna().max() > 1.5:
            df["lyrics_format_score"] = df["lyrics_format_score"] / 100.0

    return df


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "clip_alignment",
        "emotion_similarity",
        "lyrics_format_score",
        "quality_overall",
    ]

    summary = (
        df.groupby("model_key", as_index=False)[metric_cols]
        .mean()
        .sort_values("model_key")
        .reset_index(drop=True)
    )
    return summary


def plot_radar(summary: pd.DataFrame, output_path: Path):
    labels = [
        "CLIP Alignment",
        "Emotion Similarity",
        "Lyrics Format",
        "Lyrics Quality",
    ]
    metric_cols = [
        "clip_alignment",
        "emotion_similarity",
        "lyrics_format_score",
        "quality_overall",
    ]

    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Rotate so the first axis starts at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    # Radial axis
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)

    # Use matplotlib default color cycle automatically
    for _, row in summary.iterrows():
        values = [row[col] for col in metric_cols]
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

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    required_cols = {
        "model_key",
        "clip_alignment",
        "emotion_similarity",
        "lyrics_format_score",
        "quality_overall",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = normalize_metrics(df)
    summary = aggregate_by_model(df)

    print("Aggregated mean scores by model:")
    print(summary)

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
        print(f"Saved summary CSV to: {args.summary_output}")

    plot_radar(summary, args.output)
    print(f"Saved radar chart to: {args.output}")


if __name__ == "__main__":
    main()