#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FigureEntry:
    label: str
    family: str
    json_path: str
    highlight: bool = False


DEFAULT_ENTRIES: List[FigureEntry] = [
    FigureEntry("rgb", "Single modality", "logs/eval/eval_rgb_test_12899169.pbs101_20260216_180631.json"),
    FigureEntry("t", "Single modality", "logs/eval/eval_t_test_12844394.pbs101_20260207_163134.json"),
    FigureEntry("rgbt_base", "Simple fusion", "logs/eval/eval_rgbt_base_test_13104430.pbs101_20260310_165317.json"),
    FigureEntry("rgbt_late", "Simple fusion", "logs/eval/eval_rgbt_late_test_12844746.pbs101_20260207_205927.json"),
    FigureEntry("rgbt_early", "Simple fusion", "logs/eval/eval_rgbt_early_test_12844745.pbs101_20260207_205921.json"),
    FigureEntry("rgbt_adaptive_late", "Adaptive fusion", "logs/eval/eval_rgbt_adaptive_late_test_12844390.pbs101_20260207_163108.json"),
    FigureEntry("Adaptive FPN (heavy)", "Adaptive fusion", "logs/eval/eval_rgbt_adaptive_fpn_test_13091850.pbs101_20260309_111949.json"),
    FigureEntry("adaptive_fpn_lite", "Adaptive fusion", "logs/eval/eval_rgbt_adaptive_fpn_lite_test_13232762.pbs101_20260318_221051.json"),
    FigureEntry("Adaptive FPN (calibrated)", "Proposed", "logs/eval/eval_rgbt_adaptive_fpn_lite_cal_test_13244557.pbs101_20260319_120238.json", highlight=True),
]


FAMILY_STYLE: Dict[str, Dict[str, object]] = {
    "Single modality": {"color": "#6b7280", "marker": "o"},
    "Simple fusion": {"color": "#2563eb", "marker": "o"},
    "Adaptive fusion": {"color": "#d97706", "marker": "^"},
    "Proposed": {"color": "#059669", "marker": "*"},
}


def parse_args():
    ap = argparse.ArgumentParser(description="Generate Figure 5.1 accuracy-efficiency scatter plot.")
    ap.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "outputs" / "report_figures" / "figure_5_1"),
        help="Directory to write PNG/PDF/CSV outputs.",
    )
    ap.add_argument(
        "--title",
        default="Figure 5.1: Accuracy versus efficiency across in-house model variants",
        help="Plot title.",
    )
    return ap.parse_args()


def load_metrics(entry: FigureEntry) -> Dict[str, object]:
    path = PROJECT_ROOT / entry.json_path
    with path.open("r") as f:
        obj = json.load(f)
    return {
        "label": entry.label,
        "family": entry.family,
        "highlight": entry.highlight,
        "json_path": str(path),
        "params_m": float(obj["params_total"]) / 1_000_000.0,
        "fps": float(obj["eval_fps"]),
        "mae": float(obj["mae"]),
        "rmse": float(obj["rmse"]),
    }


def annotate_points(ax, rows: List[Dict[str, object]]):
    offsets = {
        "rgb": (6, 6),
        "t": (6, -14),
        "rgbt_base": (6, -12),
        "rgbt_late": (6, 6),
        "rgbt_early": (6, -14),
        "rgbt_adaptive_late": (-78, -4),
        "Adaptive FPN (heavy)": (-10, -14),
        "adaptive_fpn_lite": (8, 8),
        "Adaptive FPN (calibrated)": (8, -16),
    }
    for row in rows:
        dx, dy = offsets.get(str(row["label"]), (6, 6))
        ax.annotate(
            str(row["label"]),
            (float(row["fps"]), float(row["mae"])),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.5,
        )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_metrics(entry) for entry in DEFAULT_ENTRIES]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=220)

    for family, style in FAMILY_STYLE.items():
        family_rows = [r for r in rows if r["family"] == family]
        if not family_rows:
            continue
        xs = [float(r["fps"]) for r in family_rows]
        ys = [float(r["mae"]) for r in family_rows]
        sizes = [26.0 + float(r["params_m"]) * 7.5 for r in family_rows]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=style["color"],
            marker=style["marker"],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.6,
            label=family,
            zorder=3 if family == "Proposed" else 2,
        )

    annotate_points(ax, rows)

    ax.set_title(args.title, fontsize=12)
    ax.set_xlabel("Evaluation throughput (FPS)")
    ax.set_ylabel("MAE (lower is better)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    max_fps = max(float(r["fps"]) for r in rows)
    min_mae = min(float(r["mae"]) for r in rows)
    max_mae = max(float(r["mae"]) for r in rows)
    ax.set_xlim(0, math.ceil((max_fps + 4.0) / 5.0) * 5.0)
    ax.set_ylim(math.floor(min_mae - 1.0), math.ceil(max_mae + 2.0))
    ax.set_yticks([20, 25, 30, 35, 40, 45])

    ymin = min(float(r["mae"]) for r in rows)
    ymax = max(float(r["mae"]) for r in rows)
    ax.set_ylim(math.floor(ymin - 1.0), math.ceil(ymax + 2.0))

    legend_handles = []
    for family, style in FAMILY_STYLE.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                label=family,
                markerfacecolor=style["color"],
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=7 if family != "Proposed" else 10,
                linewidth=0,
            )
        )
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)

    fig.text(
        0.99,
        0.02,
        "Point size increases slightly with parameter count (millions).",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#374151",
    )

    png_path = out_dir / "figure_5_1_scatter.png"
    pdf_path = out_dir / "figure_5_1_scatter.pdf"
    csv_path = out_dir / "figure_5_1_scatter.csv"
    json_path = out_dir / "figure_5_1_scatter.json"

    fig.tight_layout(rect=(0.02, 0.04, 1.0, 1.0))
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "family", "params_m", "fps", "mae", "rmse", "json_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with json_path.open("w") as f:
        json.dump({"entries": rows}, f, indent=2)

    print(f"[png] {png_path}")
    print(f"[pdf] {pdf_path}")
    print(f"[csv] {csv_path}")
    print(f"[json] {json_path}")


if __name__ == "__main__":
    main()
