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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FigureEntry:
    label: str
    family: str
    json_path: str
    highlight: bool = False


DEFAULT_ENTRIES: List[FigureEntry] = [
    FigureEntry("RGB", "Single modality", "logs/eval/eval_rgb_test_12899169.pbs101_20260216_180631.json"),
    FigureEntry("T-only", "Single modality", "logs/eval/eval_t_test_12844394.pbs101_20260207_163134.json"),
    FigureEntry("RGBT Base", "Simple fusion", "logs/eval/eval_rgbt_base_test_13104430.pbs101_20260310_165317.json"),
    FigureEntry("RGBT Late", "Simple fusion", "logs/eval/eval_rgbt_late_test_12844746.pbs101_20260207_205927.json"),
    FigureEntry("RGBT Early", "Simple fusion", "logs/eval/eval_rgbt_early_test_12844745.pbs101_20260207_205921.json"),
    FigureEntry("RGBT Adaptive Late", "Adaptive fusion", "logs/eval/eval_rgbt_adaptive_late_test_12844390.pbs101_20260207_163108.json"),
    FigureEntry("Adaptive FPN (Heavy)", "Adaptive fusion", "logs/eval/eval_rgbt_adaptive_fpn_test_13091850.pbs101_20260309_111949.json"),
    FigureEntry("Adaptive FPN (Lite)", "Adaptive fusion", "logs/eval/eval_rgbt_adaptive_fpn_lite_test_13232762.pbs101_20260318_221051.json"),
    FigureEntry("Adaptive FPN (Lite + Calibrated)", "Proposed", "logs/eval/eval_rgbt_adaptive_fpn_lite_cal_test_13244557.pbs101_20260319_120238.json", highlight=True),
]


FAMILY_STYLE: Dict[str, Dict[str, object]] = {
    "Single modality": {"color": "#6b7280", "marker": "o"},
    "Simple fusion": {"color": "#2563eb", "marker": "o"},
    "Adaptive fusion": {"color": "#d97706", "marker": "o"},
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
    cluster_labels = {"RGBT Base", "Adaptive FPN (Lite)", "Adaptive FPN (Lite + Calibrated)"}
    offsets = {
        "RGB": (6, 6),
        "T-only": (6, -14),
        "RGBT Late": (6, 6),
        "RGBT Early": (6, -14),
        "RGBT Adaptive Late": (-108, -18),
        "Adaptive FPN (Heavy)": (8, -16),
    }
    for row in rows:
        if str(row["label"]) in cluster_labels:
            continue
        dx, dy = offsets.get(str(row["label"]), (6, 6))
        ax.annotate(
            str(row["label"]),
            (float(row["fps"]), float(row["mae"])),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )


def annotate_cluster_inset(ax, rows: List[Dict[str, object]]):
    offsets = {
        "RGBT Base": (-54, -8),
        "Adaptive FPN (Lite)": (8, 10),
        "Adaptive FPN (Lite + Calibrated)": (10, -18),
    }
    for row in rows:
        dx, dy = offsets[str(row["label"])]
        ax.annotate(
            str(row["label"]),
            (float(row["fps"]), float(row["mae"])),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
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
        sizes = []
        for r in family_rows:
            if bool(r["highlight"]):
                sizes.append(240.0)
            else:
                sizes.append(150.0)
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=style["color"],
            marker=style["marker"],
            alpha=0.95,
            edgecolors="black",
            linewidths=0.6,
            label=family,
            zorder=3 if family == "Proposed" else 2,
        )

    annotate_points(ax, rows)

    cluster_labels = {"RGBT Base", "Adaptive FPN (Lite)", "Adaptive FPN (Lite + Calibrated)"}
    cluster_rows = [r for r in rows if str(r["label"]) in cluster_labels]
    axins = inset_axes(ax, width="35%", height="32%", loc="center right", borderpad=1.5)
    for family, style in FAMILY_STYLE.items():
        family_rows = [r for r in cluster_rows if r["family"] == family]
        if not family_rows:
            continue
        xs = [float(r["fps"]) for r in family_rows]
        ys = [float(r["mae"]) for r in family_rows]
        sizes = [240.0 if bool(r["highlight"]) else 150.0 for r in family_rows]
        axins.scatter(
            xs,
            ys,
            s=sizes,
            c=style["color"],
            marker=style["marker"],
            alpha=0.95,
            edgecolors="black",
            linewidths=0.6,
            zorder=3 if family == "Proposed" else 2,
        )
    annotate_cluster_inset(axins, cluster_rows)
    axins.set_xlim(62.0, 69.2)
    axins.set_ylim(22.09, 22.19)
    axins.set_xticks([62, 64, 66, 68])
    axins.set_yticks([22.10, 22.14, 22.18])
    axins.tick_params(labelsize=7)
    axins.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    axins.set_title("Zoomed view", fontsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#6b7280", lw=0.7)

    ax.set_title(args.title, fontsize=12)
    ax.set_xlabel("Evaluation throughput (FPS)")
    ax.set_ylabel("MAE")
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
                markersize=7 if family != "Proposed" else 11,
                linewidth=0,
            )
        )
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)

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
