#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
patchless_fig18_problem35.py

Plot the final anchor sets for the patchless Problem (35) runs:
(a) Strategy #1, C_M=4
(b) Strategy #2, C_M=4
(c) Strategy #1, C_M=9
(d) Strategy #2, C_M=9

Expected NPZ keys (from patchless_fig17_problem35_strat12.py):
  holes_xyxy            : (4,4) array [xmin,ymin,xmax,ymax]
  S1_CM4_centers, S1_CM4_h2, S1_CM4_r2
  S2_CM4_centers, S2_CM4_h2, S2_CM4_r2
  S1_CM9_centers, S1_CM9_h2, S1_CM9_r2
  S2_CM9_centers, S2_CM9_h2, S2_CM9_r2

Visual style is intentionally kept aligned with the original Figure 18 plotter.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Colormap: blue -> purple -> red (same visual family as the original scripts)
# -----------------------------------------------------------------------------
def make_bpr_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.05, 0.10, 0.90),
        (0.55, 0.00, 0.55),
        (0.90, 0.05, 0.05),
    ]
    return LinearSegmentedColormap.from_list("bpr_bold", colors, N=256)


BPR_CMAP = make_bpr_cmap()


def robust_lognorm_from_list(v_list: List[np.ndarray], floor: float = 1e-12) -> LogNorm:
    if not v_list:
        return LogNorm(vmin=1e-6, vmax=1.0)

    v = np.concatenate([np.asarray(x, float).ravel() for x in v_list], axis=0)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return LogNorm(vmin=1e-6, vmax=1.0)

    v = np.maximum(v, floor)
    lo = max(float(np.quantile(v, 0.01)), floor)
    hi = float(np.quantile(v, 0.99))
    if hi <= lo:
        hi = lo * 10.0
    return LogNorm(vmin=lo, vmax=hi)


def add_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.5,
        -0.22,
        f"({letter})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
    )


def load_panel(npz: np.lib.npyio.NpzFile, key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(npz[f"{key}_centers"], dtype=float)

    if f"{key}_h2" in npz.files:
        h2 = np.asarray(npz[f"{key}_h2"], dtype=float).reshape(-1)
    elif f"{key}_h" in npz.files:
        h = np.asarray(npz[f"{key}_h"], dtype=float).reshape(-1)
        h2 = h * h
    else:
        raise KeyError(f"NPZ missing {key}_h2 (or fallback {key}_h).")

    if f"{key}_r2" in npz.files:
        r2 = np.asarray(npz[f"{key}_r2"], dtype=float).reshape(-1)
    elif f"{key}_eta" in npz.files:
        r2 = np.asarray(npz[f"{key}_eta"], dtype=float).reshape(-1)
    else:
        raise KeyError(f"NPZ missing {key}_r2 (or fallback {key}_eta).")

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"{key}_centers must have shape (N,2). Got {centers.shape}")
    if not (h2.shape[0] == centers.shape[0] == r2.shape[0]):
        raise ValueError(
            f"Size mismatch for {key}: centers={centers.shape}, h2={h2.shape}, r2={r2.shape}"
        )
    return centers, h2, r2


def draw_holes(ax: plt.Axes, holes_xyxy: np.ndarray, lw: float = 2.0) -> None:
    for (xmin, ymin, xmax, ymax) in holes_xyxy:
        rect = Rectangle(
            (float(xmin), float(ymin)),
            float(xmax - xmin),
            float(ymax - ymin),
            fill=False,
            edgecolor="black",
            linewidth=lw,
        )
        ax.add_patch(rect)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--npz", type=str, required=True,
                    help="figure18_problem35_patchless.npz")
    ap.add_argument("--size-scale", type=float, default=2500.0)
    ap.add_argument("--size-min", type=float, default=2.0)
    ap.add_argument("--size-max", type=float, default=200.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    npz = np.load(args.npz, allow_pickle=True)

    if "holes_xyxy" not in npz.files:
        raise KeyError("NPZ missing key: holes_xyxy")

    holes_xyxy = np.asarray(npz["holes_xyxy"], dtype=float)
    if holes_xyxy.shape != (4, 4):
        raise ValueError(f"holes_xyxy must have shape (4,4). Got {holes_xyxy.shape}")

    panel_keys = ["S1_CM4", "S2_CM4", "S1_CM9", "S2_CM9"]
    panel_titles = [
        r"Strategy \#1 ($C_M=4$)",
        r"Strategy \#2 ($C_M=4$)",
        r"Strategy \#1 ($C_M=9$)",
        r"Strategy \#2 ($C_M=9$)",
    ]
    letters = ["a", "b", "c", "d"]

    panels = {}
    all_r2 = []
    for key in panel_keys:
        centers, h2, r2 = load_panel(npz, key)
        panels[key] = (centers, h2, r2)
        all_r2.append(r2)

    norm = robust_lognorm_from_list(all_r2, floor=1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.ravel()

    for i, key in enumerate(panel_keys):
        ax = axes[i]
        centers, h2, r2 = panels[key]

        x = centers[:, 0]
        y = centers[:, 1]
        sizes = float(args.size_scale) * np.asarray(h2, float)
        if float(args.size_min) > 0:
            sizes = np.maximum(sizes, float(args.size_min))
        if float(args.size_max) > 0:
            sizes = np.minimum(sizes, float(args.size_max))

        sc = ax.scatter(
            x,
            y,
            c=np.maximum(r2, 1e-12),
            s=sizes,
            cmap=BPR_CMAP,
            norm=norm,
            linewidths=0.0,
            alpha=1.0,
            rasterized=True,
        )

        draw_holes(ax, holes_xyxy, lw=2.0)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.tick_params(labelsize=11)

        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=10)

        ax.set_title(panel_titles[i], fontsize=12)
        add_letter(ax, letters[i])

    fig.suptitle(r"Problem (35): Last set of anchors and $r_a^2$", fontsize=18)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    out_png = os.path.join(args.outdir, "figure18_problem35_patchless.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)


if __name__ == "__main__":
    main()
