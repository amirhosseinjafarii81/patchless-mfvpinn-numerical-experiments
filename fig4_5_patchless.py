#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig4_5_patchless.py

Plot snapshot panels from:
- snapshots_patchless_cm4.npz
- snapshots_patchless_cm9.npz

Supports both CM=4 and CM=9 files based on the actual snapshot schema:
- P{k}_centers
- P{k}_h
- P{k}_eta

Dot position: patch centers
Dot size: proportional to h^2
Dot color:
  - prefers P{k}_eta_gamma if present
  - else P{k}_gamma_eta if present
  - else uses P{k}_eta * (1/h^2)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# -------------------------
# Colormap (blue -> purple -> red)
# -------------------------
def make_bpr_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.05, 0.10, 0.90),  # blue
        (0.55, 0.00, 0.55),  # purple
        (0.90, 0.05, 0.05),  # red
    ]
    return LinearSegmentedColormap.from_list("bpr_bold", colors, N=256)


BPR_CMAP = make_bpr_cmap()
P_RE = re.compile(r"^P(\d+)_centers$")


def available_k(npz: np.lib.npyio.NpzFile) -> List[int]:
    ks = []
    for key in npz.files:
        m = P_RE.match(key)
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def count_at(npz: np.lib.npyio.NpzFile, k: int) -> int:
    return int(npz[f"P{k}_centers"].shape[0])


def pick_k_by_target_count(npz: np.lib.npyio.NpzFile, target_n: int) -> Tuple[int, int, bool]:
    """
    Find snapshot index k such that count_at(npz,k) matches target_n.
    If exact match exists: returns (k, actual_n, True)
    Otherwise: choose nearest-by-absolute-diff count, returns (k, actual_n, False)
    """
    ks = available_k(npz)
    if not ks:
        raise RuntimeError("No P{k}_centers found in NPZ.")

    counts = np.array([count_at(npz, k) for k in ks], dtype=int)
    diffs = np.abs(counts - int(target_n))
    j = int(np.argmin(diffs))
    k = ks[j]
    actual = int(counts[j])
    exact = (actual == int(target_n))
    return k, actual, exact


def parse_csv_ints_or_auto(s: str) -> List[int] | None:
    s = (s or "").strip().lower()
    if s in ("auto", "none", ""):
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def auto_pick_counts(npz: np.lib.npyio.NpzFile, n_panels: int = 6) -> List[int]:
    """
    Picks n_panels representative Npatch values from available snapshots,
    evenly spaced in index space.
    """
    ks = available_k(npz)
    if not ks:
        raise RuntimeError("No snapshots found in npz.")

    if len(ks) <= n_panels:
        out = [count_at(npz, k) for k in ks]
        while len(out) < n_panels:
            out.append(out[-1])
        return out[:n_panels]

    idx = np.linspace(0, len(ks) - 1, n_panels).round().astype(int)
    return [count_at(npz, ks[i]) for i in idx]


def get_centers_h(npz: np.lib.npyio.NpzFile, k: int):
    centers = np.asarray(npz[f"P{k}_centers"], dtype=float)
    if f"P{k}_h" in npz.files:
        h = np.asarray(npz[f"P{k}_h"], dtype=float).reshape(-1)
    else:
        h = np.ones((centers.shape[0],), dtype=float)
    return centers, h


def get_eta_gamma(npz: np.lib.npyio.NpzFile, k: int, h: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Prefer P{k}_eta_gamma if present; else P{k}_gamma_eta; else compute eta*(1/h^2).
    """
    key = f"P{k}_eta_gamma"
    if key in npz.files:
        return np.asarray(npz[key], dtype=float).reshape(-1), key

    alt = f"P{k}_gamma_eta"
    if alt in npz.files:
        return np.asarray(npz[alt], dtype=float).reshape(-1), alt

    eta_key = f"P{k}_eta"
    if eta_key in npz.files:
        eta = np.asarray(npz[eta_key], dtype=float).reshape(-1)
        gamma = 1.0 / np.maximum(h * h, 1e-300)
        return eta * gamma, f"{eta_key} * (1/h^2)"

    raise KeyError(f"No eta data found for P{k}.")


def robust_lognorm_from_list(v_list: List[np.ndarray], floor: float = 1e-12) -> LogNorm:
    """
    One LogNorm shared across all panels in a figure.
    Uses 1% and 99% quantiles of concatenated values.
    """
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


def add_letter(ax: plt.Axes, letter: str):
    ax.text(0.5, -0.22, f"({letter})", transform=ax.transAxes,
            ha="center", va="top", fontsize=18)


def plot_six_panels_by_counts(
    npz_path: str,
    out_png: str,
    title: str,
    target_counts: List[int] | None,
    size_scale: float,
    size_clip: Tuple[float, float],
):
    npz = np.load(npz_path, allow_pickle=True)

    if target_counts is None:
        target_counts = auto_pick_counts(npz, n_panels=6)

    panel_data = []
    for target_n in target_counts:
        k, actual_n, exact = pick_k_by_target_count(npz, int(target_n))
        centers, h = get_centers_h(npz, k)
        eta_gamma, eta_key = get_eta_gamma(npz, k, h=h)
        panel_data.append((k, actual_n, exact, centers, h, eta_gamma, eta_key))

    norm = robust_lognorm_from_list([d[5] for d in panel_data])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    letters = "abcdef"

    for i, (k, actual_n, exact, centers, h, eta_gamma, eta_key) in enumerate(panel_data):
        ax = axes[i]
        x, y = centers[:, 0], centers[:, 1]

        sizes = float(size_scale) * (h * h)
        smin, smax = size_clip
        if smin > 0:
            sizes = np.maximum(sizes, smin)
        if smax > 0:
            sizes = np.minimum(sizes, smax)

        sc = ax.scatter(
            x, y,
            c=np.maximum(eta_gamma, 1e-12),
            s=sizes,
            cmap=BPR_CMAP,
            norm=norm,
            linewidths=0.0,
            alpha=1.0,
            rasterized=True,
        )

        ticks = np.linspace(0.0, 1.0, 6)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        fmt = FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.tick_params(labelsize=11)

        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=10)

        add_letter(ax, letters[i])
        ax.set_title(f"N = {actual_n}", fontsize=10)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", type=str, required=True)

    # defaults changed to your actual uploaded filenames
    ap.add_argument("--cm4-npz", type=str, default="snapshots_patchless_cm4.npz")
    ap.add_argument("--cm9-npz", type=str, default="snapshots_patchless_cm9.npz")

    ap.add_argument("--size-scale", type=float, default=2500.0)
    ap.add_argument("--size-min", type=float, default=2.0)
    ap.add_argument("--size-max", type=float, default=200.0)

    ap.add_argument("--cm4-counts", type=str, default="auto",
                    help='6 target Npatch values for CM=4 panels (comma-separated) or "auto".')
    ap.add_argument("--cm9-counts", type=str, default="auto",
                    help='6 target Npatch values for CM=9 panels (comma-separated) or "auto".')

    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    cm4 = args.cm4_npz
    cm9 = args.cm9_npz

    if not os.path.exists(cm4):
        raise FileNotFoundError(f"CM=4 snapshot file not found: {cm4}")
    if not os.path.exists(cm9):
        raise FileNotFoundError(f"CM=9 snapshot file not found: {cm9}")

    cm4_counts = parse_csv_ints_or_auto(args.cm4_counts)
    cm9_counts = parse_csv_ints_or_auto(args.cm9_counts)

    plot_six_panels_by_counts(
        npz_path=cm4,
        out_png=os.path.join(outdir, "figure_cm4.png"),
        title=r"Anchor snapshots ($C_M = 4$)",
        target_counts=cm4_counts,
        size_scale=float(args.size_scale),
        size_clip=(float(args.size_min), float(args.size_max)),
    )

    plot_six_panels_by_counts(
        npz_path=cm9,
        out_png=os.path.join(outdir, "figure_cm9.png"),
        title=r"Anchor snapshots ($C_M = 9$)",
        target_counts=cm9_counts,
        size_scale=float(args.size_scale),
        size_clip=(float(args.size_min), float(args.size_max)),
    )

    print("Saved:", os.path.join(outdir, "figure_cm4.png"))
    print("Saved:", os.path.join(outdir, "figure_cm9.png"))


if __name__ == "__main__":
    main()
