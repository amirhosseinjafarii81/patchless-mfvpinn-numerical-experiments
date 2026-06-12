#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap


# -------------------------
# Colormap (blue -> purple -> red)
# Kept visually aligned with the original fig10/11 and fig14/15 scripts.
# -------------------------
def make_bpr_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.05, 0.10, 0.90),
        (0.55, 0.00, 0.55),
        (0.90, 0.05, 0.05),
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
    ks = available_k(npz)
    if not ks:
        raise RuntimeError("No P{k}_centers found in NPZ.")
    if len(ks) < n_panels:
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


def get_indicator(npz: np.lib.npyio.NpzFile, k: int, h: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """
    Strategy-4 patchless snapshots are expected to store raw per-anchor residual
    squares as P{k}_r2. We still keep a few fallback modes for convenience.

    mode:
      - "r2":        use P{k}_r2
      - "gamma_r2":  use (1/h^2) * P{k}_r2
      - "eta":       use P{k}_eta
      - "eta_gamma": use P{k}_eta_gamma or P{k}_gamma_eta or P{k}_eta*(1/h^2)
      - "auto":      prefer r2 if present, then gamma_r2, then eta-based fallbacks
    """
    mode = (mode or "auto").strip().lower()

    def have(key: str) -> bool:
        return key in npz.files

    if mode == "auto":
        if have(f"P{k}_r2"):
            mode = "r2"
        elif have(f"P{k}_eta_gamma") or have(f"P{k}_gamma_eta") or have(f"P{k}_eta"):
            mode = "eta_gamma"
        else:
            mode = "r2"

    if mode == "r2":
        key = f"P{k}_r2"
        if key not in npz.files:
            raise KeyError(f"{key} not found in snapshots.")
        return np.asarray(npz[key], dtype=float).reshape(-1), key

    if mode == "gamma_r2":
        key = f"P{k}_r2"
        if key not in npz.files:
            raise KeyError(f"{key} not found in snapshots.")
        r2 = np.asarray(npz[key], dtype=float).reshape(-1)
        gamma = 1.0 / np.maximum(h * h, 1e-300)
        return gamma * r2, f"(1/h^2) * {key}"

    if mode == "eta":
        key = f"P{k}_eta"
        if key not in npz.files:
            raise KeyError(f"{key} not found in snapshots.")
        return np.asarray(npz[key], dtype=float).reshape(-1), key

    if mode == "eta_gamma":
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
        raise KeyError(f"No eta data found for P{k} (eta_gamma/gamma_eta/eta).")

    raise ValueError(f"Unknown indicator mode: {mode}")


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


def add_letter(ax: plt.Axes, letter: str):
    ax.text(0.5, -0.22, f"({letter})", transform=ax.transAxes,
            ha="center", va="top", fontsize=18)


def plot_six_panels_by_counts(
    npz_path: str,
    out_png: str,
    title: str,
    target_counts: List[int] | None,
    indicator: str,
    shared_colorbar: bool,
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
        vals, key_used = get_indicator(npz, k, h=h, mode=indicator)
        panel_data.append((k, actual_n, exact, centers, h, vals, key_used))

    norm = robust_lognorm_from_list([d[5] for d in panel_data])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    letters = "abcdef"

    last_sc = None
    for i, (k, actual_n, exact, centers, h, vals, key_used) in enumerate(panel_data):
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
            c=np.maximum(vals, 1e-12),
            s=sizes,
            cmap=BPR_CMAP,
            norm=norm,
            linewidths=0.0,
            alpha=1.0,
            rasterized=True,
        )
        last_sc = sc

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.tick_params(labelsize=11)

        if not shared_colorbar:
            cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb.ax.tick_params(labelsize=10)

        add_letter(ax, letters[i])
        ax.set_title(f"N = {actual_n}", fontsize=9)

    if shared_colorbar and last_sc is not None:
        cb = fig.colorbar(last_sc, ax=axes.tolist(), fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--cm4-npz", required=True)
    ap.add_argument("--cm9-npz", required=True)

    ap.add_argument("--size-scale", type=float, default=2500.0)
    ap.add_argument("--size-min", type=float, default=2.0)
    ap.add_argument("--size-max", type=float, default=100.0)

    ap.add_argument("--cm4-counts", type=str, default="auto",
                    help='6 target anchor-count values for CM=4 panels (comma-separated) or "auto".')
    ap.add_argument("--cm9-counts", type=str, default="auto",
                    help='6 target anchor-count values for CM=9 panels (comma-separated) or "auto".')

    ap.add_argument("--indicator", choices=["auto", "r2", "gamma_r2", "eta", "eta_gamma"], default="auto")
    ap.add_argument("--shared-colorbar", action="store_true",
                    help="If set: one shared colorbar for the whole 2x3 figure. Default: per-panel.")

    args = ap.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(args.cm4_npz):
        raise FileNotFoundError(args.cm4_npz)
    if not os.path.exists(args.cm9_npz):
        raise FileNotFoundError(args.cm9_npz)

    cm4_counts = parse_csv_ints_or_auto(args.cm4_counts)
    cm9_counts = parse_csv_ints_or_auto(args.cm9_counts)

    out14 = os.path.join(outdir, "figure14.png")
    out15 = os.path.join(outdir, "figure15.png")

    plot_six_panels_by_counts(
        npz_path=args.cm4_npz,
        out_png=out14,
        title=r"Strategy #4 anchors ($C_M = 4$)",
        target_counts=cm4_counts,
        indicator=args.indicator,
        shared_colorbar=bool(args.shared_colorbar),
        size_scale=float(args.size_scale),
        size_clip=(float(args.size_min), float(args.size_max)),
    )

    plot_six_panels_by_counts(
        npz_path=args.cm9_npz,
        out_png=out15,
        title=r"Strategy #4 anchors ($C_M = 9$)",
        target_counts=cm9_counts,
        indicator=args.indicator,
        shared_colorbar=bool(args.shared_colorbar),
        size_scale=float(args.size_scale),
        size_clip=(float(args.size_min), float(args.size_max)),
    )

    print("Saved:", out14)
    print("Saved:", out15)


if __name__ == "__main__":
    main()
