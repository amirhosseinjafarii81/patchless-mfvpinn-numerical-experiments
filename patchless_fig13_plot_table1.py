#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _dedupe_last_by_n(N: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    last = {}
    for n, y in zip(N.tolist(), Y.tolist()):
        last[float(n)] = float(y)
    Ns = np.array(sorted(last.keys()), dtype=np.float64)
    Ys = np.array([last[n] for n in Ns], dtype=np.float64)
    return Ns, Ys


def _pick_metric_array(z: np.lib.npyio.NpzFile, metric_key: str) -> np.ndarray:
    key = (metric_key or "H1_raw").strip().lower()
    candidates = {
        "h1_raw": ["H1_raw", "H1", "H1_end", "H1_best"],
        "h1_end": ["H1_end", "H1", "H1_raw", "H1_best"],
        "h1_best": ["H1_best", "H1", "H1_raw", "H1_end"],
        "h1": ["H1", "H1_raw", "H1_end", "H1_best"],
    }[key]
    for cand in candidates:
        if cand in z.files:
            return np.array(z[cand], dtype=np.float64)
    raise KeyError(f"None of the metric keys {candidates} found in {getattr(z, 'filename', '<npz>')}")


def load_hist(npz_path: str, metric_key: str = "H1_raw"):
    z = np.load(npz_path)
    if "N" in z.files:
        N = np.array(z["N"], dtype=np.float64)
    elif "M" in z.files:
        N = np.array(z["M"], dtype=np.float64)
    else:
        raise KeyError(f"No N/M array found in {npz_path}")

    H1 = _pick_metric_array(z, metric_key=metric_key)
    m = np.isfinite(N) & np.isfinite(H1) & (N > 0) & (H1 > 0)
    N, H1 = N[m], H1[m]
    N, H1 = _dedupe_last_by_n(N, H1)
    idx = np.argsort(N)
    return N[idx], H1[idx]


def load_ref_csv(csv_path: str):
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        if data.dtype.names and len(data.dtype.names) >= 2:
            cols = data.dtype.names
            N = np.asarray(data[cols[0]], dtype=float)
            H1 = np.asarray(data[cols[1]], dtype=float)
        else:
            raise ValueError("fallback")
    except Exception:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data[None, :]
        N = data[:, 0].astype(float)
        H1 = data[:, 1].astype(float)

    m = np.isfinite(N) & np.isfinite(H1) & (N > 0) & (H1 > 0)
    N, H1 = N[m], H1[m]
    N, H1 = _dedupe_last_by_n(N, H1)
    idx = np.argsort(N)
    return N[idx], H1[idx]


def fit_rate_loglog(N, H1, n_drop_first=0):
    if len(N) < 3:
        return np.nan, np.nan, (np.nan, np.nan)

    N2, H2 = N.copy(), H1.copy()
    if n_drop_first > 0 and len(N2) > n_drop_first + 2:
        N2 = N2[n_drop_first:]
        H2 = H2[n_drop_first:]

    if len(N2) < 3:
        return np.nan, np.nan, (np.nan, np.nan)

    x = np.log(N2)
    y = np.log(H2)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return float(b), float(r2), (float(N2[0]), float(N2[-1]))


def _trim_to_common_start(series_cm: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    mins = []
    for _, (N, H1) in series_cm.items():
        if len(N) > 0:
            mins.append(float(N[0]))
    if not mins:
        return series_cm, None

    N_start = 49
    trimmed = {}
    for name, (N, H1) in series_cm.items():
        m = N >= N_start
        Nt = N[m]
        Ht = H1[m]
        if len(Nt) >= 1:
            trimmed[name] = (Nt, Ht)
    return trimmed, N_start


def plot_panel(series_cm: Dict[str, Tuple[np.ndarray, np.ndarray]], cm: int, out_png: str, xlabel: str, ylabel: str):
    series_cm_trim, N_start = _trim_to_common_start(series_cm)

    fig, ax = plt.subplots()
    for name, (N, H1) in series_cm_trim.items():
        ax.loglog(N, H1, marker="o", linewidth=1.5, markersize=4, label=name)

    if N_start is not None and series_cm_trim:
        allN = np.concatenate([v[0] for v in series_cm_trim.values()])
        xmin = float(N_start)
        xmax = float(np.max(allN))
        ax.set_xlim(xmin, xmax)

        allH = np.concatenate([v[1] for v in series_cm_trim.values()])
        ymin = float(np.min(allH))
        ymax = float(np.max(allH))
        ax.set_ylim(ymin * 0.95, ymax * 1.05)

    ax.margins(x=0.0, y=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Figure 13 ({'a' if cm == 4 else 'b'})  CM={cm}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--s1_cm4", required=True)
    ap.add_argument("--s1_cm9", required=True)
    ap.add_argument("--s2_cm4", required=True)
    ap.add_argument("--s2_cm9", required=True)
    ap.add_argument("--s3_cm4", required=True)
    ap.add_argument("--s3_cm9", required=True)
    ap.add_argument("--s4_cm4", required=True)
    ap.add_argument("--s4_cm9", required=True)

    ap.add_argument("--ref_cm4", default="")
    ap.add_argument("--ref_cm9", default="")

    ap.add_argument("--metric-key", type=str, default="H1_raw",
                    choices=["H1", "H1_raw", "H1_end", "H1_best"],
                    help="Which history key to use for the patchless curves. Default: H1_raw.")
    ap.add_argument("--drop-first", type=int, default=0)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    series = {
        4: {
            "Strategy 1": load_hist(args.s1_cm4, metric_key=args.metric_key),
            "Strategy 2": load_hist(args.s2_cm4, metric_key=args.metric_key),
            "Strategy 3": load_hist(args.s3_cm4, metric_key=args.metric_key),
            "Strategy 4": load_hist(args.s4_cm4, metric_key=args.metric_key),
        },
        9: {
            "Strategy 1": load_hist(args.s1_cm9, metric_key=args.metric_key),
            "Strategy 2": load_hist(args.s2_cm9, metric_key=args.metric_key),
            "Strategy 3": load_hist(args.s3_cm9, metric_key=args.metric_key),
            "Strategy 4": load_hist(args.s4_cm9, metric_key=args.metric_key),
        }
    }

    if args.ref_cm4:
        series[4]["VPINN (reference)"] = load_ref_csv(args.ref_cm4)
    if args.ref_cm9:
        series[9]["VPINN (reference)"] = load_ref_csv(args.ref_cm9)

    fig13a = os.path.join(args.outdir, "fig13a_reproduced.png")
    fig13b = os.path.join(args.outdir, "fig13b_reproduced.png")
    plot_panel(series[4], 4, fig13a, "Number of test functions (anchors)", rf"Relative $H^1$ error")
    plot_panel(series[9], 9, fig13b, "Number of test functions (anchors)", rf"Relative $H^1$ error")
    print(f"[saved] {fig13a}")
    print(f"[saved] {fig13b}")

    rows = []
    for cm in (4, 9):
        for name, (N, H1) in series[cm].items():
            b, r2, (nmin, nmax) = fit_rate_loglog(N, H1, n_drop_first=args.drop_first)
            rows.append((cm, name, b, -b, r2, nmin, nmax, len(N)))

    table_csv = os.path.join(args.outdir, "table1_rates.csv")
    with open(table_csv, "w", encoding="utf-8") as f:
        f.write("CM,method,slope_b,rate_minus_b,R2,Nmin_used,Nmax_used,npoints\n")
        for cm, name, b, minus_b, r2, nmin, nmax, npts in rows:
            f.write(f"{cm},{name},{b:.6g},{minus_b:.6g},{r2:.6g},{nmin:.6g},{nmax:.6g},{npts}\n")

    print(f"[saved] {table_csv}")

    print("\n=== Table 1 (log-log fit) ===")
    for cm, name, b, minus_b, r2, nmin, nmax, npts in rows:
        print(
            f"CM={cm:>1}  {name:<18}  slope={b: .4f}  (rate={-b: .4f})  "
            f"R2={r2:.3f}  N=[{nmin:.0f},{nmax:.0f}]  pts={npts}"
        )

if __name__ == "__main__":
    main()
