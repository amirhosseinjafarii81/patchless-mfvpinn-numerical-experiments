#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import Optional, Dict, List, Iterable, Any

import numpy as np


# -----------------------------
# Path normalization (Windows WSL UNC -> Linux path)
# -----------------------------
def normalize_path(p: str) -> str:
    if not p:
        return p

    low = p.lower()
    if low.startswith("\\\\wsl.localhost\\") or low.startswith("\\\\wsl$\\"):
        parts = p.split("\\")
        if len(parts) >= 5:
            linux_parts = parts[4:]
            return "/" + "/".join([x for x in linux_parts if x])

    return p.replace("\\", "/")


# -----------------------------
# Regex patterns
# -----------------------------
START_STRATEGY_RE = re.compile(
    r"Starting\s+Strategy\s*#?\s*(?P<sid>\d+).*?\bCM\s*=\s*(?P<cm>[49])\b",
    re.IGNORECASE,
)
RUN_CM_RE = re.compile(r"\bRUN:\s*CM\s*=\s*(?P<cm>[49])\b", re.IGNORECASE)
CM_RE = re.compile(r"\bCM\s*=\s*(?P<cm>[49])\b", re.IGNORECASE)

# Patchless logs (strategies 1..4 in your meshfree code family)
PATCHLESS_STEP_RE = re.compile(
    r"\bStep\s+(?P<step>\d+)\s*:\s*"
    r".*?\bM\s*=\s*(?P<N>\d+)\b"
    r"(?:.*?\bloss\s*=\s*(?P<loss>[0-9eE.+\-]+)\b)?"
    r"(?:.*?\bRh2\s*=\s*(?P<Rh2>[0-9eE.+\-]+)\b)?"
    r"(?:.*?\bH1_raw\s*=\s*(?P<H1raw>[0-9eE.+\-]+)\b)?"
    r"(?:.*?\bH1_end\s*=\s*(?P<H1end>[0-9eE.+\-]+)\b)?"
    r"(?:.*?\bH1_best\s*=\s*(?P<H1best>[0-9eE.+\-]+)\b)?",
    re.IGNORECASE,
)

# Patch-based Step lines
PATCH_STEP_RE = re.compile(
    r"\bStep\s+(?P<step>\d+)\s*:\s*"
    r".*?\bNpatch\s*=\s*(?P<N>\d+)\b"
    r".*?\bH1(?:_Err)?\s*=\s*(?P<H1>[0-9eE.+\-]+)\b"
    r"(?:.*?\bRh2\s*=\s*(?P<Rh2>[0-9eE.+\-]+)\b)?"
    r"(?:.*?\bloss\s*=\s*(?P<loss>[0-9eE.+\-]+)\b)?",
    re.IGNORECASE,
)

# Patch-based strategy4 style: [iter 00] Npatch= ... H1= ... Rh2= ... loss= ...
ITER_STEP_RE = re.compile(
    r"\[iter\s+(?P<step>\d+)]"
    r".*?\bNpatch\s*=\s*(?P<N>\d+)\b"
    r".*?\bH1\s*=\s*(?P<H1>[0-9eE.+\-]+)\b"
    r"(?:.*?\bRh2\s*=\s*(?P<Rh2>[0-9eE.+\-]+)\b)?"
    r"(?:.*?\bloss\s*=\s*(?P<loss>[0-9eE.+\-]+)\b)?",
    re.IGNORECASE,
)


def _nan_or_float(s: Optional[str]) -> float:
    if s is None or s == "":
        return float("nan")
    return float(s)


def _choose_metric_alias(rec: Dict[str, float], metric: str) -> float:
    metric = (metric or "h1_raw").strip().lower()
    if metric == "h1_raw":
        for k in ("H1_raw", "H1", "H1_end", "H1_best"):
            v = rec.get(k, np.nan)
            if np.isfinite(v):
                return float(v)
    elif metric == "h1_end":
        for k in ("H1_end", "H1", "H1_raw", "H1_best"):
            v = rec.get(k, np.nan)
            if np.isfinite(v):
                return float(v)
    elif metric == "h1_best":
        for k in ("H1_best", "H1", "H1_raw", "H1_end"):
            v = rec.get(k, np.nan)
            if np.isfinite(v):
                return float(v)
    elif metric == "h1":
        for k in ("H1", "H1_raw", "H1_end", "H1_best"):
            v = rec.get(k, np.nan)
            if np.isfinite(v):
                return float(v)
    return float("nan")


# -----------------------------
# Core parser
# -----------------------------
def parse_runlogs(
    runlog_paths: Iterable[str],
    force_cm: Optional[int] = None,
    metric_alias: str = "h1_raw",
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Parse one or multiple logs and return per-CM arrays.

    Output per CM contains:
      N, H1, H1_raw, H1_end, H1_best, Rh2, loss

    `H1` is an alias selected by `metric_alias`, defaulting to H1_raw when present.
    """
    points: Dict[int, List[Dict[str, Any]]] = {4: [], 9: []}
    cur_cm: Optional[int] = force_cm

    def cm_fallback() -> int:
        if cur_cm in (4, 9):
            return int(cur_cm)
        if force_cm in (4, 9):
            return int(force_cm)
        return 4

    for raw_path in runlog_paths:
        runlog_path = normalize_path(raw_path)
        if not os.path.exists(runlog_path):
            raise FileNotFoundError(f"runlog not found: {runlog_path}")

        with open(runlog_path, "r", errors="replace") as f:
            for line in f:
                m = RUN_CM_RE.search(line)
                if m:
                    cur_cm = int(m.group("cm"))
                else:
                    m = START_STRATEGY_RE.search(line)
                    if m:
                        cur_cm = int(m.group("cm"))
                    elif force_cm is None:
                        mcm = CM_RE.search(line)
                        if mcm:
                            cur_cm = int(mcm.group("cm"))

                mstep = PATCHLESS_STEP_RE.search(line)
                if mstep:
                    cm_use = cm_fallback()
                    rec = {
                        "N": int(mstep.group("N")),
                        "H1_raw": _nan_or_float(mstep.group("H1raw")),
                        "H1_end": _nan_or_float(mstep.group("H1end")),
                        "H1_best": _nan_or_float(mstep.group("H1best")),
                        "Rh2": _nan_or_float(mstep.group("Rh2")),
                        "loss": _nan_or_float(mstep.group("loss")),
                    }
                    # Alias plain H1 to the raw metric first for patchless logs.
                    rec["H1"] = _choose_metric_alias(rec, "h1_raw")
                    points[cm_use].append(rec)
                    continue

                mstep = PATCH_STEP_RE.search(line)
                if mstep:
                    cm_use = cm_fallback()
                    h1 = _nan_or_float(mstep.group("H1"))
                    rec = {
                        "N": int(mstep.group("N")),
                        "H1": h1,
                        "H1_raw": h1,
                        "H1_end": h1,
                        "H1_best": h1,
                        "Rh2": _nan_or_float(mstep.group("Rh2")),
                        "loss": _nan_or_float(mstep.group("loss")),
                    }
                    points[cm_use].append(rec)
                    continue

                mstep = ITER_STEP_RE.search(line)
                if mstep:
                    cm_use = cm_fallback()
                    h1 = _nan_or_float(mstep.group("H1"))
                    rec = {
                        "N": int(mstep.group("N")),
                        "H1": h1,
                        "H1_raw": h1,
                        "H1_end": h1,
                        "H1_best": h1,
                        "Rh2": _nan_or_float(mstep.group("Rh2")),
                        "loss": _nan_or_float(mstep.group("loss")),
                    }
                    points[cm_use].append(rec)
                    continue

    parsed: Dict[int, Dict[str, np.ndarray]] = {}

    for cm in (4, 9):
        if not points[cm]:
            continue

        # keep last occurrence for each N
        last_by_N: Dict[int, Dict[str, float]] = {}
        for p in points[cm]:
            last_by_N[int(p["N"])] = {
                "H1_raw": float(p.get("H1_raw", np.nan)),
                "H1_end": float(p.get("H1_end", np.nan)),
                "H1_best": float(p.get("H1_best", np.nan)),
                "H1_plain": float(p.get("H1", np.nan)),
                "Rh2": float(p.get("Rh2", np.nan)),
                "loss": float(p.get("loss", np.nan)),
            }

        Ns = np.array(sorted(last_by_N.keys()), dtype=np.int64)
        H1_raw = np.array([last_by_N[n]["H1_raw"] for n in Ns], dtype=np.float64)
        H1_end = np.array([last_by_N[n]["H1_end"] for n in Ns], dtype=np.float64)
        H1_best = np.array([last_by_N[n]["H1_best"] for n in Ns], dtype=np.float64)
        H1_plain = np.array([last_by_N[n]["H1_plain"] for n in Ns], dtype=np.float64)
        Rh2 = np.array([last_by_N[n]["Rh2"] for n in Ns], dtype=np.float64)
        loss = np.array([last_by_N[n]["loss"] for n in Ns], dtype=np.float64)

        alias_vals = []
        for i in range(len(Ns)):
            rec = {
                "H1": H1_plain[i],
                "H1_raw": H1_raw[i],
                "H1_end": H1_end[i],
                "H1_best": H1_best[i],
            }
            alias_vals.append(_choose_metric_alias(rec, metric_alias))
        H1_alias = np.array(alias_vals, dtype=np.float64)

        parsed[cm] = {
            "N": Ns,
            "H1": H1_alias,
            "H1_raw": H1_raw,
            "H1_end": H1_end,
            "H1_best": H1_best,
            "Rh2": Rh2,
            "loss": loss,
        }

    return parsed


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse patchless / patch-based run logs into history npz files per CM."
    )
    ap.add_argument(
        "--runlog",
        required=True,
        nargs="+",
        help="Path(s) to run logs. You can pass one or multiple logs to parse / merge.",
    )
    ap.add_argument(
        "--out-prefix",
        required=True,
        help="Example: /path/history_strategy1  -> saves history_strategy1_CM4.npz and/or _CM9.npz",
    )
    ap.add_argument(
        "--force-cm",
        type=int,
        choices=[4, 9],
        default=None,
        help="Force CM if the log never prints CM=4/9.",
    )
    ap.add_argument(
        "--metric-alias",
        type=str,
        default="h1_raw",
        choices=["h1", "h1_raw", "h1_end", "h1_best"],
        help="Which series should be aliased into the saved `H1` key. Default: h1_raw.",
    )

    args = ap.parse_args()
    runlogs = [normalize_path(p) for p in args.runlog]
    out_prefix = normalize_path(args.out_prefix)

    try:
        parsed = parse_runlogs(runlogs, force_cm=args.force_cm, metric_alias=args.metric_alias)
    except Exception as e:
        print(f"[error] {e}")
        return 2

    if not parsed:
        print(
            "[warn] No usable step lines found.\n"
            "Supported examples include:\n"
            "  Step 00: M=  25 | loss=... | H1_raw=... | H1_end=... | H1_best=...\n"
            "  Step 0: Npatch=5, H1_Err=..., Rh2=...\n"
            "  [iter 00] Npatch=     1  H1=...  Rh2=...  loss=..."
        )
        return 0

    for cm, data in parsed.items():
        out_path = f"{out_prefix}_CM{cm}.npz"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        np.savez(
            out_path,
            N=data["N"],
            H1=data["H1"],
            H1_raw=data["H1_raw"],
            H1_end=data["H1_end"],
            H1_best=data["H1_best"],
            Rh2=data["Rh2"],
            loss=data["loss"],
            cm=np.array([cm], dtype=np.int64),
        )
        print(f"[saved] {out_path}  (#points={len(data['N'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
