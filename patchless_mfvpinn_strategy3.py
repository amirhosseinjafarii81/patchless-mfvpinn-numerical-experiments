# -*- coding: utf-8 -*-
"""
fig3_strategy1_kernelized_rbf_patchless.py

Patchless / coverless "MF-VPINN" variant for Figure 3 style benchmark (unit square, corner singularity).

Core idea (reviewer-proof meshfree):
- No mesh, no triangulation, no patch-bank cover.
- Test space is RKHS / kernelized RBF: v_j(x) = b(x) * k(||x-ξ_j||; ℓ_j)
  where b(x)=x(1-x)y(1-y) enforces v_j|∂Ω = 0.
- Weak residuals are computed by global QMC integration over Ω:
      R_j(θ) = ∫_Ω ∇u_θ · ∇v_j  dx
  and loss is (whitened / normalized):
      L = mean_j  a_j * (R_j / sqrt(G_j+eps))^2
      G_j = ∫_Ω ||∇v_j||^2 dx
  which is a practical inf-sup stabilizer (diagonal whitening).
- Adaptivity adds *anchors* (ξ_j) each outer iteration using a strong-residual indicator:
      η(z)=|Δu_θ(z)|  (Laplace since f=0 in this benchmark)
  sampled on a global candidate cloud (QMC) with Poisson-disk style rejection.
- New anchors enter with small weight a_j and are ramped in over outer iterations.
- Monotone reporting: we track best-so-far holdout H1 and plot that (guaranteed non-increasing).

This script is intentionally self-contained and "production" oriented:
- float64 throughout (matches the original)
- GPU-safe batching for (points × anchors)
- optional L-BFGS stage (can be disabled by setting maxiter=0)

Author: (modified by assistant)
"""

import gc
import copy
import argparse
import math

_math = math  # alias used in a few places
import os
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np

_np = np  # alias used throughout (kept for backward-compat)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import scipy.optimize as sopt
from scipy.stats import qmc as sp_qmc
from scipy.spatial import cKDTree


# -------------------------
# KDTree-based kernel pruning (LATEX-aligned, truly meshfree)
# -------------------------
class KDTreePruner:
    """
    Stores per-anchor frozen neighbor index lists I_j = {k: ||x_k - a_j|| <= c * ell_j}.
    - Strict nestedness: once an anchor is created, its I_j never changes (because a_j, ell_j are frozen).
    - Dense fallback: if an anchor's neighborhood is too large, we mark it as "dense" (None) and use the
      original dense matmul kernels for evaluation/energy. This avoids pathological memory blowups.
    """

    def __init__(self, Xq_np: np.ndarray, cutoff_factor: float = 4.0,
                 leaf_size: int = 40, max_local: int = 0, dense_ratio: float = 0.75):
        self.Xq_np = np.asarray(Xq_np, dtype=np.float64)
        self.Nq = int(self.Xq_np.shape[0])
        self.cutoff_factor = float(cutoff_factor)
        self.leaf_size = int(leaf_size)
        self.max_local = int(max_local)  # 0 => no hard cap, but dense_ratio still applies
        self.dense_ratio = float(dense_ratio)
        self.tree = cKDTree(self.Xq_np, leafsize=self.leaf_size)
        self.idx_lists = []  # list[Optional[np.ndarray]] of length M
        # Optional per-anchor stencil bank (CPU speed win): stores pre-sampled point indices
        # for each anchor neighborhood so training can gather edges in TF without rebuilding
        # giant tf.constant edge lists each step.
        self._stencil_enabled = False
        self._stencil_K = 0
        self._stencil_Smax = 0
        self._stencil_seed = 0
        self._stencil_np = None  # shape (M, K, Smax) int32
        self._nloc_np = None  # shape (M,) int32 (|I_j| or Nq for dense anchors)

    @classmethod
    def build(cls, Xq_np: np.ndarray, cutoff_factor: float = 4.0,
              leaf_size: int = 40, max_local: int = 0, dense_ratio: float = 0.75) -> "KDTreePruner":
        return cls(Xq_np, cutoff_factor=cutoff_factor, leaf_size=leaf_size,
                   max_local=max_local, dense_ratio=dense_ratio)

    def _make_idx(self, a: np.ndarray, ell: float):
        r = float(self.cutoff_factor) * float(ell)
        idx = self.tree.query_ball_point(a, r)
        if not idx:
            return np.empty((0,), dtype=np.int32)
        idx = np.asarray(idx, dtype=np.int32)

        # Dense fallback rule (robust + less trigger-happy):
        # - Primary trigger: neighborhood is near-global (ratio-based).
        # - Optional hard cap (max_local) is treated as a *soft* guard: we only force dense if the
        #   neighborhood is both above max_local AND near-global. This prevents "unnecessary dense"
        #   when max_local is set too low.
        thr_ratio = int(self.dense_ratio * self.Nq)
        too_big_ratio = (idx.shape[0] > thr_ratio)
        too_big_hard = (0 < self.max_local < idx.shape[0])
        near_global = (idx.shape[0] > int(0.90 * self.Nq))

        if too_big_ratio or (too_big_hard and near_global):
            return None
        return idx

    def init_for_anchors(self, anchors: np.ndarray, ell: np.ndarray):
        self.idx_lists = []
        if self._stencil_enabled:
            self._stencil_np = None
            self._nloc_np = None
        anchors = np.asarray(anchors, dtype=np.float64)
        ell = np.asarray(ell, dtype=np.float64).reshape(-1)
        for j in range(int(anchors.shape[0])):
            self.idx_lists.append(self._make_idx(anchors[j], float(ell[j])))

    def append(self, anchors_new: np.ndarray, ell_new: np.ndarray):
        anchors_new = np.asarray(anchors_new, dtype=np.float64)
        ell_new = np.asarray(ell_new, dtype=np.float64).reshape(-1)
        for j in range(int(anchors_new.shape[0])):
            self.idx_lists.append(self._make_idx(anchors_new[j], float(ell_new[j])))
        # Invalidate stencil bank (if enabled); it will be rebuilt lazily on the next training call.
        if self._stencil_enabled:
            self._stencil_np = None
            self._nloc_np = None

    def is_pruned(self, anchor_id: int) -> bool:
        v = self.idx_lists[int(anchor_id)]
        return v is not None

    def local_size(self, anchor_id: int) -> int:
        v = self.idx_lists[int(anchor_id)]
        return int(v.shape[0]) if v is not None else self.Nq

    def summary(self) -> Dict[str, Any]:
        """Quick stats for tuning prune/fallback behavior."""
        M = int(len(self.idx_lists))
        n_dense = int(sum(1 for v in self.idx_lists if v is None))
        sizes = np.asarray([int(v.shape[0]) for v in self.idx_lists if v is not None], dtype=np.int64)
        if sizes.size == 0:
            return {
                "M": M, "n_dense": n_dense, "frac_dense": float(n_dense / max(1, M)),
                "mean_local": 0.0, "p50_local": 0.0, "p90_local": 0.0, "max_local": 0.0,
            }
        return {
            "M": M, "n_dense": n_dense, "frac_dense": float(n_dense / max(1, M)),
            "mean_local": float(np.mean(sizes)),
            "p50_local": float(np.percentile(sizes, 50)),
            "p90_local": float(np.percentile(sizes, 90)),
            "max_local": float(np.max(sizes)),
        }

    def summary_str(self) -> str:
        s = self.summary()
        return ("KDTreePruner: M={M}  dense={n_dense} ({frac_dense:.1%})  "
                "local(mean/p50/p90/max)={mean_local:.0f}/{p50_local:.0f}/{p90_local:.0f}/{max_local:.0f}  "
                "Nq={Nq}  dense_ratio={dense_ratio:.2f}  max_local_cap={max_local_cap}").format(
            M=s["M"], n_dense=s["n_dense"], frac_dense=s["frac_dense"],
            mean_local=s["mean_local"], p50_local=s["p50_local"], p90_local=s["p90_local"], max_local=s["max_local"],
            Nq=self.Nq, dense_ratio=self.dense_ratio, max_local_cap=self.max_local
        )

    # ---------- Stencil bank (optional; big speed win on CPU) ----------
    def enable_stencil_bank(self, Smax: int, K: int = 4, seed: int = 1234, force_rebuild: bool = False):
        """Build a per-anchor stencil bank once, to avoid Python edge-list rebuild every step.

        Creates:
          _stencil_np: (M, K, Smax) int32, where each [j,k,:] is a point-index stencil for anchor j.
          _nloc_np:    (M,) int32, neighborhood size |I_j| (or Nq for dense anchors).
        Notes:
          - This keeps the *nestedness* contract: idx_lists are frozen; stencils are just a fixed Monte-Carlo
            sample of each neighborhood. We only rebuild if forced or if dimensions change.
          - For m < Smax we sample with replacement (still unbiased for the neighborhood mean).
        """
        Smax = int(Smax)
        K = int(K)
        seed = int(seed)
        if Smax <= 0:
            raise ValueError("Smax must be positive")
        if K <= 0:
            raise ValueError("K must be positive")

        M = int(len(self.idx_lists))
        need = force_rebuild or (not self._stencil_enabled) or (self._stencil_np is None) or (self._nloc_np is None)
        need = need or (int(self._stencil_Smax) != Smax) or (int(self._stencil_K) != K) or (
                    int(self._stencil_seed) != seed)
        need = need or (self._stencil_np is not None and int(self._stencil_np.shape[0]) != M)
        if not need:
            return

        rng = np.random.default_rng(seed)
        st = np.empty((M, K, Smax), dtype=np.int32)
        nloc = np.empty((M,), dtype=np.int32)

        for j in range(M):
            idx = self.idx_lists[j]
            if idx is None:
                nloc[j] = np.int32(self.Nq)
                # Dense: just sample from the whole global cloud.
                for k in range(K):
                    st[j, k, :] = rng.integers(0, self.Nq, size=Smax, dtype=np.int32)
                continue

            m = int(idx.shape[0])
            nloc[j] = np.int32(m)
            if m <= 0:
                # Degenerate: fall back to global points (rare, but avoids NaNs)
                for k in range(K):
                    st[j, k, :] = rng.integers(0, self.Nq, size=Smax, dtype=np.int32)
                continue

            rep = (m < Smax)
            for k in range(K):
                st[j, k, :] = rng.choice(idx, size=Smax, replace=rep).astype(np.int32, copy=False)

        self._stencil_enabled = True
        self._stencil_K = K
        self._stencil_Smax = Smax
        self._stencil_seed = seed
        self._stencil_np = st
        self._nloc_np = nloc

    def stencil_bank_np(self) -> Optional[np.ndarray]:
        return self._stencil_np

    def nloc_np(self) -> Optional[np.ndarray]:
        return self._nloc_np

    def build_edges_for_batch(self, anchor_ids: np.ndarray, points_per_anchor: int, rng: np.random.Generator):
        """
        Build an edge list for a batch of anchors.
        Returns:
          idx_flat: (E,) point indices into the *global* Xq cloud
          seg_ids:  (E,) segment ids 0..A-1 mapping each edge to a batch-anchor slot
          scales:   (A,) per-anchor scaling to form an unbiased estimate of (1/Nq) * sum_{k in I_j} (...)
                   If we sample S points from I_j: scale = |I_j|/S; dense fallback uses |I_j|=Nq.
          nloc:     (A,) local neighborhood sizes (|I_j| or Nq for dense)
        """
        anchor_ids = np.asarray(anchor_ids, dtype=np.int32).reshape(-1)
        A = int(anchor_ids.shape[0])
        S = int(max(1, points_per_anchor))

        idx_chunks = []
        seg_chunks = []
        scales = np.zeros((A,), dtype=np.float64)
        nloc = np.zeros((A,), dtype=np.int32)

        for s, aid in enumerate(anchor_ids):
            idx = self.idx_lists[int(aid)]
            if idx is None:
                # Dense: sample uniformly from all points, cutoff handled in TF.
                sel = rng.integers(0, self.Nq, size=S, dtype=np.int32)
                nloc[s] = np.int32(self.Nq)
                scales[s] = float(self.Nq) / float(S)
            else:
                m = int(idx.shape[0])
                nloc[s] = np.int32(m)
                if m <= 0:
                    # Degenerate: fall back to global sampling.
                    sel = rng.integers(0, self.Nq, size=S, dtype=np.int32)
                    nloc[s] = np.int32(self.Nq)
                    scales[s] = float(self.Nq) / float(S)
                else:
                    rep = bool(m < S)
                    sel = rng.choice(idx, size=S, replace=rep).astype(np.int32, copy=False)
                    scales[s] = float(m) / float(S)

            idx_chunks.append(sel)
            seg_chunks.append(np.full((S,), s, dtype=np.int32))

        idx_flat = np.concatenate(idx_chunks, axis=0)
        seg_ids = np.concatenate(seg_chunks, axis=0)
        return idx_flat, seg_ids, scales, nloc

    def build_edges_full(self, anchor_ids: np.ndarray):
        """
        Full (non-subsampled) edges for a set of anchors: uses all points in each I_j.
        Dense anchors (None) are excluded by design: call this only for pruned ids.
        Returns idx_flat, seg_ids.
        """
        anchor_ids = np.asarray(anchor_ids, dtype=np.int32).reshape(-1)
        idx_chunks = []
        seg_chunks = []
        for s, aid in enumerate(anchor_ids):
            idx = self.idx_lists[int(aid)]
            if idx is None:
                raise ValueError("build_edges_full called on dense anchor; split ids beforehand.")
            idx_chunks.append(idx.astype(np.int32, copy=False))
            seg_chunks.append(np.full((idx.shape[0],), s, dtype=np.int32))
        if not idx_chunks:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
        return np.concatenate(idx_chunks, axis=0), np.concatenate(seg_chunks, axis=0)


@tf.function(reduce_retracing=True)
def gradv_edges_from_precomputed(xy, b0, db0, a_edge, ell_edge, cutoff_s=None):
    """
    Edge-based ∇v evaluation (no (points×anchors) matrix).
      v(x) = b0(x) * exp(-||x-a||^2 / ell^2)
    Inputs are all edge-aligned:
      xy:      (E,2)
      b0:      (E,1)
      db0:     (E,2)
      a_edge:  (E,2)
      ell_edge:(E,1)
    Returns:
      gradv: (E,2)
    """
    diff = xy - a_edge
    k, gradk = edge_kernel_k_and_grad(diff, ell_edge, cutoff_s=cutoff_s)
    gradv = db0 * k + b0 * gradk
    return gradv


@tf.function(reduce_retracing=True)
def segment_sum_1d(values, seg_ids, nseg):
    return tf.math.unsorted_segment_sum(values, seg_ids, nseg)


@tf.function(reduce_retracing=True)
def segment_sum_2d(values, seg_ids, nseg):
    return tf.math.unsorted_segment_sum(values, seg_ids, nseg)


try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

tf.keras.backend.set_floatx("float64")
DTYPE = tf.float64
# Kernel selection (set from CLI in main before first TF tracing)
KERNEL_KIND = "gauss"  # "gauss" or "wendland_c2"
KERNEL_CUTOFF_FACTOR = 4.0


# -------------------------
# Utilities (SciPy / TF interop)
# -------------------------
def _flatten_grads(grads, vars_):
    """Flatten gradients (handling None / IndexedSlices) into 1D numpy float64 for SciPy."""
    if vars_ is None:
        raise ValueError("vars_ must be provided to flatten grads safely.")
    parts = []
    for g, v in zip(grads, vars_):
        if g is None:
            parts.append(tf.zeros((tf.size(v),), dtype=v.dtype))
            continue
        if isinstance(g, tf.IndexedSlices):
            g = tf.convert_to_tensor(g)
        g = tf.cast(g, v.dtype)
        parts.append(tf.reshape(g, (-1,)))
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    flat_tf = tf.concat(parts, axis=0)
    # SciPy expects float64; casting here is fine since LBFGS is optional and expensive anyway.
    return flat_tf.numpy().astype(np.float64)


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    # Network
    L_LAYERS: int = 5
    WIDTH: int = 50

    # Training
    LR0: float = 1e-2
    LR1: float = 1e-4
    LAMBDA_REG: float = 1e-5

    ADAM_STEPS_FIRST: int = 2500
    ADAM_STEPS_AFTER_GROW: int = 1200

    LBFGS_MAXITER_FIRST: int = 200
    LBFGS_MAXITER_NEXT: int = 600
    LBFGS_FTOL: float = 1e-14
    LBFGS_GTOL: float = 1e-8  # dtype-aware default is also applied in lbfgs_optimize
    LBFGS_MAXFUN: int = 0  # 0 => SciPy default or 20*maxiter
    LBFGS_MAXCOR: int = 50

    # Batch sizes (GPU-safety)
    ADAM_POINT_BATCH: int = 4096
    ADAM_ANCHOR_BATCH: int = 64
    ADAM_GRAD_ACCUM: int = 16

    EVAL_POINT_BATCH: int = 8192
    EVAL_ANCHOR_BATCH: int = 128
    EVAL_EDGE_BATCH: int = 2048

    LBFGS_POINT_BATCH: int = 0  # 0 => use EVAL_POINT_BATCH
    LBFGS_ANCHOR_BATCH: int = 0  # 0 => use EVAL_ANCHOR_BATCH
    LBFGS_EDGE_BATCH: int = 0  # 0 => use EVAL_EDGE_BATCH

    # Global QMC points for weak residual integration
    NQ_GLOBAL: int = 65536
    QMC_KIND: str = "sobol"  # "sobol" or "halton"
    QMC_SCRAMBLE: bool = True
    QMC_SHIFT_STRIDE: int = 4096

    # Strong-residual (Laplace) candidate pool for anchor growth
    NCAND: int = 65536
    # --- Stability knobs (research-grade defaults) ---
    DOMAIN_NORMALIZE: int = 1  # map NN inputs from [0,1]^d to [-1,1]^d internally
    ELL_FROM_H: int = 1  # set base ell from mean NN spacing h
    ELL_C_RHO: float = 2.0  # rho = C_rho*h, with rho = cutoff*ell
    NQ_FACTOR: float = 4.0  # ensure NQ_GLOBAL >= NQ_FACTOR * TARGET_NANCHOR
    W_PDE: float = 1.0
    W_BC: float = 5.0  # (used in gnet training only; main solve uses lift+bubble)
    W_ENERGY: float = 0.0  # grad(u) regularization weight (0 => off; see ENERGY_POINT_BATCH)
    CAND_SEED: int = 12345
    ETA_POWER: float = 1.5  # sampling density p(z) ∝ (η+eps)^ETA_POWER
    ETA_EPS: float = 1e-12

    # Kernel (Gaussian by default): k(r)=exp(-(r/ell)^2)
    KERNEL_KIND: str = "gauss"  # "gauss" | "wendland_c2"
    ELL0: float = 0.15
    ELL_MIN: float = 0.015
    ELL_SCHED_P: float = 0.5  # ell(M) = max(ELL_MIN, ELL0*(M0/M)^ELL_SCHED_P)
    USE_MULTI_SCALE: bool = False
    ELL0_SMALL: float = 0.06  # used if USE_MULTI_SCALE
    ELL_MIX_FRAC_SMALL: float = 0.5
    # Kernel cutoff factor C: ignore contributions with r/ell > C (speed, safe when C~3-5)
    KERNEL_CUTOFF: float = 4.0

    # KDTree pruning (meshfree local quadrature subsets, strict nestedness)
    USE_KDTREE_PRUNING: int = 1
    KDTREE_LEAF: int = 40
    KDTREE_MAX_LOCAL: int = 0  # 0 => no hard cap; dense_ratio still applies
    KDTREE_DENSE_RATIO: float = 0.75
    # Stencil bank (major CPU speed win): pre-sample per-anchor quadrature indices once,
    # then gather in TF (no Python loops / no tf.constant huge edges each step).
    USE_STENCIL_BANK: int = 1
    STENCIL_BANK_K: int = 4  # number of stencils per anchor (cycled during training)
    STENCIL_SMAX: int = 0  # 0 => auto = ADAM_POINT_BATCH // ADAM_ANCHOR_BATCH (clamped >=8)
    STENCIL_SEED: int = 1234
    KDTREE_REPORT: int = 1
    # Cache energy G_j per outer iteration (recommended when WHITEN_MODE='energy')
    CACHE_ENERGY: int = 1
    NESTED_ENERGY: int = 1

    # Whitening (normalization) term eps in sqrt(G+eps)
    WHITEN_EPS: float = 1e-12

    # Whitening mode: "energy" = divide by sqrt(G+eps); "none" = no normalization
    WHITEN_MODE: str = "energy"

    # Adaptive anchor sampling: "eta" uses |Δu| weighting; "uniform" disables it
    ADAPTIVE_MODE: str = "eta"
    # Anchor growth policy
    ANCHOR_INIT: int = 25
    TARGET_NANCHOR: int = 1000
    OUTER_ITERS: int = 12

    ADD_FRAC_MAX: float = 1.0  # max added per outer iter as fraction of current M
    MIN_ADD_PER_ITER: int = 16
    MAX_ADD_PER_ITER: int = 4096

    # Growth policy for number of new anchors per outer iteration.
    # - "fraction": (default) use ADD_FRAC_MAX + MIN/MAX_ADD_PER_ITER
    # - "strategy1": mark+refine like fig3_strategy1 (Doerfler-style marking)
    # - "strategy2": fixed child-center refinement around marked anchors (fig6-style anchor refinement)
    # - "seq": follow a user-provided schedule of total anchor counts per step
    # - "strategy3": Strategy #3 (Small level gap constraint on low-level anchors)
    ANCHOR_GROW_POLICY: str = "strategy3"
    ANCHOR_CHILD_POLICY: str = "strategy2"
    MARK_FRAC: float = 0.75
    MARK_CAP: float = 0.30
    ANCHOR_CM: int = 4

    ANCHOR_SLG_GAP = 2  # L_max (admissible low-level)
    MARK_FRAC_L = 0.50  # Doerfler fraction for low-level set
    MARK_CAP_L = 0.20  # Cap for low-level marking

    # --- New: SLG "hard" constraints ---
    # Minimum refinement level allowed to produce children (enforce growth away from very low levels).
    L_MIN_GROW: int = 1

    # Penalize low-level anchors in the Strategy3 weights:
    # effective_weight = weight * exp(-SLG_LEVEL_PENALTY * (L_star - anchor_lvl)),
    # where L_star is a reference level (e.g. current max level). Zero disables penalty.
    SLG_LEVEL_PENALTY: float = 0.25

    # Hard cap on how many times an anchor can act as a parent (to avoid over-used parents).
    ANCHOR_MAX_REFINES: int = 1000000  # large default = effectively "no limit"

    # Strategy1 CM-refinement geometry (meshfree "children around parent anchor")
    # Mirrors the patch-based CM idea in fig3_strategy1.py:
    #   scale = sqrt(A_RATIO / CM) and lam ∈ [LAM_LO, LAM_HI]
    ANCHOR_CHILD_A_RATIO: float = 1.25
    ANCHOR_CHILD_LAM_LO: float = 0.9
    ANCHOR_CHILD_LAM_HI: float = 10.0 / 9.0
    ANCHOR_CHILD_OVERSAMPLE: int = 2  # propose CM*oversample candidates per marked anchor, pick best CM
    STRATEGY1_FILL_GLOBAL: int = 1  # if local CM proposals can't fill add_count, backfill from global candidates
    ANCHOR_GROW_SEQ: tuple = ()  # tuple[int, ...] parsed from CLI when policy=="seq"

    # Anchor weights (stability continuation)
    ANCHOR_WEIGHT_NEW: float = 0.0
    ANCHOR_WEIGHT_RAMP: float = 0.25

    # Poisson-disk style min separation between anchors
    MIN_SEP_ALPHA: float = 0.7
    REJECT_MAX_TRIES: int = 200000

    # Bubble for u (same as original options)
    BUBBLE_SCALE: float = 16.0
    LAPLACE_H: float = 1e-3  # FD step for Laplacian indicator in adaptive anchor growth
    BUBBLE_KIND: str = "product"  # "product" or "softmin" (product enforces u=0 on ∂Ω exactly)
    SOFTMIN_TAU: float = 1e-3
    BUBBLE_POWER: float = 1.0

    # Lift
    GNET_STEPS: int = 20000
    GNET_LR: float = 1e-3
    GNET_BATCH: int = 4096
    GNET_PRINT_EVERY: int = 2000

    GNET_CORNER_POWER: float = 10.0
    GNET_P_FOCUS: float = 0.70

    GNET_P_NORM: float = 8.0
    GNET_BETA_P: float = 0.05
    GNET_TOP_FRAC: float = 0.10
    GNET_BETA_TOP: float = 0.50

    GNET_GRID_N_PER_EDGE: int = 20000
    GNET_EVAL_CHUNK: int = 32768
    GNET_MAXERR_WARN: float = 5e-3

    # Relative H1 error
    H1_NSAMPLES: int = 65536
    H1_GRADE_P: float = 3.0
    # GI-QMC H1 evaluation (meshfree, low-variance)
    H1_USE_GIQMC: int = 1
    H1_IMPORT_ALPHA: float = 0.75
    H1_IMPORT_EPS: float = 1e-10
    H1_ETA_EMA_BETA: float = 0.0

    # H1 robustness / reporting controls
    # Freeze GI-QMC importance weights after a few calls to make H1 comparable across iterations.
    H1_FREEZE_IMPORTANCE: int = 1  # 1 => freeze q(x) after H1_FREEZE_AFTER_CALLS calls
    H1_FREEZE_AFTER_CALLS: int = 1  # freeze after this many GI-QMC weight builds (>=1)

    # Monotone H1 tracking / optional rollback (for paper-quality plots and stability)
    MONOTONE_H1_REPORT: int = 0  # 1 => report/store best-so-far H1 as H1_hold each step
    MONOTONE_H1_RESTORE: int = 0  # 1 => if current H1 worsens, restore best network weights
    MONOTONE_H1_TOL: float = 1e-3  # allow small relative increases before restoring

    RESTORE_BEST_AFTER_GROW: bool = False

    # Marking (strategy1-style) for adaptive anchor growth
    MARKING: str = "eta"  # eta | sqrtgamma_eta | gamma_eta

    # Growth scoring (strategy1)
    GROWTH_SCORE_MODE: str = "eta"  # eta | corr | projection
    GROWTH_ALPHA: float = 0.7
    GROWTH_BETA: float = 0.3
    GROWTH_PROJ_REG: float = 1e-6
    GROWTH_LOCAL_K: int = 8
    GROWTH_RHO_EPS: float = 1e-12
    GROWTH_LOC_NQ: int = 512
    GROWTH_SHORTLIST_FACTOR: float = 4.0
    H1_CHUNK: int = 16384
    H1_HOLDOUT_SAMPLES: int = 32768
    H1_HOLDOUT_SEED: int = 424242

    # Conditioning control (meshfree kernel stability / Gram conditioning)
    CONDCTRL_ENABLE: int = 1
    CONDCTRL_KAPPA_UNSTABLE: float = 1e8
    CONDCTRL_KAPPA_REJECT: float = 1e10
    CONDCTRL_SHRINK_GAMMA: float = 0.9
    CONDCTRL_MAX_SHRINK: int = 4
    CONDCTRL_EPS0: float = 1e-12
    CONDCTRL_ALPHA_LAMMAX: float = 1e-10
    CONDCTRL_ALPHA_TRACEM: float = 1e-8
    CONDCTRL_WHITEN: int = 1
    CONDCTRL_GROWTH_GAMMA: float = 0.0  # multiplier exponent for (lam_min/lam_max)^gamma; 0 disables

    # -------------------------
    # Low-oscillation, no-rollback monotone H1 controls
    # -------------------------
    DETERMINISTIC_BATCHING: int = 1
    FREEZE_CANDIDATES: int = 1
    RAMP_STAGES: int = 4

    # Enforce monotone decrease of *raw* H1 (no weight rollback):
    H1_MONO_ENFORCE: int = 1
    H1_MONO_TOL_REL: float = 0.0
    H1_MONO_MAX_PHASES: int = 8
    H1_MONO_EXTRA_STEPS: int = 200
    H1_MONO_LR_DECAY: float = 0.5
    H1_MONO_MIN_LR_SCALE: float = 0.05
    H1_MONO_VERBOSE: int = 1

    # Hard monotone mode: keep training (smaller LR) until H1<=target; if still fails, STOP (no rollback).
    H1_MONO_HARD: int = 1
    H1_MONO_HARD_BUDGET: int = 4000  # total extra Adam steps allowed beyond MAX_PHASES loop
    H1_MONO_HARD_EXTRA_STEPS: int = 200  # per hard round Adam steps
    H1_MONO_HARD_LR_DECAY: float = 0.5  # additional LR decay in hard mode
    H1_MONO_HARD_MIN_LR_SCALE: float = 0.02  # allow smaller LR than MIN_LR_SCALE
    H1_MONO_HARD_VERBOSE: int = 1
    # Seeds
    SEED: int = 2000


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 1234):
    """Set Python/NumPy/TF seeds in one place (best-effort reproducibility)."""
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except Exception:
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))


# -------------------------
# Exact solution (same as original)
# -------------------------
ALPHA = 2.0 / 3.0


def u_exact_np(x, y):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    u = np.zeros_like(r, dtype=np.float64)
    m = r > 0
    u[m] = (r[m] ** ALPHA) * np.sin(ALPHA * (th[m] + np.pi / 2.0))
    return u


def grad_u_exact_np(x, y):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    gx = np.zeros_like(r, dtype=np.float64)
    gy = np.zeros_like(r, dtype=np.float64)
    m = r > 0
    ang = ALPHA * (th[m] + np.pi / 2.0) - th[m]
    coeff = ALPHA * (r[m] ** (ALPHA - 1.0))
    gx[m] = coeff * np.sin(ang)
    gy[m] = coeff * np.cos(ang)
    return gx, gy


@tf.function
def u_exact_tf(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    r = tf.sqrt(tf.maximum(x * x + y * y, tf.constant(0.0, dtype=DTYPE)))
    th = tf.atan2(y, x)
    a = tf.constant(ALPHA, dtype=DTYPE)
    u = tf.pow(r, a) * tf.sin(a * (th + tf.constant(np.pi / 2.0, dtype=DTYPE)))
    u = tf.where(r > tf.constant(0.0, dtype=DTYPE), u, tf.zeros_like(u))
    return u


# -------------------------
# Lift: Coons patch from boundary data (exact on boundary)
# -------------------------
@tf.function
def g_coons_tf(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    uL = u_exact_tf(tf.concat([tf.zeros_like(y), y], axis=1))
    uR = u_exact_tf(tf.concat([tf.ones_like(y), y], axis=1))
    uB = u_exact_tf(tf.concat([x, tf.zeros_like(x)], axis=1))
    uT = u_exact_tf(tf.concat([x, tf.ones_like(x)], axis=1))

    u00 = u_exact_tf(tf.constant([[0.0, 0.0]], dtype=DTYPE))
    u10 = u_exact_tf(tf.constant([[1.0, 0.0]], dtype=DTYPE))
    u01 = u_exact_tf(tf.constant([[0.0, 1.0]], dtype=DTYPE))
    u11 = u_exact_tf(tf.constant([[1.0, 1.0]], dtype=DTYPE))

    term1 = (1.0 - x) * uL + x * uR
    term2 = (1.0 - y) * uB + y * uT
    corner = (1.0 - x) * (1.0 - y) * u00 + x * (1.0 - y) * u10 + (1.0 - x) * y * u01 + x * y * u11
    return term1 + term2 - corner


# -------------------------
# MLP
# -------------------------
class MLP(tf.keras.Model):
    def __init__(self, width=50, layers=5, normalize_input=False):
        super().__init__()
        self.normalize_input = bool(normalize_input)
        init = tf.keras.initializers.GlorotNormal()
        self.h = []
        for _ in range(layers - 1):
            self.h.append(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer=init))
        self.out = tf.keras.layers.Dense(1, activation=None, kernel_initializer=init)

    def call(self, x):
        z = x
        if self.normalize_input:
            # assume physical domain ~[0,1]^d, map to [-1,1]^d for conditioning
            z = 2.0 * z - 1.0
        for lyr in self.h[:-1]:
            z = lyr(z)
        z = self.h[-1](z)
        return self.out(z)


# -------------------------
# QMC sequences
# -------------------------
def _vdc_single(n: int, base: int) -> float:
    vdc, denom = 0.0, 1.0
    while n:
        n, remainder = divmod(n, base)
        denom *= base
        vdc += remainder / denom
    return vdc


def vdc_sequence(n: int, base: int = 2, start_index: int = 1) -> np.ndarray:
    n = int(n)
    start_index = int(start_index)
    out = np.empty((n,), dtype=np.float64)
    for i in range(n):
        out[i] = _vdc_single(start_index + i, base)
    return out


def halton_sequence(n: int, dim: int = 2, bases=(2, 3), start_index: int = 1) -> np.ndarray:
    n = int(n)
    dim = int(dim)
    if dim != len(bases):
        raise ValueError("bases length must match dim")
    cols = []
    for d in range(dim):
        cols.append(vdc_sequence(n, base=int(bases[d]), start_index=start_index))
    return np.stack(cols, axis=1)


def sobol_sequence(n: int, dim: int, seed: int, scramble: bool = True) -> np.ndarray:
    n = int(n)
    dim = int(dim)
    if n <= 0:
        return np.zeros((0, dim), dtype=np.float64)
    m = int(math.ceil(math.log2(max(2, n))))
    eng = sp_qmc.Sobol(d=dim, scramble=bool(scramble), seed=int(seed))
    pts = eng.random_base2(m=m)
    return pts[:n].astype(np.float64)


def qmc_shifts_2d(n_sets: int, seed: int, stride: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random((int(n_sets), 2), dtype=np.float64)


def build_ref_cloud_square(n: int, seed: int, cfg: Config, dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if cfg.QMC_KIND.lower() == "sobol":
        pts = sobol_sequence(n, dim=dim, seed=seed, scramble=cfg.QMC_SCRAMBLE)
    else:
        start = 1 + int(seed) * int(cfg.QMC_SHIFT_STRIDE)
        bases = (2, 3) if dim == 2 else (2,)
        pts = halton_sequence(n, dim=dim, bases=bases, start_index=start)
        if cfg.QMC_SCRAMBLE:
            sh = qmc_shifts_2d(1, seed=seed + 123)[0]
            pts = (pts + sh[None, :]) % 1.0
    w = np.full((n,), 1.0 / float(n), dtype=np.float64)
    return pts.astype(np.float64), w


# -------------------------
# Bubble phi for u (same as original)
# -------------------------
@tf.function
def bubble_phi(xy, bubble_scale, bubble_mode, softmin_tau, power):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    s = tf.constant(float(bubble_scale), dtype=DTYPE)
    p = tf.cast(power, DTYPE)
    tau = tf.cast(softmin_tau, DTYPE)

    def product_phi():
        base = x * (1.0 - x) * y * (1.0 - y)
        return s * tf.pow(tf.maximum(base, tf.constant(0.0, dtype=DTYPE)), p)

    def softmin_phi():
        d0 = x
        d1 = 1.0 - x
        d2 = y
        d3 = 1.0 - y
        D = tf.concat([d0, d1, d2, d3], axis=1)
        sm = -tau * (tf.reduce_logsumexp(-D / tau, axis=1, keepdims=True) - tf.math.log(tf.constant(4.0, dtype=DTYPE)))
        sm = tf.maximum(sm, 0.0)

        # IMPORTANT: enforce exact boundary vanishing for the *trial* bubble too.
        # softmin(sm) alone is >0 on most of ∂Ω (only hits 0 at the corners), which breaks the
        # hard Dirichlet condition when u = g + phi*w. We fix that by multiplying a scaled
        # product bubble factor (still smooth in the interior).
        base = x * (1.0 - x) * y * (1.0 - y)  # in [0, 1/16]
        base_fac = tf.clip_by_value(tf.constant(16.0, dtype=DTYPE) * base, 0.0, 1.0)
        return s * tf.pow(sm, p) * base_fac

    return tf.switch_case(bubble_mode, branch_fns=[product_phi, softmin_phi], default=softmin_phi)


# For test functions we use a *simple* exact-zero-on-boundary bubble b0
@tf.function(reduce_retracing=True)
def bubble0_and_grad(xy, scale=1.0, mode=0, softmin_tau=1e-3, power=1.0):
    """
    Boundary bubble used for test functions v_j:
      b(x,y) = 0 on ∂Ω.

    Backward-compatible:
      - bubble0_and_grad(xy) -> exact product bubble (scale=1, p=1)
      - bubble0_and_grad(xy, scale=..., mode=..., softmin_tau=..., power=...)
        matches bubble_phi's options but also returns ∇b.

    Returns:
      b:  (N,1)
      db: (N,2) = [∂b/∂x, ∂b/∂y]
    """
    xy = tf.convert_to_tensor(xy, dtype=DTYPE)
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    s = tf.cast(scale, DTYPE)
    p = tf.cast(power, DTYPE)
    tau = tf.cast(softmin_tau, DTYPE)
    mode_i = tf.cast(mode, tf.int32)

    # Exact product bubble base and its gradient
    base = x * (1.0 - x) * y * (1.0 - y)
    dbase_dx = (1.0 - 2.0 * x) * y * (1.0 - y)
    dbase_dy = (1.0 - 2.0 * y) * x * (1.0 - x)

    def product_branch():
        base_pos = tf.maximum(base, tf.constant(0.0, dtype=DTYPE))
        fac = tf.where(base_pos > 0.0,
                       tf.pow(base_pos, tf.maximum(p - 1.0, 0.0)),
                       tf.zeros_like(base_pos))
        b = s * tf.pow(base_pos, p)
        dbx = s * p * fac * dbase_dx
        dby = s * p * fac * dbase_dy
        return b, tf.concat([dbx, dby], axis=1)

    def softmin_branch():
        # Smooth min of distances to the boundary
        d0 = x
        d1 = 1.0 - x
        d2 = y
        d3 = 1.0 - y
        D = tf.concat([d0, d1, d2, d3], axis=1)  # (N,4)

        lse = tf.reduce_logsumexp(-D / tau, axis=1, keepdims=True)
        sm = -tau * (lse - tf.math.log(tf.constant(4.0, dtype=DTYPE)))
        sm = tf.maximum(sm, 0.0)

        # dsm/dD_j = softmax(-D/tau)_j
        w = tf.nn.softmax(-D / tau, axis=1)
        dsm_dx = w[:, 0:1] - w[:, 1:2]
        dsm_dy = w[:, 2:3] - w[:, 3:4]

        fac = tf.where(sm > 0.0,
                       tf.pow(sm, tf.maximum(p - 1.0, 0.0)),
                       tf.zeros_like(sm))
        b = s * tf.pow(sm, p)
        dbx = s * p * fac * dsm_dx
        dby = s * p * fac * dsm_dy

        z = tf.zeros_like(dbx)
        dbx = tf.where(sm > 0.0, dbx, z)
        dby = tf.where(sm > 0.0, dby, z)
        b = tf.where(sm > 0.0, b, tf.zeros_like(b))
        return b, tf.concat([dbx, dby], axis=1)

    return tf.switch_case(mode_i, branch_fns=[product_branch, softmin_branch], default=softmin_branch)


# -------------------------
# Operator B: u = phi*w + g, and grad u
# lift_mode: 0=exact, 1=gnet, 2=coons
# -------------------------
@tf.function
def Bu_and_grad_lift(unet, gnet, lift_mode, xy, bubble_scale, bubble_mode, softmin_tau, bubble_power):
    with tf.GradientTape() as tape:
        tape.watch(xy)
        w = unet(xy)
        phi = bubble_phi(xy, bubble_scale, bubble_mode, softmin_tau, bubble_power)

        def g0():
            return u_exact_tf(xy)

        def g1():
            # Lift via gnet, but *clamp* boundary values to the exact Dirichlet data.
            # This keeps the "gnet extension" idea meshfree while eliminating a very real
            # floor on achievable H1 caused by imperfect boundary fit.
            if gnet is None:
                return u_exact_tf(xy)
            g_pred = gnet(xy)
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            eps = tf.cast(1e-12, DTYPE)
            on_b = tf.logical_or(
                tf.logical_or(x <= eps, x >= (tf.cast(1.0, DTYPE) - eps)),
                tf.logical_or(y <= eps, y >= (tf.cast(1.0, DTYPE) - eps)),
            )
            g_true = u_exact_tf(xy)
            return tf.where(on_b, g_true, g_pred)

        def g2():
            return g_coons_tf(xy)

        g = tf.switch_case(lift_mode, branch_fns=[g0, g1, g2], default=g0)
        u = phi * w + g

    grad_u = tape.gradient(u, xy)
    return u, grad_u


@tf.function(reduce_retracing=True)
def grad_u_tf(unet, gnet, lift_mode, xy, bubble_scale, bubble_mode, softmin_tau, bubble_power):
    _, gu = Bu_and_grad_lift(unet, gnet, lift_mode, xy, bubble_scale, bubble_mode, softmin_tau, bubble_power)
    return gu


@tf.function(reduce_retracing=True)
def _reflect01_tf(z):
    """Reflect z into [0,1] with period 2 and reflection (Neumann-like for FD stencils)."""
    two = tf.cast(2.0, z.dtype)
    one = tf.cast(1.0, z.dtype)
    z2 = tf.math.floormod(z, two)  # [0,2)
    return tf.where(z2 > one, two - z2, z2)


@tf.function(reduce_retracing=True)
def reflect_unit_square_tf(xy):
    x = _reflect01_tf(xy[:, 0:1])
    y = _reflect01_tf(xy[:, 1:2])
    return tf.concat([x, y], axis=1)


def laplace_u_tf(unet, gnet, lift_mode, xy, h, bubble_scale, bubble_mode, softmin_tau, bubble_power):
    """
    Finite-difference Laplacian of the *trainable* part of u (for adaptivity indicator).
    Vectorized: evaluates the network once on a stacked (5N,2) tensor instead of 5 separate calls.
    """
    dtype = xy.dtype
    h = tf.cast(h, dtype)
    dx = tf.stack([h, tf.cast(0.0, dtype)], axis=0)
    dy = tf.stack([tf.cast(0.0, dtype), h], axis=0)

    xy0 = xy
    xy_all = tf.concat([xy0, xy0 + dx, xy0 - dx, xy0 + dy, xy0 - dy], axis=0)
    xy_all = reflect_unit_square_tf(xy_all)

    w_all = unet(xy_all)
    phi_all = bubble_phi(xy_all, bubble_scale, bubble_mode, softmin_tau, bubble_power)

    # IMPORTANT: for lift_mode==0 (exact lift), indicator should be based on the NN-part only (phi*w),
    # not including the exact lift term, to avoid polluting adaptivity with the known analytic component.
    def _u_wonly():
        return phi_all * w_all

    def _u_full():
        def g_exact():
            return u_exact_tf(xy_all)

        def g_gnet():
            if gnet is None:
                return u_exact_tf(xy_all)
            return gnet(xy_all)

        def g_coons():
            return g_coons_tf(xy_all)

        g_all = tf.switch_case(
            lift_mode,
            branch_fns={
                0: g_exact,
                1: g_gnet,
                2: g_coons,
            },
            default=g_exact,
        )
        return phi_all * w_all + g_all

    u_all = tf.cond(tf.equal(lift_mode, 0), _u_wonly, _u_full)

    u0, uxp, uxm, uyp, uym = tf.split(u_all, num_or_size_splits=5, axis=0)
    inv_h2 = tf.cast(1.0, dtype) / (h * h)
    lap = (uxp - tf.cast(2.0, dtype) * u0 + uxm) * inv_h2 + (uyp - tf.cast(2.0, dtype) * u0 + uym) * inv_h2
    return lap


@tf.function(reduce_retracing=True)
def kernel_gauss_k_gradxy_matmul(xy, xi, ell, cutoff_s=None):
    """
    xy: (N,2), xi: (M,2), ell: (M,1)
    returns k:(N,M), gkx:(N,M), gky:(N,M)
    """
    cutoff_s = _get_cutoff_s_default(cutoff_s)
    # r^2 = ||x||^2 + ||a||^2 - 2 x·a^T
    x2 = tf.reduce_sum(tf.square(xy), axis=1, keepdims=True)  # (N,1)
    a2 = tf.reduce_sum(tf.square(xi), axis=1, keepdims=True)  # (M,1)
    xa = tf.linalg.matmul(xy, xi, transpose_b=True)  # (N,M)
    r2 = x2 + tf.transpose(a2) - 2.0 * xa  # (N,M)

    ell2 = tf.transpose(tf.square(ell))  # (1,M)
    s = r2 / (ell2 + tf.constant(1e-30, dtype=DTYPE))  # (N,M)
    if cutoff_s is not None:
        cutoff_s = tf.cast(cutoff_s, DTYPE)
        mask = tf.cast(s <= cutoff_s, DTYPE)
    else:
        mask = None

    k = tf.exp(-s)
    if mask is not None:
        k = k * mask

    # grad w.r.t. x: ∂k/∂x = k * (-2/ell^2) * (x - a)
    inv_ell2 = 1.0 / (ell2 + tf.constant(1e-30, dtype=DTYPE))  # (1,M)

    # (N,1) - (1,M) -> (N,M)
    dx = xy[:, 0:1] - tf.transpose(xi[:, 0:1])
    dy = xy[:, 1:2] - tf.transpose(xi[:, 1:2])

    scale = -2.0 * inv_ell2  # (1,M)
    gkx = k * scale * dx
    gky = k * scale * dy
    return k, gkx, gky


@tf.function(reduce_retracing=True)
def _gradv_from_precomputed_single(xy, b0, db0, xi, ell, cutoff_s=None):
    """
    Computes ∇v_j(x) for a batch of points and anchors using precomputed bubble:
      v_j(x) = b0(x) * k(||x-a_j||; ell_j)

    Returns:
      gradvx: (N,M), gradvy: (N,M)
    """
    k, gkx, gky = kernel_k_gradxy_matmul(xy, xi, ell, cutoff_s=cutoff_s)
    dbx = db0[:, 0:1]
    dby = db0[:, 1:2]
    gradvx = k * dbx + b0 * gkx
    gradvy = k * dby + b0 * gky
    return gradvx, gradvy


def _gradv_from_precomputed_batch(cfg, K, Kx, Ky, Kxx, Kyy, Kxy, b0, db0):
    """
    Batch version used by projection-based novelty/confidence scoring.

    Inputs (all TensorFlow tensors):
      K, Kx, Ky, Kxx, Kyy, Kxy: (N,M) kernel value and derivatives w.r.t. x,y
      b0: (N,1) bubble value, db0: (N,2) bubble gradient

    Returns:
      gvx: (N,M), gvy: (N,M), divv: (N,M) approximate Laplacian of v (bubble Hessian ignored).
    """
    dbx = db0[:, 0:1]
    dby = db0[:, 1:2]

    gvx = K * dbx + b0 * Kx
    gvy = K * dby + b0 * Ky

    # Approximate Laplacian(div grad) of v, ignoring second-derivatives of bubble (not available).
    divv = (2.0 * dbx * Kx + b0 * Kxx) + (2.0 * dby * Ky + b0 * Kyy)
    return gvx, gvy, divv


def gradv_from_precomputed(*args, **kwargs):
    """Dispatch wrapper: supports both single-anchor and batch precompute call patterns."""
    if args and hasattr(args[0], "KERNEL_KIND"):
        return _gradv_from_precomputed_batch(*args, **kwargs)
    return _gradv_from_precomputed_single(*args, **kwargs)


@tf.function(reduce_retracing=True)
def energy_G_block(xy, b0, db0, xi, ell, cutoff_s=None):
    """
    Returns sum_i ||∇v(x_i)||^2 over points i (no quadrature weights applied).
    Output shape: (M,)
    """
    gvx, gvy = gradv_from_precomputed(xy, b0, db0, xi, ell, cutoff_s=cutoff_s)
    gv2 = tf.square(gvx) + tf.square(gvy)  # (N,M)
    return tf.reduce_sum(gv2, axis=0)  # (M,)


@tf.function(reduce_retracing=True)
def R_block(unet, gnet, lift_mode_tf, xy, b0, db0, xi, ell,
            bubble_scale, bubble_mode, softmin_tau, bubble_power,
            cutoff_s=None):
    """
    Returns sum_i (∇u · ∇v_j)(x_i) over points i for each anchor j.
    Output shape: (M,)
    """
    gu = grad_u_tf(unet, gnet, lift_mode_tf, xy, bubble_scale, bubble_mode, softmin_tau, bubble_power)  # (N,2)
    gvx, gvy = gradv_from_precomputed(xy, b0, db0, xi, ell, cutoff_s=cutoff_s)  # (N,M)
    dot = gu[:, 0:1] * gvx + gu[:, 1:2] * gvy  # (N,M)
    return tf.reduce_sum(dot, axis=0)  # (M,)


def _get_cutoff_s_default(cutoff_s):
    """If cutoff_s is None, use global KERNEL_CUTOFF_FACTOR (squared) when >0."""
    if cutoff_s is not None:
        return cutoff_s
    cf = float(globals().get("KERNEL_CUTOFF_FACTOR", 0.0))
    if cf and cf > 0.0:
        return tf.constant(cf * cf, dtype=DTYPE)
    return None


@tf.function(reduce_retracing=True)
def kernel_wendland_c2_k_gradxy_matmul(xy, xi, ell, cutoff_s=None):
    """
    Compactly-supported Wendland C2 RBF (polynomial, truly sparse):
        phi(rho) = (1-rho)^4 * (4*rho + 1),  rho = r / (C*ell), 0<=rho<=1 else 0.
    cutoff_s uses the same CLI C via cutoff_s=C^2 (keeps the rest of the code unchanged).
    Returns:
      k:(N,M), gkx:(N,M), gky:(N,M)
    """
    cutoff_s = _get_cutoff_s_default(cutoff_s)
    xy = tf.convert_to_tensor(xy, dtype=DTYPE)
    xi = tf.convert_to_tensor(xi, dtype=DTYPE)
    ell = tf.reshape(tf.convert_to_tensor(ell, dtype=DTYPE), (1, -1))  # (1,M)

    diff = xy[:, None, :] - xi[None, :, :]  # (N,M,2)
    r2 = tf.reduce_sum(tf.square(diff), axis=2)  # (N,M)
    r = tf.sqrt(tf.maximum(r2, tf.constant(0.0, dtype=DTYPE)))

    cf = tf.sqrt(tf.cast(cutoff_s, DTYPE)) if (cutoff_s is not None) else tf.constant(1.0, dtype=DTYPE)
    R = ell * cf + tf.cast(1e-30, DTYPE)  # support radius per-anchor (1,M)
    rho = r / R  # (N,M)
    t = tf.maximum(tf.constant(1.0, dtype=DTYPE) - rho, tf.constant(0.0, dtype=DTYPE))

    # k = t^4 (4rho+1) on support
    k = tf.pow(t, 4) * (tf.constant(4.0, dtype=DTYPE) * rho + tf.constant(1.0, dtype=DTYPE))
    mask = tf.cast(rho <= tf.constant(1.0, dtype=DTYPE), DTYPE)
    k = k * mask

    # dphi/dr = (-20 rho (1-rho)^3) * (1/R)
    dphi_dr = (-20.0) * rho * tf.pow(t, 3) / R
    dphi_dr = dphi_dr * mask

    inv_r = tf.where(r > tf.cast(1e-30, DTYPE), tf.math.reciprocal(r), tf.zeros_like(r))
    gkx = dphi_dr * (diff[:, :, 0] * inv_r)
    gky = dphi_dr * (diff[:, :, 1] * inv_r)
    return k, gkx, gky


def kernel_k_gradxy_matmul(xy, xi, ell, cutoff_s=None):
    """Dispatch to selected kernel family (set once in main)."""
    if str(globals().get("KERNEL_KIND", "gauss")) == "wendland_c2":
        return kernel_wendland_c2_k_gradxy_matmul(xy, xi, ell, cutoff_s=cutoff_s)
    return kernel_gauss_k_gradxy_matmul(xy, xi, ell, cutoff_s=cutoff_s)


def kernel_matrix_and_grad(xy, xi, ell, cutoff_s=None):
    """Return kernel matrix and its gradient: gradk:(N,M,2)."""
    k, gkx, gky = kernel_k_gradxy_matmul(xy, xi, ell, cutoff_s=cutoff_s)
    return k, tf.stack([gkx, gky], axis=2)


# Back-compat (was referenced in older code paths)
def kernel_gauss_and_grad(xy, xi, ell, cutoff_s=None):
    return kernel_matrix_and_grad(xy, xi, ell, cutoff_s=cutoff_s)


def kernel_eval_precompute(*args, **kwargs):
    """
    Helper used by the projection-based growth score.

    Supports two call patterns:

      (A) kernel_eval_precompute(xy, xi, ell, bubble_scale, bubble_mode, softmin_tau, bubble_power, cutoff_s)
          -> (b0, db0, k, gkx, gky)

      (B) kernel_eval_precompute(cfg, xy, Xi, ell_loc_tf)
          -> (K, Kx, Ky, Kxx, Kyy, Kxy)  where all are (N,M)

    Notes:
      - For Wendland C2 we return K, Kx, Ky and second-derivatives; for other non-Gaussian kernels second-derivatives are zero.
      - cutoff uses cfg.KERNEL_CUTOFF (factor) converted to cutoff_s = factor^2.
    """
    # Pattern (B): batch precompute for a cloud of points vs many anchors.
    if len(args) == 4 and hasattr(args[0], "KERNEL_KIND"):
        cfg, xy, Xi, ell_loc_tf = args
        xy = tf.convert_to_tensor(xy)
        Xi = tf.convert_to_tensor(Xi, dtype=xy.dtype)
        # IMPORTANT: ell_loc_tf is typically shaped (M,1). For broadcasting against
        # tensors shaped (N,M) we must use (M,) so TensorFlow can broadcast as (1,M).
        ell = tf.cast(ell_loc_tf, xy.dtype)
        ell = tf.reshape(ell, [-1])  # (M,)

        cutoff_factor = float(getattr(cfg, "KERNEL_CUTOFF", 0.0))
        cutoff_s = tf.constant(cutoff_factor * cutoff_factor, dtype=DTYPE) if (
                    cutoff_factor and cutoff_factor > 0.0) else None

        kind = str(getattr(cfg, "KERNEL_KIND", "gauss"))
        if kind == "gauss":
            # Shapes: xy (N,2), Xi (M,2) -> outputs (N,M)
            diff = xy[:, None, :] - Xi[None, :, :]
            dx = diff[:, :, 0]
            dy = diff[:, :, 1]
            r2 = dx * dx + dy * dy
            ell2 = tf.maximum(ell * ell, tf.cast(1e-30, xy.dtype))  # (M,)
            inv_ell2 = 1.0 / ell2
            inv_ell4 = inv_ell2 * inv_ell2

            s = r2 * inv_ell2
            K = tf.exp(-s)

            if cutoff_s is not None:
                mask = tf.cast(s <= tf.cast(cutoff_s, xy.dtype), xy.dtype)
                K = K * mask

            Kx = (-2.0 * inv_ell2) * dx * K
            Ky = (-2.0 * inv_ell2) * dy * K

            Kxx = ((4.0 * dx * dx) * inv_ell4 - 2.0 * inv_ell2) * K
            Kyy = ((4.0 * dy * dy) * inv_ell4 - 2.0 * inv_ell2) * K
            Kxy = (4.0 * dx * dy) * inv_ell4 * K
            return K, Kx, Ky, Kxx, Kyy, Kxy

        if kind == "wendland_c2":
            # Shapes: xy (N,2), Xi (M,2) -> outputs (N,M)
            diff = xy[:, None, :] - Xi[None, :, :]
            dx = diff[:, :, 0]
            dy = diff[:, :, 1]
            r2 = dx * dx + dy * dy
            r = tf.sqrt(tf.maximum(r2, tf.cast(0.0, xy.dtype)))

            cf = tf.sqrt(tf.cast(cutoff_s, xy.dtype)) if (cutoff_s is not None) else tf.cast(1.0, xy.dtype)
            R = (ell[None, :] * cf) + tf.cast(1e-30, xy.dtype)
            rho = r / R
            t = tf.maximum(tf.cast(1.0, xy.dtype) - rho, tf.cast(0.0, xy.dtype))

            K = tf.pow(t, 4) * (tf.cast(4.0, xy.dtype) * rho + tf.cast(1.0, xy.dtype))
            mask = tf.cast(rho <= tf.cast(1.0, xy.dtype), xy.dtype)
            K = K * mask

            # k'(r) = dphi/dr
            k_r = (-20.0) * rho * tf.pow(t, 3) / R
            k_r = k_r * mask

            inv_r = tf.where(r > tf.cast(1e-30, xy.dtype), tf.math.reciprocal(r), tf.zeros_like(r))
            Kx = k_r * (dx * inv_r)
            Ky = k_r * (dy * inv_r)

            # k''(r) = d2phi/dr2 = (d2phi/drho2) / R^2,  d2phi/drho2 = -20*(1-rho)^2*(1-4*rho)
            k_rr = (-20.0) * tf.pow(t, 2) * (tf.cast(1.0, xy.dtype) - tf.cast(4.0, xy.dtype) * rho) / (R * R)
            k_rr = k_rr * mask

            # Hessian for radial kernels:
            #   ∂_ij k = k'' * (x_i x_j / r^2) + k' * (δ_ij / r - x_i x_j / r^3)
            eps_r2 = tf.cast(1e-30, xy.dtype)
            inv_r2 = tf.where(r2 > eps_r2, tf.math.reciprocal(r2), tf.zeros_like(r2))
            inv_r3 = inv_r2 * inv_r

            Kxx = k_rr * (dx * dx * inv_r2) + k_r * (dy * dy * inv_r3)
            Kyy = k_rr * (dy * dy * inv_r2) + k_r * (dx * dx * inv_r3)
            Kxy = k_rr * (dx * dy * inv_r2) - k_r * (dx * dy * inv_r3)

            # Stable r->0 limit: Kxx(0)=Kyy(0)=k''(0), Kxy(0)=0
            small = r <= (tf.cast(1e-12, xy.dtype) * R)
            Kxx = tf.where(small, k_rr, Kxx)
            Kyy = tf.where(small, k_rr, Kyy)
            Kxy = tf.where(small, tf.zeros_like(Kxy), Kxy)

            return K, Kx, Ky, Kxx, Kyy, Kxy

        # Fallback: no Hessian formulas available here.
        # Use a single representative ell for non-Gaussian kernels.
        ell0 = tf.reduce_mean(ell)
        K, gradk = kernel_matrix_and_grad(xy, Xi, ell0, cutoff_s=cutoff_s)
        gx = gradk[:, :, 0]
        gy = gradk[:, :, 1]
        Z = tf.zeros_like(K)
        return K, gx, gy, Z, Z, Z

    # Pattern (A): single-anchor precompute, used in some optional helpers.
    if len(args) == 8:
        xy, xi, ell, bubble_scale, bubble_mode, softmin_tau, bubble_power, cutoff_s = args
        b0, db0 = bubble0_and_grad(xy, scale=bubble_scale, mode=bubble_mode,
                                   softmin_tau=softmin_tau, power=bubble_power)
        k, gkx, gky = kernel_k_gradxy_matmul(xy, xi, ell, cutoff_s=cutoff_s)
        return b0, db0, k, gkx, gky

    raise TypeError(f"kernel_eval_precompute: unsupported signature (got {len(args)} args)")


def edge_kernel_k_and_grad(diff, ell_edge, cutoff_s=None):
    """
    Edge-aligned kernel and gradient:
      diff:(E,2)=x-a, ell_edge:(E,1)
    Returns:
      k:(E,1), gradk:(E,2)
    """
    cutoff_s = _get_cutoff_s_default(cutoff_s)

    if str(globals().get("KERNEL_KIND", "gauss")) == "wendland_c2":
        r2 = tf.reduce_sum(tf.square(diff), axis=1, keepdims=True)  # (E,1)
        r = tf.sqrt(tf.maximum(r2, tf.constant(0.0, dtype=DTYPE)))
        cf = tf.sqrt(tf.cast(cutoff_s, DTYPE)) if (cutoff_s is not None) else tf.constant(1.0, dtype=DTYPE)
        R = ell_edge * cf + tf.cast(1e-30, DTYPE)
        rho = r / R
        t = tf.maximum(tf.constant(1.0, dtype=DTYPE) - rho, tf.constant(0.0, dtype=DTYPE))
        k = tf.pow(t, 4) * (tf.constant(4.0, dtype=DTYPE) * rho + tf.constant(1.0, dtype=DTYPE))
        mask = tf.cast(rho <= tf.constant(1.0, dtype=DTYPE), DTYPE)
        k = k * mask
        dphi_dr = (-20.0) * rho * tf.pow(t, 3) / R
        dphi_dr = dphi_dr * mask
        inv_r = tf.where(r > tf.cast(1e-30, DTYPE), tf.math.reciprocal(r), tf.zeros_like(r))
        gradk = dphi_dr * diff * inv_r
        return k, gradk

    # Gaussian
    ell2 = tf.square(ell_edge) + tf.cast(1e-30, DTYPE)
    r2 = tf.reduce_sum(tf.square(diff), axis=1, keepdims=True)
    s = r2 / ell2
    if cutoff_s is not None:
        mask = tf.cast(s <= tf.cast(cutoff_s, DTYPE), DTYPE)
        k = tf.exp(-s) * mask
    else:
        k = tf.exp(-s)
    gradk = k * (-2.0 / ell2) * diff
    return k, gradk


def compute_G_full_cached(Xq_tf, b0_all_tf, db0_all_tf,
                          anchors: np.ndarray, ell: np.ndarray,
                          cfg: Config,
                          cutoff_factor: float = 0.0,
                          prev_G: Optional[np.ndarray] = None,
                          M_old: int = 0) -> np.ndarray:
    """
    Compute per-anchor energy G_j = mean_q |∇v_j|^2 on the global quadrature set.

    v15 upgrade: incremental cache support.
      - If prev_G is provided with shape (M_old,), we only compute G for anchors[M_old:],
        and return np.concatenate([prev_G, G_new]).
      - Otherwise we compute all anchors as before.

    Note: This is safe under the "frozen per-anchor ell_j" assumption (no edits/removals).
    If you remove/reorder anchors or change ell for existing anchors, you MUST discard prev_G.
    """
    anchors = np.asarray(anchors, np.float64)
    ell = np.asarray(ell, np.float64)

    anchors_tf = tf.constant(anchors, dtype=DTYPE)
    ell_tf = tf.constant(ell, dtype=DTYPE)

    Nq = int(Xq_tf.shape[0])
    M = int(anchors_tf.shape[0])

    pb = int(getattr(cfg, "EVAL_POINT_BATCH", 8192))
    ab = int(getattr(cfg, "EVAL_ANCHOR_BATCH", 64))

    cutoff_s = None
    if cutoff_factor and float(cutoff_factor) > 0:
        cutoff_s = tf.constant(float(cutoff_factor) ** 2, dtype=DTYPE)

    invNq = tf.cast(1.0 / float(Nq), DTYPE)

    blocks = []
    start = 0

    # Incremental prefix (prev_G)
    if prev_G is not None:
        try:
            prev = np.asarray(prev_G, np.float64).reshape(-1)
            M_old_i = int(M_old)
            if M_old_i < 0:
                M_old_i = 0
            if M_old_i > M:
                M_old_i = M
            if 0 < M_old_i == prev.shape[0]:
                blocks.append(tf.constant(prev, dtype=DTYPE))
                start = M_old_i
        except Exception:
            # Fall back to full recompute on any mismatch/format issue.
            blocks = []
            start = 0

    for s in range(start, M, ab):
        e = min(s + ab, M)
        xi = anchors_tf[s:e]
        el = ell_tf[s:e]

        accG = tf.zeros((e - s,), dtype=DTYPE)
        for p in range(0, Nq, pb):
            q = min(p + pb, Nq)
            xy = Xq_tf[p:q]
            b0 = b0_all_tf[p:q]
            db0 = db0_all_tf[p:q]
            accG = accG + energy_G_block(xy, b0, db0, xi, el, cutoff_s=cutoff_s)

        G = accG * invNq
        blocks.append(G)

    if len(blocks) == 0:
        return np.zeros((0,), dtype=np.float64)
    return tf.concat(blocks, axis=0).numpy().astype(np.float64)


def _split_pruned_dense(anchor_ids: np.ndarray, pruner: KDTreePruner):
    anchor_ids = np.asarray(anchor_ids, dtype=np.int32).reshape(-1)
    pruned = []
    dense = []
    for aid in anchor_ids:
        if pruner.idx_lists[int(aid)] is None:
            dense.append(int(aid))
        else:
            pruned.append(int(aid))
    return np.asarray(pruned, np.int32), np.asarray(dense, np.int32)


def compute_G_for_anchor_ids_pruned(
        Xq_tf, b0_all_tf, db0_all_tf,
        anchors: np.ndarray, ell: np.ndarray,
        anchor_ids: np.ndarray,
        pruner: KDTreePruner,
        cfg: Config,
        cutoff_factor: float = 4.0,
) -> np.ndarray:
    """Compute G_j for *pruned* anchors only (anchor_ids must all be pruned)."""
    anchor_ids = np.asarray(anchor_ids, dtype=np.int32).reshape(-1)
    if anchor_ids.size == 0:
        return np.zeros((0,), dtype=np.float64)

    Nq = int(Xq_tf.shape[0])
    invNq = tf.cast(1.0 / float(Nq), DTYPE)
    eb = int(getattr(cfg, "EVAL_EDGE_BATCH", 2048))

    # Build edge list using all local points.
    idx_flat, seg_ids = pruner.build_edges_full(anchor_ids)
    if idx_flat.size == 0:
        return np.zeros((anchor_ids.shape[0],), dtype=np.float64)

    # Per-batch anchor arrays (seg_ids indexes these)
    a_batch = tf.constant(np.asarray(anchors, np.float64)[anchor_ids], dtype=DTYPE)
    e_batch = tf.constant(np.asarray(ell, np.float64)[anchor_ids].reshape(-1, 1), dtype=DTYPE)

    # Edge tensors
    idx_tf = tf.constant(idx_flat, dtype=tf.int32)
    seg_tf = tf.constant(seg_ids, dtype=tf.int32)

    cutoff_s = None
    if cutoff_factor and float(cutoff_factor) > 0:
        cutoff_s = tf.constant(float(cutoff_factor) ** 2, dtype=DTYPE)

    # Chunked accumulation to avoid OOM when the total edge count is large.
    A = int(anchor_ids.shape[0])
    E = int(idx_flat.shape[0])
    sums = tf.zeros((A,), dtype=DTYPE)

    for t in range(0, E, eb):
        idx_c = idx_tf[t:t + eb]
        seg_c = seg_tf[t:t + eb]

        xy = tf.gather(Xq_tf, idx_c)
        b0 = tf.gather(b0_all_tf, idx_c)
        db0 = tf.gather(db0_all_tf, idx_c)

        a_edge = tf.gather(a_batch, seg_c)
        e_edge = tf.gather(e_batch, seg_c)

        gv = gradv_edges_from_precomputed(xy, b0, db0, a_edge, e_edge, cutoff_s=cutoff_s)  # (Ec,2)
        gv2 = tf.reduce_sum(tf.square(gv), axis=1)  # (Ec,)

        sums = sums + tf.math.unsorted_segment_sum(gv2, seg_c, num_segments=A)  # (A,)

    G = (sums * invNq).numpy().astype(np.float64)
    return G


def compute_G_for_anchor_ids_mixed(
        Xq_tf, b0_all_tf, db0_all_tf,
        anchors: np.ndarray, ell: np.ndarray,
        anchor_ids: np.ndarray,
        pruner: KDTreePruner,
        cfg: Config,
        cutoff_factor: float = 4.0,
) -> np.ndarray:
    """
    Compute G_j for arbitrary anchor_ids. Pruned anchors use KDTree local edges.
    Dense anchors fall back to the original dense matmul kernel path (stable for large neighborhoods).
    """
    anchor_ids = np.asarray(anchor_ids, dtype=np.int32).reshape(-1)
    out = np.zeros((anchor_ids.shape[0],), dtype=np.float64)
    if anchor_ids.size == 0:
        return out

    pruned_ids, dense_ids = _split_pruned_dense(anchor_ids, pruner)

    # Pruned part
    if pruned_ids.size > 0:
        Gp = compute_G_for_anchor_ids_pruned(
            Xq_tf, b0_all_tf, db0_all_tf,
            anchors, ell,
            pruned_ids, pruner, cfg,
            cutoff_factor=cutoff_factor
        )
        # scatter back
        pos = {int(a): i for i, a in enumerate(anchor_ids.tolist())}
        for a, g in zip(pruned_ids.tolist(), Gp.tolist()):
            out[pos[int(a)]] = float(g)

    # Dense part: use the existing dense evaluator on the subset
    if dense_ids.size > 0:
        anchors_sub = np.asarray(anchors, np.float64)[dense_ids]
        ell_sub = np.asarray(ell, np.float64)[dense_ids]
        Gd = compute_G_full_cached(
            Xq_tf, b0_all_tf, db0_all_tf,
            anchors_sub, ell_sub, cfg,
            cutoff_factor=cutoff_factor
        ).reshape(-1)
        pos = {int(a): i for i, a in enumerate(anchor_ids.tolist())}
        for a, g in zip(dense_ids.tolist(), Gd.tolist()):
            out[pos[int(a)]] = float(g)

    return out


def compute_G_full_mixed(
        Xq_tf, b0_all_tf, db0_all_tf,
        anchors: np.ndarray, ell: np.ndarray,
        pruner: KDTreePruner,
        cfg: Config,
        cutoff_factor: float = 4.0,
) -> np.ndarray:
    """Compute G for all anchors using mixed pruned/dense logic."""
    M = int(np.asarray(anchors).shape[0])
    ids = np.arange(M, dtype=np.int32)
    return compute_G_for_anchor_ids_mixed(
        Xq_tf, b0_all_tf, db0_all_tf,
        anchors, ell,
        ids, pruner, cfg,
        cutoff_factor=cutoff_factor
    )


def eval_full_cached(unet, gnet, lift_mode: int,
                     Xq_tf, b0_all_tf, db0_all_tf,
                     anchors: np.ndarray, ell: np.ndarray, aw: np.ndarray,
                     G_full: np.ndarray,
                     cfg: Config,
                     bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                     cutoff_factor: float = 4.0) -> Dict[str, Any]:
    """
    Full evaluation with point×anchor batching; reuses cached G_full (energy).
    Key improvement: avoids per-batch .numpy() syncs; converts to NumPy only once at the end.
    """
    lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)
    Nq = int(Xq_tf.shape[0])
    M = int(anchors.shape[0])
    pb = int(getattr(cfg, "EVAL_POINT_BATCH", 8192))
    ab = int(getattr(cfg, "EVAL_ANCHOR_BATCH", 64))

    cutoff_s = None
    if cutoff_factor and float(cutoff_factor) > 0:
        cutoff_s = tf.constant(float(cutoff_factor) ** 2, dtype=DTYPE)

    anchors_tf = tf.constant(anchors, dtype=DTYPE)
    ell_tf = tf.constant(ell, dtype=DTYPE)
    aw_tf = tf.constant(aw, dtype=DTYPE)

    use_energy = (int(whiten_mode_tf.numpy()) == 0)  # one sync, outside loops
    eps = tf.cast(float(cfg.WHITEN_EPS), DTYPE)
    if use_energy:
        G_tf = tf.constant(G_full.reshape(-1), dtype=DTYPE)
        denom_tf = tf.sqrt(tf.maximum(G_tf, tf.cast(0.0, DTYPE)) + eps)[:, None]  # (M,1)
        eps_floor = tf.cast(1e-8, DTYPE)
        denom_tf = tf.maximum(denom_tf, eps_floor)
    else:
        denom_tf = tf.ones((M, 1), dtype=DTYPE)

    loss_sum = tf.zeros([], dtype=DTYPE)
    M0_base = tf.cast(float(getattr(cfg, 'ANCHOR_INIT', 1)), DTYPE)
    rabs_blocks = []

    invNq = tf.cast(1.0 / float(Nq), DTYPE)

    for s in range(0, M, ab):
        e = min(s + ab, M)
        xi = anchors_tf[s:e]
        el = ell_tf[s:e]
        aww = aw_tf[s:e]
        den = denom_tf[s:e]

        # accumulate sum over points, then divide by Nq
        accR = tf.zeros((e - s,), dtype=DTYPE)
        for p in range(0, Nq, pb):
            q = min(p + pb, Nq)
            xy = Xq_tf[p:q]
            b0 = b0_all_tf[p:q]
            db0 = db0_all_tf[p:q]
            accR = accR + R_block(
                unet, gnet, lift_mode_tf, xy, b0, db0, xi, el,
                bubble_scale=cfg.BUBBLE_SCALE,
                bubble_mode=bubble_mode_tf,
                softmin_tau=softmin_tau_tf,
                bubble_power=bubble_power_tf,
                cutoff_s=cutoff_s
            )

        R = accR * invNq  # (ab,)
        r = R[:, None] / den  # (ab,1)

        # Nested loss: sum over anchors (no 1/M dilution); optionally normalized by M0 for scale.
        Lb = tf.reduce_sum(aww * tf.square(r))
        loss_sum = loss_sum + Lb
        rabs_blocks.append(tf.abs(R))

    # normalize by initial anchor count to keep a stable loss scale across refinement
    w_sum = tf.reduce_sum(tf.reshape(aw_tf, (-1,)))
    loss_mean = loss_sum / tf.maximum(tf.cast(1.0, DTYPE), tf.cast(w_sum, DTYPE))
    Rabs_tf = tf.concat(rabs_blocks, axis=0)

    G_out = (np.asarray(G_full).reshape(-1) if G_full is not None else None)
    return {"loss": float(loss_mean.numpy()), "Rabs": Rabs_tf.numpy().astype(np.float64), "G": G_out}


def eval_full_cached_mixed(unet, gnet, lift_mode: int,
                           Xq_tf, b0_all_tf, db0_all_tf,
                           anchors: np.ndarray, ell: np.ndarray, aw: np.ndarray,
                           G_full: np.ndarray,
                           cfg: Config,
                           bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                           pruner: KDTreePruner,
                           cutoff_factor: float = 4.0) -> Dict[str, Any]:
    """
    Full evaluation using KDTree-pruned local quadrature subsets where possible.
    Dense anchors fall back to the original dense batching kernels.

    v15 upgrade: avoid per-block TF->NumPy syncs for Rabs.
      - We keep Rabs as a TF variable and update via scatter, then convert once at the end.
    """
    lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)
    Nq = int(Xq_tf.shape[0])
    M = int(anchors.shape[0])
    ab = int(getattr(cfg, "EVAL_ANCHOR_BATCH", 64))
    pb = int(getattr(cfg, "EVAL_POINT_BATCH", 8192))

    cutoff_s = None
    if cutoff_factor and float(cutoff_factor) > 0:
        cutoff_s = tf.constant(float(cutoff_factor) ** 2, dtype=DTYPE)

    anchors_tf = tf.constant(np.asarray(anchors, np.float64), dtype=DTYPE)
    ell_tf = tf.constant(np.asarray(ell, np.float64).reshape(-1, 1), dtype=DTYPE)
    aw_tf = tf.constant(np.asarray(aw, np.float64).reshape(-1, 1), dtype=DTYPE)

    use_energy = (int(whiten_mode_tf.numpy()) == 0)  # one sync
    eps = tf.cast(float(cfg.WHITEN_EPS), DTYPE)
    if use_energy:
        G_tf = tf.constant(np.asarray(G_full, np.float64).reshape(-1), dtype=DTYPE)
        denom_tf = tf.sqrt(tf.maximum(G_tf, tf.cast(0.0, DTYPE)) + eps)[:, None]  # (M,1)
        eps_floor = tf.cast(1e-8, DTYPE)
        denom_tf = tf.maximum(denom_tf, eps_floor)
    else:
        denom_tf = tf.ones((M, 1), dtype=DTYPE)

    invNq = tf.cast(1.0 / float(Nq), DTYPE)
    M0_base = tf.cast(float(getattr(cfg, 'ANCHOR_INIT', 1)), DTYPE)

    ids_all = np.arange(M, dtype=np.int32)
    pruned_ids, dense_ids = _split_pruned_dense(ids_all, pruner)

    loss_sum = tf.zeros([], dtype=DTYPE)
    Rabs_var = tf.Variable(tf.zeros((M,), dtype=DTYPE), trainable=False)

    # ---- pruned anchors: edge-based accumulation over local quadrature subsets ----
    for s in range(0, int(pruned_ids.shape[0]), ab):
        e = min(s + ab, int(pruned_ids.shape[0]))
        ids = pruned_ids[s:e]
        if ids.size == 0:
            continue

        idx_flat, seg_ids = pruner.build_edges_full(ids)
        if idx_flat.size == 0:
            continue

        eb = int(getattr(cfg, "EVAL_EDGE_BATCH", 2048))

        idx_tf = tf.constant(idx_flat, dtype=tf.int32)
        seg_tf = tf.constant(seg_ids, dtype=tf.int32)

        ids_tf = tf.constant(ids, dtype=tf.int32)
        a_batch = tf.gather(anchors_tf, ids_tf)  # (A,2)
        e_batch = tf.gather(ell_tf, ids_tf)  # (A,1)
        aw_batch = tf.gather(aw_tf, ids_tf)  # (A,1)
        den_batch = tf.gather(denom_tf, ids_tf)  # (A,1)

        A = int(ids.shape[0])
        E = int(idx_flat.shape[0])
        sums = tf.zeros((A,), dtype=DTYPE)

        for t in range(0, E, eb):
            idx_c = idx_tf[t:t + eb]
            seg_c = seg_tf[t:t + eb]

            xy = tf.gather(Xq_tf, idx_c)
            b0 = tf.gather(b0_all_tf, idx_c)
            db0 = tf.gather(db0_all_tf, idx_c)

            a_edge = tf.gather(a_batch, seg_c)
            e_edge = tf.gather(e_batch, seg_c)

            gu = grad_u_tf(unet, gnet, lift_mode_tf, xy,
                           cfg.BUBBLE_SCALE, bubble_mode_tf, softmin_tau_tf, bubble_power_tf)  # (Ec,2)
            gv = gradv_edges_from_precomputed(xy, b0, db0, a_edge, e_edge, cutoff_s=cutoff_s)  # (Ec,2)

            dot = tf.reduce_sum(gu * gv, axis=1)  # (Ec,)
            sums = sums + tf.math.unsorted_segment_sum(dot, seg_c, num_segments=A)  # (A,)

        R = sums * invNq  # (A,)
        r = (R[:, None] / den_batch)
        loss_sum = loss_sum + tf.reduce_sum(aw_batch * tf.square(r))

        Rabs_var.scatter_nd_update(tf.reshape(ids_tf, (-1, 1)), tf.abs(R))

    # ---- dense anchors: original batching kernels ----
    for s in range(0, int(dense_ids.shape[0]), ab):
        e = min(s + ab, int(dense_ids.shape[0]))
        ids = dense_ids[s:e]
        if ids.size == 0:
            continue

        ids_tf = tf.constant(ids, dtype=tf.int32)
        xi = tf.gather(anchors_tf, ids_tf)
        el = tf.gather(ell_tf, ids_tf)
        awb = tf.gather(aw_tf, ids_tf)
        den = tf.gather(denom_tf, ids_tf)

        accR = tf.zeros((int(ids.shape[0]),), dtype=DTYPE)
        for p in range(0, Nq, pb):
            q = min(p + pb, Nq)
            xy = Xq_tf[p:q]
            b0 = b0_all_tf[p:q]
            db0 = db0_all_tf[p:q]
            accR = accR + R_block(
                unet, gnet, lift_mode_tf, xy, b0, db0, xi, el,
                bubble_scale=cfg.BUBBLE_SCALE,
                bubble_mode=bubble_mode_tf,
                softmin_tau=softmin_tau_tf,
                bubble_power=bubble_power_tf,
                cutoff_s=cutoff_s
            )

        R = accR * invNq
        r = (R[:, None] / den)
        loss_sum = loss_sum + tf.reduce_sum(awb * tf.square(r))

        Rabs_var.scatter_nd_update(tf.reshape(ids_tf, (-1, 1)), tf.abs(R))

    w_sum = tf.reduce_sum(tf.reshape(aw_tf, (-1,)))

    loss_mean = loss_sum / tf.maximum(tf.cast(1.0, DTYPE), tf.cast(w_sum, DTYPE))
    G_out = (np.asarray(G_full).reshape(-1) if G_full is not None else None)
    return {"loss": float(loss_mean.numpy()), "Rabs": Rabs_var.numpy().astype(np.float64), "G": G_out}


def adam_train_cached_kdtree(unet, gnet, lift_mode: int,
                             Xq_tf, b0_all_tf, db0_all_tf,
                             anchors_tf: tf.Tensor, ell_tf: tf.Tensor, aw_tf: tf.Tensor,
                             denom_tf: tf.Tensor,
                             cfg: Config,
                             bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                             steps: int,
                             seed: int,
                             pruner: KDTreePruner,
                             cutoff_factor: float = 4.0, lr0_override: float = None, lr1_override: float = None,
                             deterministic: int = None):
    """
    Adam training with KDTree-pruned edge lists:
      - For each anchor in a minibatch, sample points from its frozen neighborhood I_j.
      - Accumulate contributions via unsorted_segment_sum (edge list), avoiding (points×anchors) matrices.
    Dense anchors (huge neighborhoods) fall back to global-uniform point sampling (still edge-based).
    """
    steps = int(steps)
    if steps <= 0:
        return

    vars_ = unet.trainable_variables
    rng = np.random.default_rng(int(seed))

    det = int(getattr(cfg, "DETERMINISTIC_BATCHING", 0)) if deterministic is None else int(deterministic)

    # LR overrides (used by monotone-H1 guard phases)
    lr0 = float(cfg.LR0) if lr0_override is None else float(lr0_override)
    lr1 = float(cfg.LR1) if lr1_override is None else float(lr1_override)
    if lr0 <= 0.0:
        lr0 = 1e-6
    if lr1 <= 0.0:
        lr1 = min(lr0, 1e-6)

    decay_rate = (lr1 / lr0) ** (1.0 / float(max(1, steps - 1)))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr0,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=False,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    reg_coeff = tf.constant(cfg.LAMBDA_REG, dtype=DTYPE)
    lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)

    Nq = int(Xq_tf.shape[0])
    M = int(anchors_tf.shape[0])
    M0_base = tf.cast(float(getattr(cfg, 'ANCHOR_INIT', 1)), DTYPE)
    w_sum_total_tf = tf.maximum(tf.reduce_sum(aw_tf), tf.cast(1e-30, DTYPE))  # constant sum_aw over all anchors

    pb0 = int(cfg.ADAM_POINT_BATCH)
    ab0 = int(cfg.ADAM_ANCHOR_BATCH)
    ga = int(cfg.ADAM_GRAD_ACCUM)

    # OOM-safe dynamic batches for the KDTree Adam path (backs off on ResourceExhaustedError)
    pb_cur = int(pb0)
    ab_cur = int(ab0)

    # Optional stencil-bank path (fast on CPU): pre-sampled per-anchor point indices, gathered in TF.
    use_stencil = (int(getattr(cfg, "USE_STENCIL_BANK", 1)) == 1)
    stencil_bank_tf = None
    nloc_tf = None
    Kbank = int(getattr(cfg, "STENCIL_BANK_K", 4))
    Smax = int(getattr(cfg, "STENCIL_SMAX", 0))
    if Smax <= 0:
        Smax = max(8, int(pb0) // max(1, int(ab0)))
    st_seed = int(getattr(cfg, "STENCIL_SEED", seed))
    if use_stencil:
        pruner.enable_stencil_bank(Smax=Smax, K=Kbank, seed=st_seed, force_rebuild=False)
        sb_np = pruner.stencil_bank_np()
        nl_np = pruner.nloc_np()
        if (sb_np is None) or (nl_np is None):
            use_stencil = False
        else:
            stencil_bank_tf = tf.constant(sb_np, dtype=tf.int32)
            nloc_tf = tf.constant(nl_np, dtype=tf.int32)

    cutoff_s = None
    if cutoff_factor and float(cutoff_factor) > 0:
        cutoff_s = tf.constant(float(cutoff_factor) ** 2, dtype=DTYPE)

    @tf.function(reduce_retracing=True)
    def train_step_edges(xy, xy_energy, b0, db0, a_edge, e_edge,
                         seg_ids, scale_per_anchor, aw, den,
                         M_tf):
        A = tf.shape(scale_per_anchor)[0]
        with tf.GradientTape() as tape:
            gu = grad_u_tf(unet, gnet, lift_mode_tf, xy,
                           cfg.BUBBLE_SCALE, bubble_mode_tf, softmin_tau_tf, bubble_power_tf)  # (E,2)
            gv = gradv_edges_from_precomputed(xy, b0, db0, a_edge, e_edge, cutoff_s=cutoff_s)  # (E,2)
            dot = tf.reduce_sum(gu * gv, axis=1)  # (E,)
            sums = tf.math.unsorted_segment_sum(dot, seg_ids, num_segments=A)  # (A,)
            R = (sums * scale_per_anchor) / tf.cast(float(Nq), DTYPE)  # (A,)
            r = R[:, None] / den  # (A,1)
            # PDE term: unbiased estimator of (sum_j aw_j r_j^2) / sum_aw_total with uniform anchor sampling.
            num = tf.reduce_sum(aw * tf.square(r))
            A = tf.cast(tf.shape(aw)[0], DTYPE)
            L_pde = (tf.cast(M_tf, DTYPE) / tf.maximum(A, tf.cast(1.0, DTYPE))) * (num / w_sum_total_tf)

            # Energy regularizer on a separate *uniform* point batch (reduces bias vs. using stencil-local points).
            gu_e = grad_u_tf(unet, gnet, lift_mode_tf, xy_energy,
                             cfg.BUBBLE_SCALE, bubble_mode_tf, softmin_tau_tf, bubble_power_tf)
            L_energy = tf.reduce_mean(tf.reduce_sum(tf.square(gu_e), axis=1))

            L = tf.cast(cfg.W_PDE, DTYPE) * L_pde + tf.cast(cfg.W_ENERGY, DTYPE) * L_energy
            reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) if vars_ else tf.constant(0.0, dtype=DTYPE)
            J = L + reg_coeff * reg
        grads = tape.gradient(J, vars_)
        return J, grads

    M_tf = tf.constant(int(M), dtype=tf.int32)

    for _it in range(steps):
        grads_accum = _zeros_like(vars_) if vars_ else []
        for _ in range(ga):
            retries = 0
            while True:
                try:
                    A = int(max(1, ab_cur))
                    S = int(max(1, pb_cur // max(1, A)))

                    aids = (((_it * ga + _) * A + np.arange(A, dtype=np.int64)) % M).astype(
                        np.int32) if det == 1 else rng.integers(0, M, size=A, dtype=np.int32)
                    aids_tf = tf.constant(aids, dtype=tf.int32)

                    # Build edge list (stencil-bank fast path when enabled; falls back to original KDTree sampler)

                    if use_stencil:
                        # Clamp S to the prebuilt stencil length (keeps this robust under OOM-backoff).
                        S_eff = int(min(int(S), int(Smax)))
                        S_eff = int(max(1, S_eff))
                        k = int(((_it * ga) + _) % max(1, int(Kbank)))
                        st_view = stencil_bank_tf[:, k, :]  # (M, Smax)
                        idx_mat_full = tf.gather(st_view, aids_tf)  # (A, Smax)
                        idx_mat = idx_mat_full[:, :S_eff]  # (A, S_eff)
                        idx_tf = tf.reshape(idx_mat, (-1,))  # (E,)
                        seg_tf = tf.repeat(tf.range(A, dtype=tf.int32), repeats=S_eff)  # (E,)
                        nloc_b = tf.gather(nloc_tf, aids_tf)  # (A,)
                        scale_tf = tf.cast(nloc_b, DTYPE) / tf.cast(float(S_eff), DTYPE)  # (A,)
                    else:
                        # Original (unbiased per-step) sampler: slower on CPU due to Python edge-list build.
                        idx_flat, seg_ids, scales, _nloc = pruner.build_edges_for_batch(aids, S, rng=rng)
                        idx_tf = tf.constant(idx_flat, dtype=tf.int32)
                        seg_tf = tf.constant(seg_ids, dtype=tf.int32)
                        scale_tf = tf.constant(scales, dtype=DTYPE)

                    xy = tf.gather(Xq_tf, idx_tf)
                    b0 = tf.gather(b0_all_tf, idx_tf)
                    db0 = tf.gather(db0_all_tf, idx_tf)

                    xi = tf.gather(anchors_tf, aids_tf)  # (A,2)
                    el = tf.gather(ell_tf, aids_tf)  # (A,1)
                    awb = tf.gather(aw_tf, aids_tf)  # (A,1)
                    den = tf.gather(denom_tf, aids_tf)  # (A,1)

                    a_edge = tf.gather(xi, seg_tf)
                    e_edge = tf.gather(el, seg_tf)

                    # Uniform points for energy regularization (unbiased over Ω)
                    pb_e0 = int(getattr(cfg, "ENERGY_POINT_BATCH", 0) or int(pb_cur))
                    pb_e = int(max(32, min(int(Nq), pb_e0)))
                    pids_e = (((_it * ga + _) * pb_e + np.arange(pb_e, dtype=np.int64)) % Nq).astype(
                        np.int32) if det == 1 else rng.integers(0, Nq, size=pb_e, dtype=np.int32)
                    xy_energy = tf.gather(Xq_tf, tf.constant(pids_e, dtype=tf.int32))

                    J, g = train_step_edges(xy, xy_energy, b0, db0, a_edge, e_edge,
                                            seg_tf, scale_tf, awb, den, M_tf)
                    grads_accum = _safe_add_grads(grads_accum, g)
                    break

                except tf.errors.ResourceExhaustedError:
                    retries += 1
                    if retries > 16:
                        raise
                    # Backoff: first cut point batch, then anchor batch.
                    if pb_cur > 128:
                        pb_cur = max(128, pb_cur // 2)
                    elif ab_cur > 1:
                        ab_cur = max(1, ab_cur // 2)
                    else:
                        raise
                    print(f"[OOM-backoff kd-adam] pb={pb_cur} ab={ab_cur} (retry {retries}/16)", flush=True)

        inv = tf.constant(1.0 / float(ga), dtype=DTYPE)
        grads_accum = [g * inv for g in grads_accum]
        clip = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0))
        if clip and float(clip) > 0.0:
            grads_accum, _ = tf.clip_by_global_norm(grads_accum, tf.cast(float(clip), DTYPE))
        opt.apply_gradients(zip(grads_accum, vars_))


def adam_train_cached(unet, gnet, lift_mode: int,
                      Xq_tf, b0_all_tf, db0_all_tf,
                      anchors_tf: tf.Tensor, ell_tf: tf.Tensor, aw_tf: tf.Tensor,
                      denom_tf: tf.Tensor,
                      cfg: Config,
                      bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                      steps: int,
                      seed: int,
                      cutoff_factor: float = 4.0, lr0_override: float = None, lr1_override: float = None,
                      deterministic: int = None):
    """
    Adam training with cached energy denominators and cached bubble (b0, db0) on global points.
    Biggest speed win: removes re-computation of G_j inside every minibatch step.
    """
    steps = int(steps)
    if steps <= 0:
        return

    vars_ = unet.trainable_variables
    rng = np.random.default_rng(int(seed))

    det = int(getattr(cfg, "DETERMINISTIC_BATCHING", 0)) if deterministic is None else int(deterministic)

    # LR overrides (used by monotone-H1 guard phases)
    lr0 = float(cfg.LR0) if lr0_override is None else float(lr0_override)
    lr1 = float(cfg.LR1) if lr1_override is None else float(lr1_override)
    if lr0 <= 0.0:
        lr0 = 1e-6
    if lr1 <= 0.0:
        lr1 = min(lr0, 1e-6)

    decay_rate = (lr1 / lr0) ** (1.0 / float(max(1, steps - 1)))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr0,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=False,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    reg_coeff = tf.constant(cfg.LAMBDA_REG, dtype=DTYPE)
    lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)

    Nq = int(Xq_tf.shape[0])
    M = int(anchors_tf.shape[0])
    M0_base = tf.cast(float(getattr(cfg, 'ANCHOR_INIT', 1)), DTYPE)
    w_sum_total_tf = tf.maximum(tf.reduce_sum(aw_tf), tf.cast(1e-30, DTYPE))  # sum of anchor weights

    pb0 = int(cfg.ADAM_POINT_BATCH)
    ab0 = int(cfg.ADAM_ANCHOR_BATCH)
    ga0 = int(cfg.ADAM_GRAD_ACCUM)

    pb = int(pb0)
    ab = int(ab0)
    ga = int(ga0)

    cutoff_s = None
    if cutoff_factor and float(cutoff_factor) > 0:
        cutoff_s = tf.constant(float(cutoff_factor) ** 2, dtype=DTYPE)

    @tf.function(reduce_retracing=True)
    def train_step(xy, b0, db0, xi, el, aw, den):
        with tf.GradientTape() as tape:
            # sum over points -> mean to approximate integral (uniform QMC weights)
            gu = grad_u_tf(unet, gnet, lift_mode_tf, xy,
                           cfg.BUBBLE_SCALE, bubble_mode_tf, softmin_tau_tf, bubble_power_tf)  # (pb,2)
            gvx, gvy = gradv_from_precomputed(xy, b0, db0, xi, el, cutoff_s=cutoff_s)  # (pb,ab)
            dot = gu[:, 0:1] * gvx + gu[:, 1:2] * gvy  # (pb,ab)
            R = tf.reduce_mean(dot, axis=0)  # (ab,)
            r = (R[:, None] / den)
            # PDE term: unbiased estimator of (sum_j aw_j r_j^2) / sum_aw_total
            # because anchors are sampled uniformly with replacement.
            num = tf.reduce_sum(aw * tf.square(r))
            A = tf.cast(tf.shape(aw)[0], DTYPE)
            L_pde = (tf.cast(M, DTYPE) / tf.maximum(A, tf.cast(1.0, DTYPE))) * (num / w_sum_total_tf)

            # Energy regularizer: average ||∇u||^2 on the same *uniform* point batch (helps smooth H1).
            L_energy = tf.reduce_mean(tf.reduce_sum(tf.square(gu), axis=1))

            L = tf.cast(cfg.W_PDE, DTYPE) * L_pde + tf.cast(cfg.W_ENERGY, DTYPE) * L_energy
            reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) if vars_ else tf.constant(0.0, dtype=DTYPE)
            J = L + reg_coeff * reg
        grads = tape.gradient(J, vars_)
        return J, grads

    for _it in range(steps):
        while True:
            try:
                grads_accum = _zeros_like(vars_) if vars_ else []
                for _acc in range(ga):
                    step_idx = _it * ga + _acc
                    pids = ((step_idx * pb + np.arange(pb, dtype=np.int64)) % Nq).astype(
                        np.int32) if det == 1 else rng.integers(0, Nq, size=pb, dtype=np.int32)
                    aids = ((step_idx * ab + np.arange(ab, dtype=np.int64)) % M).astype(
                        np.int32) if det == 1 else rng.integers(0, M, size=ab, dtype=np.int32)

                    xy = tf.gather(Xq_tf, pids)
                    b0 = tf.gather(b0_all_tf, pids)
                    db0 = tf.gather(db0_all_tf, pids)

                    xi = tf.gather(anchors_tf, aids)
                    el = tf.gather(ell_tf, aids)
                    aw = tf.gather(aw_tf, aids)
                    den = tf.gather(denom_tf, aids)

                    J, g = train_step(xy, b0, db0, xi, el, aw, den)
                    if vars_:
                        grads_accum = _safe_add_grads(grads_accum, g)

                if vars_:
                    inv = tf.constant(1.0 / float(ga), dtype=DTYPE)
                    grads_accum = [g * inv for g in grads_accum]
                    clip = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0))
                    if clip and float(clip) > 0.0:
                        grads_accum, _ = tf.clip_by_global_norm(grads_accum, tf.cast(float(clip), DTYPE))
                    opt.apply_gradients(zip(grads_accum, vars_))
                break

            except tf.errors.ResourceExhaustedError:
                # Progressive backoff: shrink pb/ab; increase grad_accum to preserve effective batch.
                pb = max(256, pb // 2)
                ab = max(1, ab // 2)
                eff0 = int(pb0) * int(cfg.ADAM_GRAD_ACCUM)
                ga = int(min(64, max(1, math.ceil(eff0 / max(1, pb)))))
                print(f"[OOM backoff] Adam pb -> {pb}, ab -> {ab}, grad_accum -> {ga}")


@tf.function
def residuals_R_and_G(unet, gnet, lift_mode, xy, w_pts, xi, ell,
                      bubble_scale, bubble_mode, softmin_tau, bubble_power,
                      whiten_eps, whiten_mode):
    # xy:(N,2) w_pts:(N,1) xi:(M,2) ell:(M,1)
    gu = grad_u_tf(unet, gnet, lift_mode, xy, bubble_scale, bubble_mode, softmin_tau, bubble_power)  # (N,2)
    b0, db0 = bubble0_and_grad(xy)  # b0:(N,1), db0:(N,2)
    k, gradk = kernel_gauss_and_grad(xy, xi, ell)  # k:(N,M), gradk:(N,M,2)

    # gradv = k*db0 + b0*gradk
    gradv = k[:, :, None] * db0[:, None, :] + b0[:, None, :] * gradk  # (N,M,2)

    dot = tf.reduce_sum(gu[:, None, :] * gradv, axis=2)  # (N,M)
    R = tf.reduce_sum(w_pts * dot, axis=0)  # (M,)
    gv2 = tf.reduce_sum(tf.square(gradv), axis=2)  # (N,M)
    G = tf.reduce_sum(w_pts * gv2, axis=0)  # (M,)

    denom_energy = tf.sqrt(tf.maximum(G, tf.constant(0.0, dtype=DTYPE)) + tf.constant(whiten_eps, dtype=DTYPE))

    eps_floor = tf.cast(1e-8, DTYPE)

    denom_energy = tf.maximum(denom_energy, eps_floor)
    denom = tf.cond(tf.equal(tf.cast(whiten_mode, tf.int32), 0),
                    lambda: denom_energy,
                    lambda: tf.ones_like(denom_energy))
    r = R / denom
    return R, G, r


@tf.function
def loss_on_batches(unet, gnet, lift_mode,
                    xy, w_pts,
                    xi, ell, a_w,
                    bubble_scale, bubble_mode, softmin_tau, bubble_power,
                    whiten_eps, whiten_mode):
    R, G, r = residuals_R_and_G(
        unet, gnet, lift_mode, xy, w_pts, xi, ell,
        bubble_scale, bubble_mode, softmin_tau, bubble_power,
        whiten_eps, whiten_mode
    )
    aw = tf.reshape(a_w, [-1])  # (M,)
    L = tf.reduce_sum(aw * tf.square(r)) / tf.maximum(tf.reduce_sum(aw), tf.cast(1e-30, DTYPE))
    return L, tf.abs(R), G


# -------------------------
# H1 error (same grading trick as original)
# -------------------------
def build_H1_points(cfg: Config, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Common-random-number (CRN) Sobol points with grading.
    Returns pts, uv, jac. uv are fixed Sobol points in [0,1]^2; pts are graded points x=u^p,y=v^p; jac is grading Jacobian.
    """
    n = int(n)
    p = float(cfg.H1_GRADE_P)
    uv, _ = build_ref_cloud_square(n, seed=seed, cfg=cfg, dim=2)
    u = uv[:, 0]
    v = uv[:, 1]
    x = u ** p
    y = v ** p
    jac = (p * (u ** (p - 1.0))) * (p * (v ** (p - 1.0)))
    pts = np.stack([x, y], axis=1).astype(np.float64)
    return pts, uv.astype(np.float64), jac.astype(np.float64)


def giqmc_weights_from_eta(uv: np.ndarray, jac: np.ndarray, eta: np.ndarray, cfg: Config,
                           ema_state: Dict[str, np.ndarray] = None, key: str = "h1") -> Tuple[
    np.ndarray, Dict[str, np.ndarray]]:
    """GI-QMC weights w = J/q normalized to |Ω| (Ω=[0,1]^2 so |Ω|=1).

    q(x_i) ∝ (eta(x_i)+eps)^alpha, with optional EMA smoothing on eta across outer steps.
    uv are fixed (CRN) so the estimator is low-variance across outer steps.
    """
    eps = float(getattr(cfg, "H1_IMPORT_EPS", 1e-10))
    alpha = float(getattr(cfg, "H1_IMPORT_ALPHA", 0.75))
    beta = float(getattr(cfg, "H1_ETA_EMA_BETA", 0.0))

    eta = np.asarray(eta, np.float64).reshape(-1)
    jac = np.asarray(jac, np.float64).reshape(-1)

    if ema_state is None:
        ema_state = {}
    if beta > 0.0:
        prev = ema_state.get(key, None)
        if prev is None or prev.shape != eta.shape:
            ema = eta.copy()
        else:
            ema = beta * prev + (1.0 - beta) * eta
        ema_state[key] = ema
        eta_eff = ema
    else:
        eta_eff = eta

    q = np.power(eta_eff + eps, alpha)
    s = float(np.sum(q))
    if not np.isfinite(s) or s <= 0.0:
        q = np.ones_like(q)
        s = float(len(q))
    q = q / s

    w = jac / (q + 1e-300)
    w = w / float(len(w))  # Monte Carlo normalization
    area_est = float(np.sum(w))  # should be close to 1 for Ω=[0,1]^2
    if np.isfinite(area_est) and area_est > 0.0:
        w = w / area_est  # enforce exact area normalization
    return w.astype(np.float64), ema_state


# -------------------------
# GI-QMC H1 evaluator (graded + importance, CRN, control-variate integrand)
# -------------------------
@dataclass
class H1GIQMCEvaluator:
    """
    Meshfree low-variance estimator for the (absolute) H1 error:
        E = ∫ (e^2 + |∇e|^2) dx,  e=u_theta - u_ref
    using:
      - fixed Sobol CRN points in [0,1]^2,
      - grading toward the corner singularity,
      - importance weights based on eta=|Δu_theta|,
      - optional EMA smoothing of eta to reduce iteration-to-iteration noise.

    Notes:
      - Points (uv) are generated once and never change (strict CRN).
      - The grading map x=u^p, y=v^p is fixed (strictly meshfree).
      - Only the importance weights vary with the current model.
    """
    pts: np.ndarray  # graded points (N,2)
    uv: np.ndarray  # raw Sobol points (N,2)
    jac: np.ndarray  # grading Jacobian (N,)
    cfg: Config
    ema_eta: np.ndarray = None
    fixed_w: np.ndarray = None  # frozen GI-QMC weights (optional)
    giqmc_calls: int = 0  # number of times GI-QMC weights were built

    @staticmethod
    def build(cfg: Config, n: int, seed: int) -> "H1GIQMCEvaluator":
        pts, uv, jac = build_H1_points(cfg, n=int(n), seed=int(seed))
        return H1GIQMCEvaluator(pts=pts, uv=uv, jac=jac, cfg=cfg, ema_eta=None)

    def _eta_abs_laplace(self, unet, gnet, lift_mode: int,
                         bubble_mode_tf, softmin_tau_tf, bubble_power_tf) -> np.ndarray:
        """eta(x)=|Δu_theta(x)| on the evaluator points (chunked, GPU-safe)."""
        lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)
        Z = self.pts
        N = int(Z.shape[0])
        B = int(min(32768, max(1024, getattr(self.cfg, "H1_CHUNK", 16384))))
        outs = []
        for s in range(0, N, B):
            e = min(s + B, N)
            z_tf = tf.constant(Z[s:e], dtype=DTYPE)
            lap_tf = laplace_u_tf(
                unet, gnet, lift_mode_tf, z_tf,
                float(self.cfg.LAPLACE_H),
                float(self.cfg.BUBBLE_SCALE), bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
            )
            outs.append(tf.abs(lap_tf))
        return tf.concat(outs, axis=0).numpy().reshape(-1).astype(np.float64)

    def weights_giqmc(self, eta: np.ndarray) -> np.ndarray:
        """GI-QMC weights: w ∝ J/q with Σw=|Ω|=1."""

        # Optional: freeze importance weights after a few calls to make the metric comparable across iterations.
        self.giqmc_calls = int(getattr(self, "giqmc_calls", 0)) + 1
        freeze = int(getattr(self.cfg, "H1_FREEZE_IMPORTANCE", 0)) == 1
        freeze_after = int(getattr(self.cfg, "H1_FREEZE_AFTER_CALLS", 1))
        if freeze and (freeze_after >= 1) and (self.fixed_w is not None) and (self.giqmc_calls > freeze_after):
            return np.asarray(self.fixed_w, np.float64)
        eps = float(getattr(self.cfg, "H1_IMPORT_EPS", 1e-10))
        alpha = float(getattr(self.cfg, "H1_IMPORT_ALPHA", 0.75))
        beta = float(getattr(self.cfg, "H1_ETA_EMA_BETA", 0.0))

        eta = np.asarray(eta, np.float64).reshape(-1)
        jac = np.asarray(self.jac, np.float64).reshape(-1)

        if beta > 0.0:
            if self.ema_eta is None or self.ema_eta.shape != eta.shape:
                self.ema_eta = eta.copy()
            else:
                self.ema_eta = beta * self.ema_eta + (1.0 - beta) * eta
            eta_eff = self.ema_eta
        else:
            eta_eff = eta

        q = np.power(eta_eff + eps, alpha)
        s = float(np.sum(q))
        if (not np.isfinite(s)) or s <= 0.0:
            q = np.ones_like(q, dtype=np.float64) / float(len(q))
        else:
            q = (q / s).astype(np.float64)

        w = jac / (q + 1e-300)
        w = w / float(len(w))  # MC normalization
        area = float(np.sum(w))  # should approximate |Ω|=1
        if np.isfinite(area) and area > 0.0:
            w = w / area  # enforce exact area normalization

        if freeze and (freeze_after >= 1) and (self.fixed_w is None) and (self.giqmc_calls == freeze_after):
            self.fixed_w = w.astype(np.float64).copy()
        return w.astype(np.float64)

    def weights_graded(self) -> np.ndarray:
        """Pure graded QMC weights (no importance): w ∝ J with Σw=1."""
        jac = np.asarray(self.jac, np.float64).reshape(-1)
        w = jac / float(len(jac))
        area = float(np.sum(w))
        if np.isfinite(area) and area > 0.0:
            w = w / area
        return w.astype(np.float64)

    def reference_energy(self) -> float:
        """Precompute ∫ (u_ref^2 + |∇u_ref|^2) dx using graded-only weights (stable denom)."""
        w0 = self.weights_graded()
        x = self.pts[:, 0]
        y = self.pts[:, 1]
        u_ex = u_exact_np(x, y)
        gx_ex, gy_ex = grad_u_exact_np(x, y)
        return float(np.sum(w0 * (u_ex * u_ex + gx_ex * gx_ex + gy_ex * gy_ex)))

    def error_energy(self, unet, gnet, lift_mode: int,
                     bubble_mode_tf, softmin_tau_tf, bubble_power_tf) -> float:
        """Absolute H1 error energy E = ∫ (e^2 + |∇e|^2) dx."""
        if int(getattr(self.cfg, "H1_USE_GIQMC", 1)) == 1:
            eta = self._eta_abs_laplace(unet, gnet, lift_mode, bubble_mode_tf, softmin_tau_tf, bubble_power_tf)
            w = self.weights_giqmc(eta)
        else:
            w = self.weights_graded()

        u_pred, gx_pred, gy_pred = _eval_Bu_and_grad_np(
            unet, gnet, lift_mode, self.pts, self.cfg,
            bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
            chunk=int(getattr(self.cfg, "H1_CHUNK", 16384)),
        )
        x = self.pts[:, 0]
        y = self.pts[:, 1]
        u_ex = u_exact_np(x, y)
        gx_ex, gy_ex = grad_u_exact_np(x, y)

        e0 = u_pred - u_ex
        ex = gx_pred - gx_ex
        ey = gy_pred - gy_ex
        return float(np.sum(w * (e0 * e0 + ex * ex + ey * ey)))

    def rel_H1(self, unet, gnet, lift_mode: int,
               bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
               ref_energy_const: float) -> float:
        """Relative H1 error sqrt(E_err / E_ref_const)."""
        Eerr = self.error_energy(unet, gnet, lift_mode, bubble_mode_tf, softmin_tau_tf, bubble_power_tf)
        return float(np.sqrt(max(Eerr, 0.0) / max(ref_energy_const, 1e-300)))


def _eval_Bu_and_grad_np(unet, gnet, lift_mode, XY, cfg: Config,
                         bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
                         chunk=16384):
    XY = np.asarray(XY, np.float64)
    N = XY.shape[0]
    u_out = np.zeros((N,), np.float64)
    gx_out = np.zeros((N,), np.float64)
    gy_out = np.zeros((N,), np.float64)

    lift_mode_tf_local = tf.constant(int(lift_mode), dtype=tf.int32)
    s = 0
    chunk = int(chunk)
    while s < N:
        e = min(s + chunk, N)
        xy_tf = tf.constant(XY[s:e, :], dtype=DTYPE)
        u_tf, g_tf = Bu_and_grad_lift(unet, gnet, lift_mode_tf_local, xy_tf,
                                      cfg.BUBBLE_SCALE, bubble_mode_tf, softmin_tau_tf, bubble_power_tf)
        u = u_tf.numpy().reshape(-1)
        g = g_tf.numpy()
        u_out[s:e] = u
        gx_out[s:e] = g[:, 0]
        gy_out[s:e] = g[:, 1]
        s = e
    return u_out, gx_out, gy_out


def rel_H1_error_on_points(unet, gnet, lift_mode, pts, w, cfg: Config,
                           bubble_mode_tf, softmin_tau_tf, bubble_power_tf) -> float:
    u_pred, gx_pred, gy_pred = _eval_Bu_and_grad_np(
        unet, gnet, lift_mode, pts, cfg,
        bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
        chunk=int(cfg.H1_CHUNK),
    )

    u_ex = u_exact_np(pts[:, 0], pts[:, 1])
    gx_ex, gy_ex = grad_u_exact_np(pts[:, 0], pts[:, 1])

    e0 = u_pred - u_ex
    ex = gx_pred - gx_ex
    ey = gy_pred - gy_ex

    num = np.sum(w * (e0 * e0 + ex * ex + ey * ey))
    den = np.sum(w * (u_ex * u_ex + gx_ex * gx_ex + gy_ex * gy_ex))
    return float(np.sqrt(max(num, 0.0) / max(den, 1e-300)))


# -------------------------
# Boundary sampling for gnet (unchanged)
# -------------------------
def sample_boundary_corner_biased_np(rng, n, power=10.0, p_focus=0.70):
    n = int(n)
    m = rng.random(n) < float(p_focus)
    xy = np.empty((n, 2), dtype=np.float64)

    k = int(np.sum(m))
    if k > 0:
        edge = rng.integers(0, 2, size=k)
        u = rng.random(k) ** float(power)
        xy_f = np.zeros((k, 2), dtype=np.float64)
        xy_f[edge == 0, 0] = 0.0
        xy_f[edge == 0, 1] = u[edge == 0]
        xy_f[edge == 1, 1] = 0.0
        xy_f[edge == 1, 0] = u[edge == 1]
        xy[m] = xy_f

    r = n - k
    if r > 0:
        side = rng.integers(0, 4, size=r)
        t = rng.random(r)
        xy_u = np.empty((r, 2), dtype=np.float64)
        xy_u[side == 0] = np.stack([np.zeros(np.sum(side == 0)), t[side == 0]], axis=1)
        xy_u[side == 1] = np.stack([np.ones(np.sum(side == 1)), t[side == 1]], axis=1)
        xy_u[side == 2] = np.stack([t[side == 2], np.zeros(np.sum(side == 2))], axis=1)
        xy_u[side == 3] = np.stack([t[side == 3], np.ones(np.sum(side == 3))], axis=1)
        xy[~m] = xy_u

    return xy


def boundary_max_error_qmc(gnet, cfg: Config, n_per_edge=20000, chunk=32768, seed=0):
    t, _ = build_ref_cloud_square(int(n_per_edge), seed=int(seed) + 101, cfg=cfg, dim=1)
    t = t.reshape(-1)
    e0 = np.stack([np.zeros_like(t), t], axis=1)
    e1 = np.stack([np.ones_like(t), t], axis=1)
    e2 = np.stack([t, np.zeros_like(t)], axis=1)
    e3 = np.stack([t, np.ones_like(t)], axis=1)
    xy = np.vstack([e0, e1, e2, e3]).astype(np.float64)

    u_true = u_exact_np(xy[:, 0], xy[:, 1]).reshape(-1, 1)
    u_pred = np.zeros_like(u_true)

    for s in range(0, xy.shape[0], int(chunk)):
        e = min(s + int(chunk), xy.shape[0])
        u_pred[s:e] = gnet(tf.constant(xy[s:e], dtype=DTYPE)).numpy()

    return float(np.max(np.abs(u_pred - u_true)))


def train_gnet_boundary(gnet, cfg: Config, seed=0):
    rng = np.random.default_rng(int(seed))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(cfg.GNET_LR),
        decay_steps=2000,
        decay_rate=0.9,
        staircase=False,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

    p_norm = tf.constant(float(cfg.GNET_P_NORM), dtype=DTYPE)
    beta_p = tf.constant(float(cfg.GNET_BETA_P), dtype=DTYPE)
    beta_top = tf.constant(float(cfg.GNET_BETA_TOP), dtype=DTYPE)
    top_frac = float(cfg.GNET_TOP_FRAC)

    @tf.function
    def train_step(xy, u_true):
        with tf.GradientTape() as tape:
            u_pred = gnet(xy)
            e = u_pred - u_true
            l2 = tf.reduce_mean(tf.square(e))
            lp = tf.reduce_mean(tf.pow(tf.abs(e) + tf.constant(1e-12, dtype=DTYPE), p_norm))
            abs_e = tf.reshape(tf.abs(e), [-1])
            n = tf.size(abs_e)
            k_float = tf.constant(top_frac, dtype=DTYPE) * tf.cast(n, DTYPE)
            k = tf.maximum(tf.constant(1, dtype=tf.int32), tf.cast(tf.round(k_float), tf.int32))
            topk = tf.nn.top_k(abs_e, k=k, sorted=False).values
            ltop = tf.reduce_mean(tf.square(topk))
            loss = l2 + beta_p * lp + beta_top * ltop
        grads = tape.gradient(loss, gnet.trainable_variables)
        opt.apply_gradients(zip(grads, gnet.trainable_variables))
        return loss, l2, ltop

    steps = int(cfg.GNET_STEPS)
    batch = int(cfg.GNET_BATCH)

    for k in range(1, steps + 1):
        xy_b = sample_boundary_corner_biased_np(
            rng, batch,
            power=float(cfg.GNET_CORNER_POWER),
            p_focus=float(cfg.GNET_P_FOCUS),
        )
        u_b = u_exact_np(xy_b[:, 0], xy_b[:, 1]).reshape(-1, 1).astype(np.float64)
        while True:
            try:
                loss, l2, ltop = train_step(tf.constant(xy_b, dtype=DTYPE), tf.constant(u_b, dtype=DTYPE))
                break
            except tf.errors.ResourceExhaustedError:
                if batch <= 128:
                    raise
                batch = max(128, batch // 2)
                xy_b = xy_b[:batch]
                u_b = u_b[:batch]
                print(f"[OOM backoff] gnet batch -> {batch}")

        if cfg.GNET_PRINT_EVERY and (k % int(cfg.GNET_PRINT_EVERY) == 0):
            print(
                f"    [gnet] step {k:05d}  l2={float(l2.numpy()):.3e}  top={float(ltop.numpy()):.3e}  loss={float(loss.numpy()):.3e}")

    maxerr = boundary_max_error_qmc(
        gnet,
        cfg,
        n_per_edge=int(cfg.GNET_GRID_N_PER_EDGE),
        chunk=int(cfg.GNET_EVAL_CHUNK),
        seed=seed,
    )
    print(f"  [lift] gnet max boundary |err| ≈ {maxerr:.3e}")
    if maxerr > float(cfg.GNET_MAXERR_WARN):
        print("  [lift] WARNING: boundary fit is loose; this can invert Fig.3 trend.")
    return maxerr


# -------------------------
# Anchor bank helpers
# -------------------------
def init_anchors(cfg: Config, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initial anchors: Sobol points + equal ell, full weight."""
    pts, _ = build_ref_cloud_square(cfg.ANCHOR_INIT, seed=seed, cfg=cfg, dim=2)
    M0 = int(cfg.ANCHOR_INIT)
    ell0 = float(cfg.ELL0)
    ell = np.full((M0, 1), ell0, dtype=np.float64)
    if cfg.USE_MULTI_SCALE:
        m_small = int(round(cfg.ELL_MIX_FRAC_SMALL * M0))
        ell[:m_small, 0] = float(cfg.ELL0_SMALL)
        ell[m_small:, 0] = float(cfg.ELL0)
    aw = np.ones((M0, 1), dtype=np.float64)
    return pts.astype(np.float64), ell, aw


def ell_schedule(cfg: Config, M: int, M0: int) -> float:
    return float(max(cfg.ELL_MIN, cfg.ELL0 * (float(M0) / float(max(1, M))) ** float(cfg.ELL_SCHED_P)))


def mean_nn_spacing_np(xy: np.ndarray) -> float:
    """Mean nearest-neighbour spacing h (excluding self)."""
    if xy is None or xy.shape[0] < 2:
        return 1.0
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    nn = d[:, 1]
    return float(np.mean(nn))


def compute_base_ell(cfg, anchors: np.ndarray, M_future: int, M0: int) -> float:
    """Base ell for new anchors.

    Default: scheduled ell vs. M.
    If ELL_FROM_H: use rho=C_rho*h with rho=cutoff*ell (clamped).
    """
    base = float(ell_schedule(cfg, M_future, M0))
    if int(getattr(cfg, "ELL_FROM_H", 0)) and anchors is not None and anchors.shape[0] >= 2:
        h = mean_nn_spacing_np(anchors)
        rho = float(getattr(cfg, "ELL_C_RHO", 2.0)) * h
        cutoff = float(getattr(cfg, "KERNEL_CUTOFF", 1.0))
        base = rho / max(1e-12, cutoff)
        ell0 = float(getattr(cfg, "ELL0", base))
        ell_min = float(getattr(cfg, "ELL_MIN", base))
        base = float(np.clip(base, ell_min, 2.0 * ell0))
    return base


def _reflect_unit_square_np(xy: np.ndarray) -> np.ndarray:
    """Reflect points into [0,1]^2 (less degenerate than hard clipping)."""
    xy = np.asarray(xy, dtype=np.float64)
    # reflect on x
    xy[:, 0] = np.where(xy[:, 0] < 0.0, -xy[:, 0], xy[:, 0])
    xy[:, 0] = np.where(xy[:, 0] > 1.0, 2.0 - xy[:, 0], xy[:, 0])
    # reflect on y
    xy[:, 1] = np.where(xy[:, 1] < 0.0, -xy[:, 1], xy[:, 1])
    xy[:, 1] = np.where(xy[:, 1] > 1.0, 2.0 - xy[:, 1], xy[:, 1])
    return np.clip(xy, 0.0, 1.0)


def _basis_novelty_projection(anchors_np, ell_np, shortlist_idx, cfg):
    """
    Conditioning-controlled local Gram projection novelty.

    For each anchor i in `shortlist_idx`, we estimate how much its gradient-feature
    is already explained by its K nearest neighbors using a local Gram matrix:

        G = <∇v_a, ∇v_b>_q   (quadrature / probe cloud)

    Novelty is:
        novelty_i = sqrt( max(0, 1 - g^T G_nn^{-1} g / G_ii ) )

    Conditioning control layers (meshfree, kernel/RBF stable):
      1) Spectral monitoring (eigs) and condition thresholds
      2) Geometry-aware stabilization via local support shrink (probe radius + ell scale)
      3) Adaptive, scale-aware ridge regularization
      4) Whitening-based projection (proj = ||W g||^2)
    """
    anchors_np = _np.asarray(anchors_np)
    ell_np = _np.asarray(ell_np).reshape((-1, 1))
    shortlist_idx = _np.asarray(shortlist_idx, dtype=_np.int32).reshape((-1,))
    M = int(anchors_np.shape[0])
    if shortlist_idx.size == 0 or M == 0:
        return _np.zeros((0,), dtype=_np.float64)

    # ---- user-tunable knobs (with safe defaults) ----
    K = int(getattr(cfg, "GROWTH_LOCAL_K", 8))
    base_reg = float(getattr(cfg, "GROWTH_PROJ_REG", 1e-6))
    cutoff = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))

    cc_enable = int(getattr(cfg, "CONDCTRL_ENABLE", 1))
    kappa_unstable = float(getattr(cfg, "CONDCTRL_KAPPA_UNSTABLE", 1e8))
    kappa_reject = float(getattr(cfg, "CONDCTRL_KAPPA_REJECT", 1e10))
    shrink_gamma = float(getattr(cfg, "CONDCTRL_SHRINK_GAMMA", 0.9))
    max_shrink = int(getattr(cfg, "CONDCTRL_MAX_SHRINK", 4))
    eps0 = float(getattr(cfg, "CONDCTRL_EPS0", 1e-12))
    alpha_lammax = float(getattr(cfg, "CONDCTRL_ALPHA_LAMMAX", 1e-10))
    alpha_tracemean = float(getattr(cfg, "CONDCTRL_ALPHA_TRACEM", 1e-8))
    do_whiten = int(getattr(cfg, "CONDCTRL_WHITEN", 1))

    ell_min = float(getattr(cfg, "ELL_MIN", 0.0))

    # KDTree for KNN neighborhoods
    tree = cKDTree(anchors_np)
    dnn, inn = tree.query(anchors_np[shortlist_idx], k=min(M, K + 1))

    nq = int(getattr(cfg, "GROWTH_LOC_NQ", 512))
    if nq <= 0:
        nq = 512

    # Deterministic RNG seed (stable across runs)
    seed0 = int(getattr(cfg, "SEED", 2000)) + 9173

    out = _np.ones((shortlist_idx.size,), dtype=_np.float64)

    with tf.device("/CPU:0"):
        for t, i in enumerate(shortlist_idx.tolist()):
            nn_ids = inn[t, 1:]
            local_ids = _np.concatenate(([i], nn_ids.astype(_np.int32, copy=False)), axis=0)
            local_ids = _np.unique(local_ids)
            if local_ids.size < 2:
                out[t] = 1.0
                continue
            if local_ids[0] != i:
                local_ids = _np.concatenate(([i], local_ids[local_ids != i]), axis=0)

            ell_i = float(ell_np[i, 0])
            if ell_i <= 0 or (not _np.isfinite(ell_i)):
                out[t] = 1.0
                continue
            rad0 = 0.5 * cutoff * ell_i

            Xi = tf.constant(anchors_np[local_ids], dtype=DTYPE)
            ell_local_base = _np.maximum(ell_np[local_ids, 0].astype(_np.float64, copy=True),
                                         ell_min if ell_min > 0 else 0.0)

            novelty_i = 1.0
            n_attempts = (max_shrink + 1) if cc_enable else 1

            for s in range(n_attempts):
                scale = (shrink_gamma ** s) if (cc_enable and s > 0) else 1.0
                rad = rad0 * scale

                rng = _np.random.default_rng(seed0 + 1315423911 * int(i) + 10007 * int(s))
                pts = anchors_np[i] + (rng.random((nq, 2)) - 0.5) * (2.0 * rad)
                pts = _reflect_unit_square_np(pts)

                xy = tf.constant(pts, dtype=DTYPE)
                b0, db0 = bubble0_and_grad(xy)

                ell_local = _np.maximum(ell_local_base * scale, ell_min if ell_min > 0 else 0.0)
                ell_loc_tf = tf.constant(ell_local.reshape((-1, 1)), dtype=DTYPE)

                K, Kx, Ky, Kxx, Kyy, Kxy = kernel_eval_precompute(cfg, xy, Xi, ell_loc_tf)
                gvx, gvy, _ = gradv_from_precomputed(cfg, K, Kx, Ky, Kxx, Kyy, Kxy, b0, db0)

                gx = gvx.numpy().astype(_np.float64, copy=False)
                gy = gvy.numpy().astype(_np.float64, copy=False)

                G = (gx.T @ gx + gy.T @ gy) / float(nq)
                G = 0.5 * (G + G.T)

                Gii = float(G[0, 0])
                if (not _np.isfinite(Gii)) or (Gii <= eps0):
                    novelty_i = 1.0
                    break

                if G.shape[0] == 2:
                    g01 = float(G[0, 1])
                    denom = float(G[1, 1] + max(base_reg, eps0))
                    frac = min(1.0, (g01 * g01) / max(eps0, denom * Gii))
                    novelty_i = float(_np.sqrt(max(0.0, 1.0 - frac)))
                    break

                g = G[0, 1:].astype(_np.float64, copy=False)
                Gnn = G[1:, 1:].astype(_np.float64, copy=False)
                Gnn = 0.5 * (Gnn + Gnn.T)

                try:
                    evals_raw = _np.linalg.eigvalsh(Gnn)
                except Exception:
                    evals_raw = None

                if evals_raw is None or evals_raw.size == 0 or (not _np.isfinite(evals_raw).all()):
                    try:
                        sol = _np.linalg.solve(Gnn + max(base_reg, eps0) * _np.eye(Gnn.shape[0]), g)
                        proj = float(g @ sol)
                        frac = min(1.0, proj / max(eps0, Gii))
                        novelty_i = float(_np.sqrt(max(0.0, 1.0 - frac)))
                    except Exception:
                        novelty_i = 1.0
                    break

                lam_min_raw = float(max(evals_raw[0], 0.0))
                lam_max_raw = float(max(evals_raw[-1], lam_min_raw))
                kappa_raw = lam_max_raw / max(eps0, lam_min_raw)

                if cc_enable and (kappa_raw > kappa_reject) and (s < n_attempts - 1):
                    continue

                trace = float(_np.trace(Gnn))
                mnn = int(Gnn.shape[0])
                alpha = max(eps0,
                            lam_max_raw * alpha_lammax,
                            (trace / float(max(1, mnn))) * alpha_tracemean,
                            base_reg)

                if cc_enable and (kappa_raw > kappa_unstable) and (lam_min_raw > 0.0):
                    denom = max(kappa_unstable - 1.0, 1.0)
                    alpha_need = (lam_max_raw - kappa_unstable * lam_min_raw) / denom
                    if alpha_need > 0.0:
                        alpha = max(alpha, alpha_need)

                Ga = Gnn + alpha * _np.eye(mnn, dtype=_np.float64)

                try:
                    wG, Q = _np.linalg.eigh(Ga)
                except Exception:
                    wG, Q = _np.linalg.eigh(0.5 * (Ga + Ga.T))
                wG = _np.maximum(wG, eps0)
                kappa_reg = float(wG[-1] / max(eps0, wG[0]))

                if cc_enable and (kappa_reg > kappa_reject):
                    novelty_i = 0.0
                    break

                if do_whiten:
                    inv_sqrt = 1.0 / _np.sqrt(wG)
                    W = (Q * inv_sqrt) @ Q.T
                    wg = W @ g
                    proj = float(wg @ wg)
                else:
                    sol = _np.linalg.solve(Ga, g)
                    proj = float(g @ sol)

                frac = min(1.0, proj / max(eps0, Gii))
                novelty_i = float(_np.sqrt(max(0.0, 1.0 - frac)))
                break

            if not _np.isfinite(novelty_i):
                novelty_i = 1.0
            out[t] = novelty_i

    return _np.asarray(out, dtype=_np.float64)


def _basis_condition_confidence(anchors_np, ell_np, shortlist_idx, cfg):
    """
    Returns per-anchor spectral confidence in (0,1]:
        conf_i = lam_min / lam_max   (after adaptive ridge)
    using the same local Gram construction/repair logic as _basis_novelty_projection.
    """
    anchors_np = _np.asarray(anchors_np)
    ell_np = _np.asarray(ell_np).reshape((-1, 1))
    shortlist_idx = _np.asarray(shortlist_idx, dtype=_np.int32).reshape((-1,))
    M = int(anchors_np.shape[0])
    if shortlist_idx.size == 0 or M == 0:
        return _np.zeros((0,), dtype=_np.float64)

    K = int(getattr(cfg, "GROWTH_LOCAL_K", 8))
    base_reg = float(getattr(cfg, "GROWTH_PROJ_REG", 1e-6))
    cutoff = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))

    cc_enable = int(getattr(cfg, "CONDCTRL_ENABLE", 1))
    kappa_unstable = float(getattr(cfg, "CONDCTRL_KAPPA_UNSTABLE", 1e8))
    kappa_reject = float(getattr(cfg, "CONDCTRL_KAPPA_REJECT", 1e10))
    shrink_gamma = float(getattr(cfg, "CONDCTRL_SHRINK_GAMMA", 0.9))
    max_shrink = int(getattr(cfg, "CONDCTRL_MAX_SHRINK", 4))
    eps0 = float(getattr(cfg, "CONDCTRL_EPS0", 1e-12))
    alpha_lammax = float(getattr(cfg, "CONDCTRL_ALPHA_LAMMAX", 1e-10))
    alpha_tracemean = float(getattr(cfg, "CONDCTRL_ALPHA_TRACEM", 1e-8))

    ell_min = float(getattr(cfg, "ELL_MIN", 0.0))

    tree = cKDTree(anchors_np)
    dnn, inn = tree.query(anchors_np[shortlist_idx], k=min(M, K + 1))

    nq = int(getattr(cfg, "GROWTH_LOC_NQ", 512))
    if nq <= 0:
        nq = 512
    seed0 = int(getattr(cfg, "SEED", 2000)) + 19173

    out = _np.zeros((shortlist_idx.size,), dtype=_np.float64)

    with tf.device("/CPU:0"):
        for t, i in enumerate(shortlist_idx.tolist()):
            nn_ids = inn[t, 1:]
            local_ids = _np.concatenate(([i], nn_ids.astype(_np.int32, copy=False)), axis=0)
            local_ids = _np.unique(local_ids)
            if local_ids.size < 2:
                out[t] = 0.0
                continue
            if local_ids[0] != i:
                local_ids = _np.concatenate(([i], local_ids[local_ids != i]), axis=0)

            ell_i = float(ell_np[i, 0])
            if ell_i <= 0 or (not _np.isfinite(ell_i)):
                out[t] = 0.0
                continue
            rad0 = 0.5 * cutoff * ell_i

            Xi = tf.constant(anchors_np[local_ids], dtype=DTYPE)
            ell_local_base = _np.maximum(ell_np[local_ids, 0].astype(_np.float64, copy=True),
                                         ell_min if ell_min > 0 else 0.0)

            conf_i = 0.0
            n_attempts = (max_shrink + 1) if cc_enable else 1

            for s in range(n_attempts):
                scale = (shrink_gamma ** s) if (cc_enable and s > 0) else 1.0
                rad = rad0 * scale

                rng = _np.random.default_rng(seed0 + 1315423911 * int(i) + 10007 * int(s))
                pts = anchors_np[i] + (rng.random((nq, 2)) - 0.5) * (2.0 * rad)
                pts = _reflect_unit_square_np(pts)

                xy = tf.constant(pts, dtype=DTYPE)
                b0, db0 = bubble0_and_grad(xy)

                ell_local = _np.maximum(ell_local_base * scale, ell_min if ell_min > 0 else 0.0)
                ell_loc_tf = tf.constant(ell_local.reshape((-1, 1)), dtype=DTYPE)

                K, Kx, Ky, Kxx, Kyy, Kxy = kernel_eval_precompute(cfg, xy, Xi, ell_loc_tf)
                gvx, gvy, divv = gradv_from_precomputed(cfg, K, Kx, Ky, Kxx, Kyy, Kxy, b0, db0)

                gx = gvx.numpy().astype(_np.float64, copy=False)
                gy = gvy.numpy().astype(_np.float64, copy=False)

                gd = divv.numpy().astype(_np.float64, copy=False)

                G = (gx.T @ gx + gy.T @ gy + gd.T @ gd) / float(nq)
                G = 0.5 * (G + G.T)

                if G.shape[0] <= 2:
                    conf_i = 1.0
                    break

                Gnn = G[1:, 1:].astype(_np.float64, copy=False)
                Gnn = 0.5 * (Gnn + Gnn.T)

                try:
                    evals_raw = _np.linalg.eigvalsh(Gnn)
                except Exception:
                    evals_raw = None
                if evals_raw is None or evals_raw.size == 0 or (not _np.isfinite(evals_raw).all()):
                    conf_i = 0.0
                    break

                lam_min_raw = float(max(evals_raw[0], 0.0))
                lam_max_raw = float(max(evals_raw[-1], lam_min_raw))
                kappa_raw = lam_max_raw / max(eps0, lam_min_raw)

                if cc_enable and (kappa_raw > kappa_reject) and (s < n_attempts - 1):
                    continue

                trace = float(_np.trace(Gnn))
                mnn = int(Gnn.shape[0])
                alpha = max(eps0,
                            lam_max_raw * alpha_lammax,
                            (trace / float(max(1, mnn))) * alpha_tracemean,
                            base_reg)

                if cc_enable and (kappa_raw > kappa_unstable) and (lam_min_raw > 0.0):
                    denom = max(kappa_unstable - 1.0, 1.0)
                    alpha_need = (lam_max_raw - kappa_unstable * lam_min_raw) / denom
                    if alpha_need > 0.0:
                        alpha = max(alpha, alpha_need)

                Ga = Gnn + alpha * _np.eye(mnn, dtype=_np.float64)

                try:
                    wG = _np.linalg.eigvalsh(Ga)
                except Exception:
                    wG = _np.linalg.eigvalsh(0.5 * (Ga + Ga.T))
                wG = _np.maximum(wG, eps0)

                if cc_enable and (float(wG[-1] / max(eps0, wG[0])) > kappa_reject):
                    conf_i = 0.0
                    break

                conf_i = float(wG[0] / max(eps0, wG[-1]))
                break

            if not _np.isfinite(conf_i):
                conf_i = 0.0
            out[t] = conf_i

    return _np.asarray(out, dtype=_np.float64)


def _strategy2_fixed_offsets(cm: int) -> np.ndarray:
    """
    Return fixed child-center offsets on the reference square (0,1)^2
    exactly matching Strategy‑2 in fig6_strategy2.py.
    """

    cm = int(cm)
    if cm <= 0:
        raise ValueError(f"ANCHOR_CM must be positive, got {cm}")

    # ---- exact reference patterns ----

    if cm == 4:
        return np.array([
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ], dtype=np.float64)

    if cm == 9:
        return np.array([
            [0.2, 0.2],
            [0.5, 0.2],
            [0.8, 0.2],

            [0.2, 0.5],
            [0.5, 0.5],
            [0.8, 0.5],

            [0.2, 0.8],
            [0.5, 0.8],
            [0.8, 0.8],
        ], dtype=np.float64)

    # ---- generic fallback for other perfect squares ----

    m = int(round(np.sqrt(cm)))
    if m * m != cm:
        raise ValueError(
            f"Strategy 2 requires ANCHOR_CM to be a perfect square, got {cm}"
        )

    xs = np.array([(2 * i + 1) / (2 * m) for i in range(m)], dtype=np.float64)
    grid = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)

    return grid


def _strategy2_children_candidates_np(
    rng: np.random.Generator,
    anchors: np.ndarray,
    ell: np.ndarray,
    marked_ids: np.ndarray,
    cm: int,
    cfg: Config,
    base_ell: float
):
    """
    Strategy‑2 local candidate generation around marked anchors (meshfree, patchless).
    Fixed cm children per marked parent, deterministic support size.
    """
    marked_ids = np.asarray(marked_ids, dtype=np.int32).reshape(-1)
    A = int(marked_ids.shape[0])
    cm = int(max(1, cm))
    if A <= 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
        )

    parents = np.asarray(anchors, dtype=np.float64)[marked_ids]
    ell_p = np.asarray(ell, dtype=np.float64).reshape(-1)[marked_ids]

    parent_rep = np.repeat(np.arange(A, dtype=np.int32), cm)
    parent_anchor_id = marked_ids[parent_rep]

    # --- fixed child layout (deterministic offsets)
    a_ratio = float(getattr(cfg, "ANCHOR_CHILD_A_RATIO", 1.25))
    scale = math.sqrt(max(1e-12, a_ratio / float(cm)))
    cutoff = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))
    rad = (0.5 * scale * cutoff) * ell_p[parent_rep]

    ref = _strategy2_fixed_offsets(cm) - 0.5
    ref_rep = np.tile(ref, (A, 1))
    Z = parents[parent_rep] + ref_rep * rad.reshape(-1, 1)

    # --- deterministic support size (no randomness)
    ell_base = np.minimum(base_ell, ell_p[parent_rep])
    ell_Z = np.maximum(float(getattr(cfg, "ELL_MIN", 1e-6)), ell_base)
    ell_Z = np.minimum(ell_Z, float(base_ell))

    # Reflect children to maintain unit domain bounds
    Z = _reflect_unit_square_np(Z)

    return Z.astype(np.float64), ell_Z.astype(np.float64), parent_anchor_id.astype(np.int32)


def _strategy1_children_candidates_np(rng: np.random.Generator,
                                      anchors: np.ndarray, ell: np.ndarray,
                                      marked_ids: np.ndarray,
                                      cm: int, cfg: Config,
                                      base_ell: float):
    """
    CM-style local candidate generation around marked anchors (meshfree).
    Inspired by patch refine_strategy1: for each marked parent, propose CM*oversample children
    within a local box of size ~sqrt(A_RATIO/CM) * KERNEL_CUTOFF * ell_parent.

    Returns:
      Z:         (Ncand,2) candidate points
      ell_Z:     (Ncand,) candidate ell per point (frozen after acceptance)
      parent_id: (Ncand,) parent anchor id for each candidate
    """
    marked_ids = np.asarray(marked_ids, dtype=np.int32).reshape(-1)
    A = int(marked_ids.shape[0])
    cm = int(max(1, cm))
    overs = int(max(1, getattr(cfg, "ANCHOR_CHILD_OVERSAMPLE", 2)))
    npp = cm * overs
    if A <= 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int32)

    parents = np.asarray(anchors, dtype=np.float64)[marked_ids]  # (A,2)
    ell_p = np.asarray(ell, dtype=np.float64).reshape(-1)[marked_ids]  # (A,)

    # repeat parent ids
    parent_rep = np.repeat(np.arange(A, dtype=np.int32), npp)
    parent_anchor_id = marked_ids[parent_rep]

    # CM geometry (matches fig3_strategy1's scale = sqrt(A_RATIO/CM))
    a_ratio = float(getattr(cfg, "ANCHOR_CHILD_A_RATIO", 1.25))
    scale = math.sqrt(max(1e-12, a_ratio / float(cm)))
    # Spread within parent's effective support; KERNEL_CUTOFF already in the model.
    cutoff = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))
    rad = (0.5 * scale * cutoff) * ell_p[parent_rep]  # half-range per coordinate

    U = rng.random((A * npp, 2), dtype=np.float64) - 0.5  # in [-0.5,0.5]^2
    Z = parents[parent_rep] + U * rad.reshape(-1, 1)

    # candidate ell: local + scheduled + frozen once accepted
    lam_lo = float(getattr(cfg, "ANCHOR_CHILD_LAM_LO", 0.9))
    lam_hi = float(getattr(cfg, "ANCHOR_CHILD_LAM_HI", 10.0 / 9.0))
    lam = rng.uniform(lam_lo, lam_hi, size=(A * npp,)).astype(np.float64)

    ell_base = np.minimum(float(base_ell), ell_p[parent_rep])
    ell_Z = np.maximum(float(getattr(cfg, "ELL_MIN", 1e-6)), ell_base * lam)
    # enforce global schedule upper bound (keeps refinement from re-globalizing early)
    ell_Z = np.minimum(ell_Z, float(base_ell))

    Z = _reflect_unit_square_np(Z)
    return Z.astype(np.float64), ell_Z.astype(np.float64), parent_anchor_id.astype(np.int32)


def add_anchors_strategy1_cm_meshfree(
    unet,
    gnet,
    lift_mode,
    anchors,
    ell,
    aw,
    marked_ids,
    cm,
    add_count,
    child_policy,
    cfg,
    bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
    seed: int, use_eta: bool):
    """
    Strategy1 growth for anchors (CM children around marked anchors), fully meshfree.

    This mirrors the *patch* CM idea from fig3_strategy1.py:
      - mark a small set of parents (done outside)
      - for each marked parent, propose CM children locally
      - accept the best CM per parent using eta=|Δu| on the proposed children
      - enforce Poisson separation with min_sep = MIN_SEP_ALPHA * ell_child

    Nestedness is preserved:
      - old anchors and their ell never change
      - each accepted child gets its own frozen ell_child
    """
    rng = np.random.default_rng(int(seed))
    marked_ids = np.asarray(marked_ids, dtype=np.int32).reshape(-1)
    M = int(anchors.shape[0])
    add_count = int(add_count)
    cm = int(max(1, cm))

    if add_count <= 0 or marked_ids.size == 0:
        return anchors, ell, aw, {"added": 0, "tries": 0, "rejects": 0, "base_ell": float("nan"),
                                  "min_sep": float("nan")}

    # --- schedule ell for *new* anchors only (frozen ell_j for old anchors) ---
    M0 = int(getattr(cfg, "ANCHOR_INIT", max(1, M)))
    base_ell = float(ell_schedule(cfg, M=M + add_count, M0=M0))

    # Decide how many children per parent to actually request (add_count may be clamped by room).
    # Keep parent order as given (already sorted by importance outside).
    A = int(marked_ids.size)
    cm = int(max(1, cm))
    total_full = cm * A

    if add_count >= total_full:
        # We have room / request enough to give each marked parent 'cm' children
        parents_eff = marked_ids
        per_parent = [cm] * A
        add_target = total_full
    else:
        if add_count < cm:
            # All children assigned to the first parent only
            parents_eff = marked_ids[:1]
            per_parent = [add_count]
            add_target = add_count
        else:
            # Some parents get exactly 'cm' children, one more might get the remainder
            A_full = int(add_count // cm)
            remainder = int(add_count - A_full * cm)

            parents_eff = marked_ids[:max(1, A_full + (1 if remainder > 0 else 0))]
            per_parent = [cm] * int(min(A_full, parents_eff.size))

            if remainder > 0 and len(per_parent) < int(parents_eff.size):
                per_parent.append(remainder)

            add_target = int(sum(per_parent))

    # --- NEW: candidate generation moved OUTSIDE the if/else so Z is always defined ---
    grow_policy = str(child_policy).strip().lower()

    if grow_policy == "strategy2":
        Z, ell_Z, _parent_anchor_id = _strategy2_children_candidates_np(
            rng, anchors, ell, parents_eff, cm=cm, cfg=cfg, base_ell=base_ell
        )
    else:
        Z, ell_Z, _parent_anchor_id = _strategy1_children_candidates_np(
            rng, anchors, ell, parents_eff, cm=cm, cfg=cfg, base_ell=base_ell
        )

    N = int(Z.shape[0])
    if N <= 0:
        return anchors, ell, aw, {
            "added": 0,
            "tries": 0,
            "rejects": 0,
            "base_ell": base_ell,
            "min_sep": float(cfg.MIN_SEP_ALPHA) * base_ell,
        }

    # --- eta(z) on candidate points (chunked) ---
    if not bool(use_eta):
        eta = np.ones((N,), dtype=np.float64)
    else:
        lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)
        chunk = int(min(8192, max(1024, getattr(cfg, "EVAL_POINT_BATCH", 8192))))
        laps = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            z_tf = tf.constant(Z[s:e], dtype=DTYPE)
            lap_tf = laplace_u_tf(
                unet, gnet, lift_mode_tf, z_tf,
                float(cfg.LAPLACE_H),
                float(cfg.BUBBLE_SCALE), bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
            )
            laps.append(tf.abs(lap_tf))
        eta = tf.concat(laps, axis=0).numpy().reshape(-1).astype(np.float64)

    # Greedy accept: best children per parent (stable, low-variance)
    existing_tree = cKDTree(anchors) if M > 0 else None
    accepted_pts = []
    accepted_ell = []
    acc_tree = None
    rebuild_every = 128

    tries = 0
    rejects = 0

    # --- DYNAMICALLY DETERMINE npp based on grow_policy ---
    overs = int(max(1, getattr(cfg, "ANCHOR_CHILD_OVERSAMPLE", 2)))
    grow_policy = str(child_policy).strip().lower()  # Get current policy

    # Compute per-parent child count
    if grow_policy == "strategy2":
        npp = cm  # ✅ fixed number of children per parent
    else:
        npp = cm * overs  # strategy1-style oversampling

    for p_idx in range(int(parents_eff.size)):
        want = int(per_parent[p_idx]) if p_idx < len(per_parent) else 0
        if want <= 0:
            continue

        s0 = p_idx * npp
        s1 = min((p_idx + 1) * npp, N)
        if s1 <= s0:
            continue

        # --- Policy-dependent candidate selection ---
        if grow_policy == "strategy2":
            # ✅ For strategy2: children fixed, directly appended; no eta ranking
            local_pts = Z[s0:s1]
            local_ells = ell_Z[s0:s1]
            accepted_pts.extend(local_pts.astype(np.float64))
            accepted_ell.extend(local_ells.astype(np.float64))
            continue  # skip the rest of logic, Strategy2 handled

        # --- For other policies (strategy1, etc.)
        local_eta = eta[s0:s1]
        order = np.argsort(-local_eta)
        local_pts = Z[s0:s1][order]
        local_ells = ell_Z[s0:s1][order]

        got = 0
        for z, ell_c in zip(local_pts, local_ells):
            # --- global stop check (inner loop) ---
            if len(accepted_pts) >= add_target:
                break

            tries += 1
            sep = float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * float(ell_c)

            if existing_tree is not None:
                if float(existing_tree.query(z, k=1)[0]) < sep:
                    rejects += 1
                    continue

            if acc_tree is not None and len(accepted_pts) > 0:
                if float(acc_tree.query(z, k=1)[0]) < sep:
                    rejects += 1
                    continue
            elif len(accepted_pts) > 0:
                pts_np = np.asarray(accepted_pts, dtype=np.float64)
                d2 = np.sum((pts_np - z.reshape(1, 2)) ** 2, axis=1)
                if np.min(d2) < (sep ** 2):
                    rejects += 1
                    continue

            # Accept candidate
            accepted_pts.append(z.astype(np.float64))
            accepted_ell.append(float(ell_c))
            got += 1

            if len(accepted_pts) % rebuild_every == 0:
                acc_tree = cKDTree(np.asarray(accepted_pts, dtype=np.float64))

            if got >= want:
                break  # ✅ breaks z-loop only

        # --- global stop after parent finished ---
        if len(accepted_pts) >= add_target:
            break  # ✅ breaks p_idx-loop (outer)

    # Optional global backfill if we couldn't place enough children due to min-sep/boundary effects.
    fill_global = bool(int(getattr(cfg, "STRATEGY1_FILL_GLOBAL", 1)) == 1)
    if fill_global and len(accepted_pts) < add_target:
        rem = int(add_target - len(accepted_pts))
        if len(accepted_pts) > 0:
            new_pts_np = np.asarray(accepted_pts, dtype=np.float64).reshape(-1, 2)
            new_ell_np = np.asarray(accepted_ell, dtype=np.float64).reshape(-1, 1)
            new_aw_np = np.full((new_pts_np.shape[0], 1), float(cfg.ANCHOR_WEIGHT_NEW), dtype=np.float64)
            anchors_tmp = np.concatenate((anchors, new_pts_np), axis=0)
            ell_tmp = np.concatenate((ell, new_ell_np), axis=0)
            aw_tmp = np.concatenate((aw, new_aw_np), axis=0)
        else:
            anchors_tmp, ell_tmp, aw_tmp = anchors, ell, aw

        cand_np, _ = build_ref_cloud_square(int(cfg.NCAND), seed=int(cfg.CAND_SEED + seed + 1337), cfg=cfg, dim=2)
        anchors_tmp, ell_tmp, aw_tmp, info2 = add_anchors_adaptive(
            unet, gnet, int(lift_mode),
            anchors_tmp, ell_tmp, aw_tmp,
            cand_np,
            add_count=rem,
            cfg=cfg,
            bubble_mode_tf=bubble_mode_tf,
            softmin_tau_tf=softmin_tau_tf,
            bubble_power_tf=bubble_power_tf,
            seed=seed + 99991,
            use_eta=bool(use_eta),
        )
        added_total = int(anchors_tmp.shape[0] - anchors.shape[0])
        return anchors_tmp, ell_tmp, aw_tmp, {
            "added": added_total,
            "tries": tries + int(info2.get("tries", 0)),
            "rejects": rejects + int(info2.get("rejects", 0)),
            "eta_mean": float(np.mean(eta)),
            "eta_p95": float(np.quantile(eta, 0.95)),
            "min_sep": float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * float(base_ell),
            "base_ell": float(base_ell),
            "strategy1_local_added": int(len(accepted_pts)),
            "strategy1_backfill_added": int(info2.get("added", 0)),
        }

    if len(accepted_pts) == 0:
        return anchors, ell, aw, {
            "added": 0,
            "tries": tries,
            "rejects": rejects,
            "eta_mean": float(np.mean(eta)),
            "eta_p95": float(np.quantile(eta, 0.95)),
            "min_sep": float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * float(base_ell),
            "base_ell": float(base_ell),
        }

    new_pts_np = np.asarray(accepted_pts, dtype=np.float64).reshape(-1, 2)
    new_ell_np = np.asarray(accepted_ell, dtype=np.float64).reshape(-1, 1)
    new_aw_np = np.full((new_pts_np.shape[0], 1), float(cfg.ANCHOR_WEIGHT_NEW), dtype=np.float64)

    anchors2 = np.concatenate((anchors, new_pts_np), axis=0)
    ell2 = np.concatenate((ell, new_ell_np), axis=0)
    aw2 = np.concatenate((aw, new_aw_np), axis=0)

    return anchors2, ell2, aw2, {
        "added": int(new_pts_np.shape[0]),
        "tries": tries,
        "rejects": rejects,
        "eta_mean": float(np.mean(eta)),
        "eta_p95": float(np.quantile(eta, 0.95)),
        "min_sep": float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * float(base_ell),
        "base_ell": float(base_ell),
        "strategy1_local_added": int(new_pts_np.shape[0]),
        "strategy1_backfill_added": 0,
    }


def add_anchors_adaptive(unet, gnet, lift_mode: int,
                         anchors: np.ndarray, ell: np.ndarray, aw: np.ndarray,
                         cand_pts: np.ndarray,
                         add_count: int,
                         cfg: Config,
                         bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
                         seed: int, use_eta: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Add anchors WITHOUT destroying nestedness.

    Nestedness rules enforced here:
      1) Existing anchors are never moved.
      2) Existing per-anchor lengthscales ell_j are never modified (frozen ell_j).
      3) Only *new* anchors receive a newly scheduled ell (typically smaller / more local).

    Candidate selection:
      - Score candidates by eta(z)=|Δu_theta(z)| (meshfree strong residual indicator).
      - Sample with probability proportional to (eta+eps)^ETA_POWER.
      - Enforce Poisson-disk separation min_sep = MIN_SEP_ALPHA * ell_new
        to prevent near-duplicate anchors.

    Returns updated (anchors, ell, aw) and diagnostics.
    """
    rng = np.random.default_rng(int(seed))
    add_count = int(add_count)
    if add_count <= 0:
        return anchors, ell, aw, {"added": 0, "tries": 0, "rejects": 0}

    Z = np.asarray(cand_pts, np.float64).reshape(-1, 2)
    N = int(Z.shape[0])
    if N == 0:
        return anchors, ell, aw, {"added": 0, "tries": 0, "rejects": 0}

    # --- eta(z) on candidate points (chunked) ---
    if not bool(use_eta):
        eta = np.ones((N,), dtype=np.float64)
    else:
        lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)
        chunk = int(min(8192, max(1024, getattr(cfg, "EVAL_POINT_BATCH", 8192))))
        laps = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            z_tf = tf.constant(Z[s:e], dtype=DTYPE)
            lap_tf = laplace_u_tf(
                unet, gnet, lift_mode_tf, z_tf,
                float(cfg.LAPLACE_H),
                float(cfg.BUBBLE_SCALE), bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
            )
            laps.append(tf.abs(lap_tf))
        eta = tf.concat(laps, axis=0).numpy().reshape(-1).astype(np.float64)

    w = np.power(eta + float(cfg.ETA_EPS), float(cfg.ETA_POWER))
    sw = float(np.sum(w))
    if (not np.isfinite(sw)) or sw <= 0.0:
        w = np.full((N,), 1.0 / float(N), dtype=np.float64)
    else:
        w = (w / sw).astype(np.float64)

    # --- schedule ell for *new* anchors only (frozen ell_j for old anchors) ---
    M = int(anchors.shape[0])
    M0 = int(getattr(cfg, "ANCHOR_INIT", max(1, M)))
    base_ell = float(ell_schedule(cfg, M=M + add_count, M0=M0))
    min_sep = float(cfg.MIN_SEP_ALPHA) * base_ell

    tree = cKDTree(anchors) if M > 0 else None

    new_pts = []
    tries = 0
    rejects = 0

    # To avoid endless loops when min_sep is too strict, cap tries.
    while len(new_pts) < add_count and tries < int(cfg.REJECT_MAX_TRIES):
        tries += 1
        idx = int(rng.choice(N, p=w))
        z = Z[idx]

        if tree is not None:
            nn = float(tree.query(z, k=1)[0])
            if nn < min_sep:
                rejects += 1
                continue

        ok = True
        for q in new_pts:
            if (z[0] - q[0]) ** 2 + (z[1] - q[1]) ** 2 < (min_sep ** 2):
                ok = False
                break
        if not ok:
            rejects += 1
            continue

        new_pts.append(z)

    if len(new_pts) == 0:
        return anchors, ell, aw, {
            "added": 0,
            "tries": tries,
            "rejects": rejects,
            "eta_mean": float(np.mean(eta)),
            "eta_p95": float(np.quantile(eta, 0.95)),
            "min_sep": float(min_sep),
            "base_ell": float(base_ell),
        }

    new_pts = np.asarray(new_pts, dtype=np.float64).reshape(-1, 2)
    new_M = int(new_pts.shape[0])

    new_ell = np.full((new_M, 1), base_ell, dtype=np.float64)
    if cfg.USE_MULTI_SCALE:
        m_small = int(round(cfg.ELL_MIX_FRAC_SMALL * new_M))
        ell_small = float(
            max(cfg.ELL_MIN, cfg.ELL0_SMALL * (float(M0) / float(max(1, M + new_M))) ** float(cfg.ELL_SCHED_P)))
        new_ell[:m_small, 0] = ell_small
        new_ell[m_small:, 0] = base_ell

    new_aw = np.full((new_M, 1), float(cfg.ANCHOR_WEIGHT_NEW), dtype=np.float64)

    anchors2 = np.concatenate((anchors, new_pts), axis=0)
    ell2 = np.concatenate((ell, new_ell), axis=0)
    aw2 = np.concatenate((aw, new_aw), axis=0)

    return anchors2, ell2, aw2, {
        "added": int(new_M),
        "tries": tries,
        "rejects": rejects,
        "eta_mean": float(np.mean(eta)),
        "eta_p95": float(np.quantile(eta, 0.95)),
        "min_sep": float(min_sep),
        "base_ell": float(base_ell),
    }


# -------------------------
# Training: Adam (GPU-safe, point×anchor minibatching)
# -------------------------
def _zeros_like(vars_):
    return [tf.zeros_like(v) for v in vars_]


def _safe_add_grads(accum, grads):
    out = []
    for a, g in zip(accum, grads):
        out.append(a if g is None else (a + g))
    return out


def adam_train(unet, gnet, lift_mode: int,
               Xq_tf: tf.Tensor,
               anchors_tf: tf.Tensor, ell_tf: tf.Tensor, aw_tf: tf.Tensor,
               cfg: Config,
               bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
               steps: int,
               seed: int):
    steps = int(steps)
    if steps <= 0:
        return

    vars_ = unet.trainable_variables
    rng = np.random.default_rng(int(seed))

    decay_rate = (cfg.LR1 / cfg.LR0) ** (1.0 / float(max(1, steps - 1)))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.LR0,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=False,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    reg_coeff = tf.constant(cfg.LAMBDA_REG, dtype=DTYPE)
    lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)

    Nq = int(Xq_tf.shape[0])
    M = int(anchors_tf.shape[0])
    M0_base = tf.cast(float(getattr(cfg, 'ANCHOR_INIT', 1)), DTYPE)

    pb0 = int(cfg.ADAM_POINT_BATCH)
    ab0 = int(cfg.ADAM_ANCHOR_BATCH)
    ga0 = int(cfg.ADAM_GRAD_ACCUM)

    pb = int(pb0)
    ab = int(ab0)
    ga = int(ga0)

    for _it in range(steps):
        while True:
            try:
                grads_accum = _zeros_like(vars_) if vars_ else []
                for _ in range(ga):
                    pids = rng.integers(0, Nq, size=pb, dtype=np.int32)
                    aids = rng.integers(0, M, size=ab, dtype=np.int32)

                    xy = tf.gather(Xq_tf, pids)  # (pb,2)
                    xi = tf.gather(anchors_tf, aids)  # (ab,2)
                    el = tf.gather(ell_tf, aids)  # (ab,1)
                    aw = tf.gather(aw_tf, aids)  # (ab,1)

                    w_pts = tf.fill([pb, 1], tf.constant(1.0 / float(pb), dtype=DTYPE))

                    with tf.GradientTape() as tape:
                        L, _, _ = loss_on_batches(
                            unet, gnet, lift_mode_tf,
                            xy, w_pts,
                            xi, el, aw,
                            bubble_scale=cfg.BUBBLE_SCALE,
                            bubble_mode=bubble_mode_tf,
                            softmin_tau=softmin_tau_tf,
                            bubble_power=bubble_power_tf,
                            whiten_eps=cfg.WHITEN_EPS,
                            whiten_mode=whiten_mode_tf,
                        )
                        reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) if vars_ else tf.constant(0.0,
                                                                                                               dtype=DTYPE)
                        J = L + reg_coeff * reg

                    g = tape.gradient(J, vars_)
                    if vars_:
                        grads_accum = _safe_add_grads(grads_accum, g)

                if vars_:
                    inv = tf.constant(1.0 / float(ga), dtype=DTYPE)
                    grads_accum = [g * inv for g in grads_accum]
                    clip = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0))
                    if clip and float(clip) > 0.0:
                        grads_accum, _ = tf.clip_by_global_norm(grads_accum, tf.cast(float(clip), DTYPE))
                    opt.apply_gradients(zip(grads_accum, vars_))
                break

            except tf.errors.ResourceExhaustedError:
                # Backoff (dense path): same policy as KDTree path.
                pb = max(256, pb // 2)
                ab = max(1, ab // 2)
                eff0 = int(pb0) * int(ga0)
                ga = int(min(64, max(1, math.ceil(eff0 / max(1, pb)))))
                print(f"[OOM backoff] Adam pb -> {pb}, ab -> {ab}, grad_accum -> {ga}")


# -------------------------
# Full evaluation (Rabs, G, loss) on global QMC points, batched over anchors
# -------------------------
def eval_full(unet, gnet, lift_mode: int,
              Xq_tf: tf.Tensor,
              anchors: np.ndarray, ell: np.ndarray, aw: np.ndarray,
              cfg: Config,
              bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf) -> Dict[str, Any]:
    lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)
    Nq = int(Xq_tf.shape[0])
    w_pts = tf.fill([Nq, 1], tf.constant(1.0 / float(Nq), dtype=DTYPE))

    M = int(anchors.shape[0])
    ab = int(cfg.EVAL_ANCHOR_BATCH)

    Rabs = np.zeros((M,), dtype=np.float64)
    G = np.zeros((M,), dtype=np.float64)
    loss_sum = 0.0
    weight_sum = 0.0

    for s in range(0, M, ab):
        e = min(s + ab, M)
        xi = tf.constant(anchors[s:e], dtype=DTYPE)
        el = tf.constant(ell[s:e], dtype=DTYPE)
        aww = tf.constant(aw[s:e], dtype=DTYPE)

        L, Rabs_b, G_b = loss_on_batches(
            unet, gnet, lift_mode_tf,
            Xq_tf, w_pts,
            xi, el, aww,
            bubble_scale=cfg.BUBBLE_SCALE,
            bubble_mode=bubble_mode_tf,
            softmin_tau=softmin_tau_tf,
            bubble_power=bubble_power_tf,
            whiten_eps=cfg.WHITEN_EPS,
            whiten_mode=whiten_mode_tf,
        )
        Rabs[s:e] = Rabs_b.numpy().reshape(-1)
        G[s:e] = G_b.numpy().reshape(-1)
        loss_sum += float(tf.reduce_sum(L * tf.cast(tf.shape(Rabs_b)[0], DTYPE)).numpy())
        weight_sum += float(e - s)

    loss_mean = loss_sum / max(1.0, weight_sum)
    return {"loss": float(loss_mean), "Rabs": Rabs, "G": G}


# -------------------------
# L-BFGS (optional): deterministic full loss over (global points × all anchors)
# -------------------------
def pack_weights(model: tf.keras.Model) -> np.ndarray:
    return np.concatenate([v.numpy().reshape(-1) for v in model.trainable_variables]).astype(np.float64)


def unpack_weights(model: tf.keras.Model, vec: np.ndarray):
    vec = np.asarray(vec, np.float64).reshape(-1)
    i = 0
    for v in model.trainable_variables:
        s = int(np.prod(v.shape))
        v.assign(vec[i:i + s].reshape(v.shape))
        i += s


def lbfgs_optimize(
        unet, gnet, lift_mode: int,
        Xq_tf, anchors: np.ndarray, ell: np.ndarray, aw: np.ndarray,
        cfg: Config,
        bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
        maxiter: int = 200,
        callback_every: int = 10,
        b0_all_tf=None,
        db0_all_tf=None,
        denom_tf=None,
        cutoff_factor: float = 4.0,
        pruner: Optional[KDTreePruner] = None,
):
    """
    SciPy L-BFGS-B on the nested, whitened objective.

    v15 upgrades:
      1) KDTree pruning for residual (R) when pruner is provided:
         - pruned anchors use edge lists + chunking (same math as cutoff masking, just cheaper)
         - dense anchors fall back to dense batching
      2) OOM backoff: if TF throws ResourceExhaustedError, we automatically reduce
         point/edge batch sizes and retry (so the run doesn't die mid-flight).
    """
    vars_ = unet.trainable_variables
    shapes = [v.shape for v in vars_]
    sizes = [int(tf.size(v)) for v in vars_]
    total_size = int(sum(sizes))

    def pack_vars():
        return np.concatenate([v.numpy().reshape(-1) for v in vars_]).astype(np.float64)

    def unpack_vars(x):
        offset = 0
        for v, sz, sh in zip(vars_, sizes, shapes):
            arr = x[offset:offset + sz].reshape(sh)
            v.assign(arr.astype(np.float64) if v.dtype == tf.float64 else arr.astype(np.float32))
            offset += sz

    # Precompute boundary-zero bubble basis for the test functions.
    # (Fix: precompute_bubbles_tf was a leftover name; bubble0_and_grad is the actual implementation.)
    if b0_all_tf is None or db0_all_tf is None:
        b0_all_tf, db0_all_tf = bubble0_and_grad(Xq_tf)

    cutoff_s = None
    if cutoff_factor and float(cutoff_factor) > 0:
        cutoff_s = tf.constant(float(cutoff_factor) ** 2, dtype=DTYPE)

    anchors_tf = tf.constant(np.asarray(anchors, np.float64), dtype=DTYPE)
    ell_tf = tf.constant(np.asarray(ell, np.float64).reshape(-1, 1), dtype=DTYPE)
    aw_tf = tf.constant(np.asarray(aw, np.float64).reshape(-1, 1), dtype=DTYPE)

    Nq = int(Xq_tf.shape[0])
    M = int(anchors_tf.shape[0])
    invNq = tf.cast(1.0 / float(Nq), DTYPE)
    M0_base = tf.cast(float(getattr(cfg, 'ANCHOR_INIT', 1)), DTYPE)
    lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)

    # If caller didn't pass denom, build it here.
    if denom_tf is None:
        use_energy = (int(whiten_mode_tf.numpy()) == 0)
        eps = tf.cast(float(cfg.WHITEN_EPS), DTYPE)
        if use_energy and int(getattr(cfg, "CACHE_ENERGY", 1)) == 1:
            G_full = compute_G_full_cached(Xq_tf, b0_all_tf, db0_all_tf,
                                           np.asarray(anchors, np.float64),
                                           np.asarray(ell, np.float64).reshape(-1),
                                           cfg,
                                           cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)))
            denom_tf = tf.sqrt(tf.maximum(tf.constant(G_full, dtype=DTYPE), tf.cast(0.0, DTYPE)) + eps)[:, None]
            eps_floor = tf.cast(1e-8, DTYPE)
            denom_tf = tf.maximum(denom_tf, eps_floor)
        else:
            denom_tf = tf.ones((M, 1), dtype=DTYPE)

    # Batch controls (v15: allow explicit LBFGS_* overrides, but keep backward-compatible defaults)
    pbp0 = int(getattr(cfg, "LBFGS_POINT_BATCH", 0) or getattr(cfg, "EVAL_POINT_BATCH", 8192))
    ab0 = int(getattr(cfg, "LBFGS_ANCHOR_BATCH", 0) or getattr(cfg, "EVAL_ANCHOR_BATCH", 64))
    eb0 = int(getattr(cfg, "LBFGS_EDGE_BATCH", 0) or getattr(cfg, "EVAL_EDGE_BATCH", 2048))
    pbp0 = max(8, pbp0)
    ab0 = max(1, ab0)
    eb0 = max(64, eb0)

    pbp_cur = int(pbp0)
    eb_cur = int(eb0)

    # Precompute KDTree blocks (so SciPy's many fun calls don't constantly rebuild edge lists).
    pruned_blocks = []
    dense_blocks = []

    if pruner is not None:
        ids_all = np.arange(M, dtype=np.int32)
        pruned_ids, dense_ids = _split_pruned_dense(ids_all, pruner)

        # pruned blocks (edge lists)
        for s in range(0, int(pruned_ids.shape[0]), ab0):
            e = min(s + ab0, int(pruned_ids.shape[0]))
            ids = pruned_ids[s:e]
            if ids.size == 0:
                continue
            idx_flat, seg_ids = pruner.build_edges_full(ids)
            if idx_flat.size == 0:
                continue
            pruned_blocks.append((
                tf.constant(ids, dtype=tf.int32),
                tf.constant(idx_flat, dtype=tf.int32),
                tf.constant(seg_ids, dtype=tf.int32),
                int(ids.shape[0]),
                int(idx_flat.shape[0]),
            ))

        # dense blocks (fall back to dense batching)
        for s in range(0, int(dense_ids.shape[0]), ab0):
            e = min(s + ab0, int(dense_ids.shape[0]))
            ids = dense_ids[s:e]
            if ids.size == 0:
                continue
            dense_blocks.append(tf.constant(ids, dtype=tf.int32))
    else:
        # all anchors treated as dense
        for s in range(0, M, ab0):
            e = min(s + ab0, M)
            dense_blocks.append(tf.constant(np.arange(s, e, dtype=np.int32), dtype=tf.int32))

    last_fg = {"f": None, "ginf": None}

    def fun_and_jac(x):
        """Memory-stable objective+grad for SciPy L-BFGS-B.

        v15.1 fix:
          - Avoid a single massive GradientTape across all anchors/points.
          - Compute loss in a forward streaming pass (no outer tape).
          - Compute gradient via many small tapes over *chunks* (edge chunks for pruned, point chunks for dense),
            using the exact identity d( (R/Nq)^2 ) = 2*(R/Nq)*(dR/Nq).
        """
        nonlocal pbp_cur, eb_cur
        unpack_vars(x)

        # Retry loop for OOM backoff (rare now, but keep it to avoid hard failure).
        while True:
            try:
                # --- forward pass: compute f (no tape) ---
                f_sum = 0.0
                # We'll accumulate gradients for the data term in numpy float64.
                g_sum = np.zeros((total_size,), dtype=np.float64)

                invNq_f = 1.0 / float(Nq)
                lam = float(cfg.LAMBDA_REG)

                # ---------- KDTree-pruned blocks ----------
                for ids_tf, idx_tf, seg_tf, A, E in pruned_blocks:
                    a_batch = tf.gather(anchors_tf, ids_tf)  # (A,2)
                    e_batch = tf.gather(ell_tf, ids_tf)  # (A,1)
                    awb = tf.gather(aw_tf, ids_tf)  # (A,1)
                    den = tf.gather(denom_tf, ids_tf)  # (A,1)
                    # Safety: some anchors can become effectively "dead" (near-zero H1 energy on the current
                    # quadrature set). In that case denom ~ 0 (or becomes non-finite), which can destroy LBFGS
                    # scaling. We (1) clamp denom away from 0, and (2) explicitly mask out dead anchors so they
                    # contribute exactly zero to r/alpha (better conditioning, avoids NaNs).
                    den_safe = tf.where(tf.math.is_finite(den), den, tf.zeros_like(den))
                    den_safe = tf.maximum(den_safe, tf.cast(1e-12, DTYPE))
                    dead_tol = tf.cast(1e-10, DTYPE)  # drop anchors with denom below this
                    alive = (den_safe[:, 0] >= dead_tol)

                    # Sum of dot products over relevant quadrature points (unscaled).
                    sums = tf.zeros((A,), dtype=DTYPE)
                    for t in range(0, E, eb_cur):
                        idx_c = idx_tf[t:t + eb_cur]
                        seg_c = seg_tf[t:t + eb_cur]

                        xy = tf.gather(Xq_tf, idx_c)
                        b0 = tf.gather(b0_all_tf, idx_c)
                        db0 = tf.gather(db0_all_tf, idx_c)

                        a_edge = tf.gather(a_batch, seg_c)
                        e_edge = tf.gather(e_batch, seg_c)

                        gu = grad_u_tf(
                            unet, gnet, lift_mode_tf, xy,
                            cfg.BUBBLE_SCALE, bubble_mode_tf, softmin_tau_tf, bubble_power_tf
                        )  # (Ec,2)
                        gv = gradv_edges_from_precomputed(
                            xy, b0, db0, a_edge, e_edge, cutoff_s=cutoff_s
                        )  # (Ec,2)

                        dot = tf.reduce_sum(gu * gv, axis=1)  # (Ec,)
                        sums = sums + tf.math.unsorted_segment_sum(dot, seg_c, A)

                    # Mean residual per anchor (scaled by 1/Nq).
                    R = sums * invNq  # (A,)
                    r = (R[:, None] / den_safe)  # (A,1)
                    r = tf.where(alive[:, None], r, tf.zeros_like(r))
                    f_block = tf.reduce_sum(awb * tf.square(r))
                    f_sum += float(f_block.numpy())

                    # alpha_i = 2*aw_i*R_i/(den_i^2) * (1/Nq)
                    # Use clamped den_safe to avoid NaNs/Inf in the LBFGS gradient when some tests have ~zero energy.
                    den2 = tf.square(den_safe)[:, 0]  # (A,)
                    aw_flat = awb[:, 0]  # (A,)
                    alpha_tf = (tf.cast(2.0, DTYPE) * aw_flat * R / tf.maximum(den2, tf.cast(1e-24, DTYPE))) * invNq
                    alpha_tf = tf.where(tf.math.is_finite(alpha_tf), alpha_tf, tf.zeros_like(alpha_tf))
                    alpha_tf = tf.where(alive, alpha_tf, tf.zeros_like(alpha_tf))
                    # --- gradient pass over small edge chunks ---
                    for t in range(0, E, eb_cur):
                        idx_c = idx_tf[t:t + eb_cur]
                        seg_c = seg_tf[t:t + eb_cur]
                        with tf.GradientTape() as tape:
                            xy = tf.gather(Xq_tf, idx_c)
                            b0 = tf.gather(b0_all_tf, idx_c)
                            db0 = tf.gather(db0_all_tf, idx_c)

                            a_edge = tf.gather(a_batch, seg_c)
                            e_edge = tf.gather(e_batch, seg_c)

                            gu = grad_u_tf(
                                unet, gnet, lift_mode_tf, xy,
                                cfg.BUBBLE_SCALE, bubble_mode_tf, softmin_tau_tf, bubble_power_tf
                            )  # (Ec,2)
                            gv = gradv_edges_from_precomputed(
                                xy, b0, db0, a_edge, e_edge, cutoff_s=cutoff_s
                            )  # (Ec,2)

                            dot = tf.reduce_sum(gu * gv, axis=1)  # (Ec,)
                            seg_sum = tf.math.unsorted_segment_sum(dot, seg_c, A)  # (A,)
                            s = tf.reduce_sum(alpha_tf * seg_sum)  # scalar

                        grads = tape.gradient(s, vars_)
                        g_sum += _flatten_grads(grads, vars_)

                # ---------- Dense blocks ----------
                for ids_tf in dense_blocks:
                    xi = tf.gather(anchors_tf, ids_tf)
                    el = tf.gather(ell_tf, ids_tf)
                    awb = tf.gather(aw_tf, ids_tf)
                    den = tf.gather(denom_tf, ids_tf)

                    A = int(xi.shape[0])

                    # Forward: sum residual contributions (unscaled).
                    accR = tf.zeros((A,), dtype=DTYPE)
                    for p in range(0, Nq, pbp_cur):
                        q = min(p + pbp_cur, Nq)
                        xy = Xq_tf[p:q]
                        b0 = b0_all_tf[p:q]
                        db0 = db0_all_tf[p:q]
                        accR = accR + R_block(
                            unet, gnet, lift_mode_tf, xy, b0, db0, xi, el,
                            bubble_scale=cfg.BUBBLE_SCALE,
                            bubble_mode=bubble_mode_tf,
                            softmin_tau=softmin_tau_tf,
                            bubble_power=bubble_power_tf,
                            cutoff_s=cutoff_s
                        )

                    R = accR * invNq
                    den_safe2 = tf.where(tf.math.is_finite(den), den, tf.zeros_like(den))
                    den_safe2 = tf.maximum(den_safe2, tf.cast(1e-12, DTYPE))
                    dead_tol = tf.cast(1e-10, DTYPE)
                    alive = (den_safe2[:, 0] >= dead_tol)
                    r = (R[:, None] / den_safe2)
                    r = tf.where(alive[:, None], r, tf.zeros_like(r))
                    f_block = tf.reduce_sum(awb * tf.square(r))
                    f_sum += float(f_block.numpy())

                    # Robust alpha computation (keep in TF; clamp denom).
                    den2 = tf.square(den_safe2)[:, 0]  # (A,)
                    aw_flat = awb[:, 0]  # (A,)
                    alpha_tf = (tf.cast(2.0, DTYPE) * aw_flat * R / tf.maximum(den2, tf.cast(1e-24, DTYPE))) * invNq
                    alpha_tf = tf.where(tf.math.is_finite(alpha_tf), alpha_tf, tf.zeros_like(alpha_tf))
                    alpha_tf = tf.where(alive, alpha_tf, tf.zeros_like(alpha_tf))

                    # Grad: accumulate over small point chunks (each chunk has its own tape).
                    for p in range(0, Nq, pbp_cur):
                        q = min(p + pbp_cur, Nq)
                        with tf.GradientTape() as tape:
                            xy = Xq_tf[p:q]
                            b0 = b0_all_tf[p:q]
                            db0 = db0_all_tf[p:q]
                            R_chunk = R_block(
                                unet, gnet, lift_mode_tf, xy, b0, db0, xi, el,
                                bubble_scale=cfg.BUBBLE_SCALE,
                                bubble_mode=bubble_mode_tf,
                                softmin_tau=softmin_tau_tf,
                                bubble_power=bubble_power_tf,
                                cutoff_s=cutoff_s
                            )  # (A,)
                            s = tf.reduce_sum(alpha_tf * R_chunk)  # scalar
                        grads = tape.gradient(s, vars_)
                        g_sum += _flatten_grads(grads, vars_)

                # normalize by current effective anchor weight (stable w.r.t. growth)
                w_sum = float(np.sum(aw))
                w_sum = max(1.0, w_sum)
                f_sum = f_sum / w_sum
                g_sum = g_sum / w_sum

                # L2 regularization (match original: reg is NOT divided by M0)
                if lam > 0.0 and vars_:
                    reg_val = 0.0
                    reg_grad_parts = []
                    for v in vars_:
                        arr = v.numpy().reshape(-1).astype(np.float64)
                        reg_val += float(np.sum(arr * arr))
                        reg_grad_parts.append((2.0 * lam) * arr)
                    f_sum += lam * reg_val
                    g_sum += np.concatenate(reg_grad_parts, axis=0) if reg_grad_parts else 0.0

                # cache for logging
                last_fg["f"] = float(f_sum)
                last_fg["ginf"] = float(np.max(np.abs(g_sum))) if g_sum.size else 0.0
                return float(f_sum), g_sum

            except tf.errors.ResourceExhaustedError:
                # Backoff: shrink chunk sizes and retry.
                new_p = max(8, pbp_cur // 2)
                new_e = max(64, eb_cur // 2)
                if new_p == pbp_cur and new_e == eb_cur:
                    raise
                pbp_cur, eb_cur = int(new_p), int(new_e)
                print(f"[L-BFGS][OOM-backoff][stream] reducing batches: point_batch={pbp_cur}, edge_batch={eb_cur}",
                      flush=True)
                continue

    it_counter = {"k": 0}

    def cb(xk):
        it_counter["k"] += 1
        if callback_every and (it_counter["k"] % callback_every == 0):
            f = last_fg["f"]
            gnorm = last_fg["ginf"]
            if f is not None and gnorm is not None:
                print(f"    [L-BFGS iter {it_counter['k']:04d}] f={f:.3e}  ||g||_inf={gnorm:.3e}")

    # Initial parameter vector for SciPy (fix: x0 must exist).
    # Always feed float64 to SciPy even if TF vars are float32.
    x0 = pack_vars()

    # Robust stopping for SciPy L-BFGS-B:
    # - SciPy's default gtol can be too loose for float32, causing premature 'projected gradient' convergence.
    # - Allow override via --lbfgs-gtol, otherwise choose a dtype-aware default.
    gtol0 = float(getattr(cfg, 'LBFGS_GTOL', 0.0))
    gtol = (1e-8 if (DTYPE == tf.float32) else 1e-12) if (gtol0 <= 0.0) else float(gtol0)
    maxfun0 = int(getattr(cfg, 'LBFGS_MAXFUN', 0))
    maxfun = int(maxfun0) if int(maxfun0) > 0 else int(max(200, 20 * int(maxiter)))
    options = dict(maxiter=int(maxiter), maxcor=int(cfg.LBFGS_MAXCOR), ftol=float(cfg.LBFGS_FTOL), gtol=gtol,
                   maxfun=maxfun)
    res = sopt.minimize(
        fun_and_jac,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=cb if callback_every else None,
        options=options,
    )

    # Ensure final params are assigned (scipy might stop at x_best)
    unpack_vars(res.x)

    if hasattr(res, "message"):
        msg = str(res.message)
    else:
        msg = "done"
    print(
        f"[L-BFGS] STOP: {msg}  nit={getattr(res, 'nit', None)} nfev={getattr(res, 'nfev', None)} fun={getattr(res, 'fun', None):.3e}")


def plot_figure3(x_list, h1_list, out_png="figure3_patchless.png", h1_best_list=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=200)
    ax.plot(x_list, h1_list, marker="o", linewidth=1.5, markersize=4, label="holdout")
    if h1_best_list is not None:
        ax.plot(x_list, h1_best_list, linestyle="--", linewidth=1.2, label="best-so-far")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of test functions (anchors)")
    ax.set_ylabel(r"Relative $H^1$ error (holdout)")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_figure3_compare(curves, out_png, title=None):
    """
    curves: list of (x, y, label) where x,y are 1D arrays.
    Saves a single comparison plot (log-x + log-y) similar to fig3_strategy1.py.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 4.2))
    for x, y, label in curves:
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of anchors")
    plt.ylabel("Relative $H^1$ error")
    plt.grid(True, which="both")
    if title is not None:
        plt.title(str(title))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def plot_time_vs_anchors(x_list, t_step_list, out_png="time_vs_anchors_patchless.png", cumulative=True):
    x = np.asarray(x_list, dtype=np.float64)
    t = np.asarray(t_step_list, dtype=np.float64)
    if cumulative:
        y = np.cumsum(t)
        ylabel = "Cumulative wall time [s]"
        label = "cumulative"
    else:
        y = t
        ylabel = "Wall time per outer iteration [s]"
        label = "per-iter"

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=200)
    ax.plot(x, y, marker="o", linewidth=1.5, markersize=4, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Number of test functions (anchors)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main training loop
# -------------------------
def run_patchless(cfg: Config, outdir: str, lift: str, seed_base: int,
                  snapshot_every: int = 1) -> Dict[str, Any]:
    print("\n>>> Starting PATCHLESS Kernelized MF-VPINN (no mesh, no patch cover)")

    tf.random.set_seed(seed_base)
    np.random.seed(seed_base)

    unet = MLP(width=cfg.WIDTH, layers=cfg.L_LAYERS, normalize_input=bool(cfg.DOMAIN_NORMALIZE))
    unet.call(tf.zeros((1, 2), dtype=DTYPE))

    gnet = None
    lift_mode_int = 0
    if lift == "gnet":
        gnet = MLP(width=cfg.WIDTH, layers=cfg.L_LAYERS, normalize_input=bool(cfg.DOMAIN_NORMALIZE))
        gnet.call(tf.zeros((1, 2), dtype=DTYPE))
        train_gnet_boundary(gnet, cfg, seed=seed_base)
        lift_mode_int = 1
    elif lift == "coons":
        lift_mode_int = 2
    else:
        lift_mode_int = 0

    bubble_mode_tf = tf.constant(0 if cfg.BUBBLE_KIND == "product" else 1, dtype=tf.int32)
    softmin_tau_tf = tf.constant(cfg.SOFTMIN_TAU, dtype=DTYPE)
    bubble_power_tf = tf.constant(cfg.BUBBLE_POWER, dtype=DTYPE)
    whiten_mode_tf = tf.constant(0 if str(getattr(cfg, "WHITEN_MODE", "energy")) == "energy" else 1, dtype=tf.int32)
    use_energy = (str(getattr(cfg, "WHITEN_MODE", "energy")) == "energy")

    # Global QMC integration points
    Xq_np, _ = build_ref_cloud_square(cfg.NQ_GLOBAL, seed=seed_base + 11, cfg=cfg, dim=2)
    Xq_tf = tf.constant(Xq_np, dtype=DTYPE)

    # KDTree pruning precompute on the fixed global quadrature cloud (strictly meshfree).
    pruner = None
    if int(getattr(cfg, "USE_KDTREE_PRUNING", 1)) == 1 and float(getattr(cfg, "KERNEL_CUTOFF", 0.0)) > 0.0:
        pruner = KDTreePruner.build(
            Xq_np,
            cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
            leaf_size=int(getattr(cfg, "KDTREE_LEAF", 40)),
            max_local=int(getattr(cfg, "KDTREE_MAX_LOCAL", 0)),
            dense_ratio=float(getattr(cfg, "KDTREE_DENSE_RATIO", 0.75)),
        )

    # Precompute bubble and its gradient on the fixed global quadrature cloud once.
    b0_all_tf, db0_all_tf = bubble0_and_grad(Xq_tf)
    # Frozen whitening energies (nested): computed once for existing anchors; only extended for new anchors.
    G_full = None
    # Candidate pool for adaptive growth
    # (by default we refresh each outer-iteration to avoid saturation of a fixed discrete pool)
    cand_np = None

    # Holdout H1 evaluator (fixed CRN points)
    h1_eval = H1GIQMCEvaluator.build(cfg, cfg.H1_HOLDOUT_SAMPLES, seed=cfg.H1_HOLDOUT_SEED + seed_base)
    h1_ref_energy = h1_eval.reference_energy()

    # Anchors init
    anchors, ell, aw = init_anchors(cfg, seed=seed_base + 99)
    if pruner is not None:
        pruner.init_for_anchors(anchors, ell)
        if int(getattr(cfg, 'KDTREE_REPORT', 1)) == 1:
            print(pruner.summary_str(), flush=True)
    M0 = int(anchors.shape[0])

    # ------------------------------------
    # Strategy3: anchor level state
    # ------------------------------------
    # All initial anchors live at level 0.
    anchor_lvl = np.zeros((M0,), dtype=np.int32)

    # --- New: Strategy3 refinement counters per anchor ---
    # How many times each anchor has been used as parent.
    anchor_refine_count = np.zeros((M0,), dtype=np.int32)

    # Optional: resume from saved state (if enabled and file exists)
    resume_state = bool(int(getattr(cfg, "RESUME_FROM_STATE", 0)))
    if resume_state:
        # try to load state file
        state_path = os.path.join(outdir, "anchor_state_latest.npz")
        if os.path.exists(state_path):
            try:
                with np.load(state_path) as data:
                    anchors_ckpt = np.asarray(data["anchors"], dtype=np.float64)
                    ell_ckpt = np.asarray(data["ell"], dtype=np.float64).reshape(-1)
                    aw_ckpt = np.asarray(data["aw"], dtype=np.float64).reshape(-1)

                    # anchor_lvl may be missing in old state -> fallback
                    if "anchor_lvl" in data.files:
                        anchor_lvl_ckpt = np.asarray(data["anchor_lvl"], dtype=np.int32).reshape(-1)
                    else:
                        anchor_lvl_ckpt = np.zeros((anchors_ckpt.shape[0],), dtype=np.int32)

                    # --- New: anchor_refine_count, optional in older checkpoints ---
                    if "anchor_refine_count" in data.files:
                        anchor_refine_count_ckpt = np.asarray(
                            data["anchor_refine_count"], dtype=np.int32
                        ).reshape(-1)
                    else:
                        anchor_refine_count_ckpt = np.zeros(
                            (anchors_ckpt.shape[0],), dtype=np.int32
                        )

                    # سازگاری ابعادی را چک کن، اگر mismatch بود، resume را نادیده بگیر
                    if (anchors_ckpt.shape[0] == ell_ckpt.shape[0] ==
                            aw_ckpt.shape[0] == anchor_lvl_ckpt.shape[0] ==
                            anchor_refine_count_ckpt.shape[0]):
                        anchors = anchors_ckpt
                        ell = ell_ckpt
                        aw = aw_ckpt
                        anchor_lvl = anchor_lvl_ckpt
                        anchor_refine_count = anchor_refine_count_ckpt
                        M0 = int(anchors.shape[0])
                        print(f"[RESUME] Loaded state from {state_path} (M={M0})", flush=True)
                        # KDTree را هم با انکرهای جدید re-init کن
                        tree = cKDTree(anchors)
            except Exception as e:
                print(f"[RESUME] Failed to load {state_path}: {e}", flush=True)

    best_h1 = float("inf")
    best_weights = pack_weights(unet)

    x_hist = []
    h1_hist = []
    h1_hold_hist = []
    h1_raw_hist = []
    h1_best_hist = []
    loss_hist = []
    time_hist = []
    add_hist = []
    rej_hist = []
    minsep_hist = []
    baseell_hist = []
    snaps = {}

    for it in range(int(cfg.OUTER_ITERS)):
        t0 = time.time()

        h1_after_train = None  # cache H1 after training/guard to avoid re-eval jitter
        # Ramp-in anchor weights

        # -------------------------
        # Homotopy continuation (no rollback): staged ramp + monotone H1 guard
        # -------------------------
        prev_h1 = float(h1_raw_hist[-1]) if (len(h1_raw_hist) > 0) else None

        ramp_stages = int(max(1, int(getattr(cfg, "RAMP_STAGES", 1))))
        ramp_inc = float(getattr(cfg, "ANCHOR_WEIGHT_RAMP", 0.0)) / float(ramp_stages) if ramp_stages > 0 else 0.0

        adam_steps_total = int(cfg.ADAM_STEPS_FIRST if it == 0 else cfg.ADAM_STEPS_AFTER_GROW)
        steps_per_stage = [0] * ramp_stages
        if adam_steps_total > 0:
            base = adam_steps_total // ramp_stages
            rem = adam_steps_total - base * ramp_stages
            for _s in range(ramp_stages):
                steps_per_stage[_s] = base + (1 if _s < rem else 0)
            if sum(steps_per_stage) == 0:
                steps_per_stage[-1] = adam_steps_total

        # --- Energy whitening denom (cached; recompute when anchor count changes) ---
        if use_energy:
            if G_full is None or G_full.shape[0] != anchors.shape[0]:
                if pruner is not None:
                    G_full = compute_G_full_mixed(
                        Xq_tf, b0_all_tf, db0_all_tf,
                        anchors, ell,
                        pruner=pruner, cfg=cfg,
                        cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                    )
                else:
                    G_full = compute_G_full_cached(
                        Xq_tf, b0_all_tf, db0_all_tf,
                        anchors, ell, cfg,
                        cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                    )
            denom_np = np.sqrt(np.maximum(np.asarray(G_full).reshape(-1), 0.0) + float(cfg.WHITEN_EPS)).reshape(-1, 1)
            eps_floor = max(1e-8, float(getattr(cfg, "WHITEN_EPS", 1e-12)) * 1e-2)
            denom_np = np.maximum(denom_np, eps_floor)
        else:
            G_full = None
            denom_np = np.ones((anchors.shape[0], 1), dtype=np.float64)
        denom_tf = tf.constant(denom_np, dtype=DTYPE)

        def _run_adam(steps_local: int, lr_scale: float, seed_local: int):
            if steps_local <= 0:
                return
            lr_scale = float(max(float(lr_scale), 1e-12))
            lr0 = float(cfg.LR0) * lr_scale
            lr1 = float(cfg.LR1) * lr_scale

            anchors_tf = tf.constant(anchors, dtype=DTYPE)
            ell_tf = tf.constant(ell, dtype=DTYPE)
            aw_tf = tf.constant(aw, dtype=DTYPE)

            if pruner is not None:
                adam_train_cached_kdtree(
                    unet, gnet, lift_mode_int,
                    Xq_tf, b0_all_tf, db0_all_tf,
                    anchors_tf, ell_tf, aw_tf, denom_tf,
                    cfg,
                    bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                    steps=int(steps_local),
                    seed=int(seed_local),
                    pruner=pruner,
                    cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                    lr0_override=lr0, lr1_override=lr1,
                    deterministic=int(getattr(cfg, "DETERMINISTIC_BATCHING", 0)),
                )
            else:
                adam_train_cached(
                    unet, gnet, lift_mode_int,
                    Xq_tf, b0_all_tf, db0_all_tf,
                    anchors_tf, ell_tf, aw_tf, denom_tf,
                    cfg,
                    bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                    steps=int(steps_local),
                    seed=int(seed_local),
                    cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                    lr0_override=lr0, lr1_override=lr1,
                    deterministic=int(getattr(cfg, "DETERMINISTIC_BATCHING", 0)),
                )

        # Stage-ramp + train
        for _s in range(ramp_stages):
            if ramp_inc > 0.0:
                aw[:] = np.minimum(1.0, aw + ramp_inc).astype(np.float64)
            _seed_local = seed_base if getattr(cfg, 'DETERMINISTIC_BATCHING', False) else (
                        seed_base + 1000 + it * 17 + _s)
            _run_adam(steps_per_stage[_s], lr_scale=1.0, seed_local=_seed_local)

        # Monotone raw H1 enforcement (no rollback)
        if int(getattr(cfg, "H1_MONO_ENFORCE", 0)) == 1 and (prev_h1 is not None) and (it > 0):
            tol_rel = float(getattr(cfg, "H1_MONO_TOL_REL", 0.0))
            max_phases = int(max(1, int(getattr(cfg, "H1_MONO_MAX_PHASES", 8))))
            extra_steps = int(max(10, int(getattr(cfg, "H1_MONO_EXTRA_STEPS", 200))))
            lr_decay = float(getattr(cfg, "H1_MONO_LR_DECAY", 0.5))
            min_lr_scale = float(getattr(cfg, "H1_MONO_MIN_LR_SCALE", 0.05))
            verbose = int(getattr(cfg, "H1_MONO_VERBOSE", 1))

            target = prev_h1 * (1.0 + tol_rel)
            h1_cur = float(h1_eval.rel_H1(unet, gnet, lift_mode_int,
                                          bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
                                          ref_energy_const=h1_ref_energy))

            phase = 0
            lr_scale = 1.0
            while (h1_cur > target) and (phase < max_phases):
                lr_scale = max(min_lr_scale, lr_scale * lr_decay)
                _seed_local = seed_base if getattr(cfg, 'DETERMINISTIC_BATCHING', False) else (
                            seed_base + 90000 + it * 97 + phase)
                _run_adam(extra_steps, lr_scale=lr_scale, seed_local=_seed_local)
                h1_new = float(h1_eval.rel_H1(unet, gnet, lift_mode_int,
                                              bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
                                              ref_energy_const=h1_ref_energy))
                if verbose:
                    print(
                        f"    [H1-guard] it={it:02d} phase={phase + 1}/{max_phases} lr_scale={lr_scale:.3g} H1={h1_cur:.3e} -> {h1_new:.3e} (target<={target:.3e})",
                        flush=True)
                h1_cur = h1_new
                phase += 1

            if h1_cur > target:
                # HARD monotone mode: do NOT accept a non-monotone step. Keep training (smaller LR) until H1<=target.
                if int(getattr(cfg, "H1_MONO_HARD", 1)) == 1:
                    hard_budget = int(max(0, int(getattr(cfg, "H1_MONO_HARD_BUDGET", 0))))
                    hard_steps = int(max(10, int(getattr(cfg, "H1_MONO_HARD_EXTRA_STEPS", extra_steps))))
                    hard_lr_decay = float(getattr(cfg, "H1_MONO_HARD_LR_DECAY", lr_decay))
                    hard_min_lr_scale = float(getattr(cfg, "H1_MONO_HARD_MIN_LR_SCALE", min_lr_scale))
                    hard_verbose = int(getattr(cfg, "H1_MONO_HARD_VERBOSE", verbose))

                    spent = 0
                    hard_round = 0
                    # Continue decaying LR; do not grow anchors until we satisfy monotonicity.
                    while (h1_cur > target) and (spent < hard_budget):
                        lr_scale = max(hard_min_lr_scale, lr_scale * hard_lr_decay)
                        steps_now = min(hard_steps, hard_budget - spent) if hard_budget > 0 else hard_steps
                        _seed_local = seed_base if getattr(cfg, 'DETERMINISTIC_BATCHING', False) else (
                                    seed_base + 95000 + it * 137 + hard_round)
                        _run_adam(int(steps_now), lr_scale=lr_scale, seed_local=_seed_local)
                        spent += int(steps_now)

                        h1_new = float(h1_eval.rel_H1(unet, gnet, lift_mode_int,
                                                      bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
                                                      ref_energy_const=h1_ref_energy))
                        if hard_verbose:
                            print(
                                f"    [H1-hard] it={it:02d} round={hard_round + 1} spent={spent}/{hard_budget} lr_scale={lr_scale:.3g} H1={h1_cur:.3e} -> {h1_new:.3e} (target<={target:.3e})",
                                flush=True)
                        h1_cur = h1_new
                        hard_round += 1

                    if h1_cur > target:
                        print(
                            f"[STOP] H1 monotone hard-enforce failed: H1={h1_cur:.3e} target<={target:.3e} after spent={spent} steps. Stopping run (no rollback).",
                            flush=True)
                        break
                else:
                    print(
                        f"[WARN] H1 monotone guard failed: H1={h1_cur:.3e} target<={target:.3e}. Continuing refinement (no rollback).",
                        flush=True)
            # cache H1 after training/guard for this iteration (used for logging; avoids re-eval jitter)
            h1_after_train = float(h1_cur)

        # Optional deterministic L-BFGS (FIRST + POLISH ONLY)
        # ---------------------------------------------------
        # Speed-oriented policy:
        #   - Run L-BFGS once on the first outer iteration (after the initial Adam warmup).
        #   - Run L-BFGS once on the final outer iteration as a polish step.
        #   - Skip L-BFGS on intermediate iterations (even if LBFGS_MAXITER_NEXT>0).
        #
        # "Final" means either:
        #   (a) the last planned iteration: it == OUTER_ITERS-1, OR
        #   (b) target anchors already reached at iteration start: M >= TARGET_NANCHOR
        #
        # Note: if FIRST==FINAL (e.g. fixed-M runs), we only run the FIRST stage.
        do_lbfgs = False
        lbfgs_iters = 0
        if int(getattr(cfg, 'H1_MONO_ENFORCE', 0)) == 1:
            # No-rollback monotone policy: skip L-BFGS (can overshoot, cannot be undone without rollback).
            do_lbfgs = False
            lbfgs_iters = 0

        M_now = int(anchors.shape[0])
        targetM = int(cfg.TARGET_NANCHOR) if getattr(cfg, "TARGET_NANCHOR", 0) else None
        is_final_iter = (it == int(cfg.OUTER_ITERS) - 1) or (targetM is not None and M_now >= targetM)

        if it == 0 and int(cfg.LBFGS_MAXITER_FIRST) > 0:
            do_lbfgs = True
            lbfgs_iters = int(cfg.LBFGS_MAXITER_FIRST)
        elif is_final_iter and int(cfg.LBFGS_MAXITER_NEXT) > 0:
            do_lbfgs = True
            lbfgs_iters = int(cfg.LBFGS_MAXITER_NEXT)

        if False and do_lbfgs and lbfgs_iters > 0:
            lbfgs_optimize(
                unet, gnet, lift_mode_int,
                Xq_tf, anchors, ell, aw,
                cfg,
                bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                maxiter=int(lbfgs_iters),
                callback_every=10,
                b0_all_tf=b0_all_tf,
                db0_all_tf=db0_all_tf,
                denom_tf=denom_tf,
                cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                pruner=pruner,
            )

        # Eval loss and holdout H1
        if pruner is not None:
            ev = eval_full_cached_mixed(
                unet, gnet, lift_mode_int,
                Xq_tf, b0_all_tf, db0_all_tf,
                anchors, ell, aw,
                G_full,
                cfg,
                bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                pruner=pruner,
                cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
            )
        else:
            ev = eval_full_cached(
                unet, gnet, lift_mode_int,
                Xq_tf, b0_all_tf, db0_all_tf,
                anchors, ell, aw,
                G_full,
                cfg,
                bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
            )
        h1_raw0 = float(h1_after_train) if (h1_after_train is not None) else float(
            h1_eval.rel_H1(unet, gnet, lift_mode_int,
                           bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
                           ref_energy_const=h1_ref_energy))
        h1_raw = float(h1_raw0)

        if h1_raw < best_h1 * (1.0 - 1e-12):
            best_h1 = float(h1_raw)
            best_weights = pack_weights(unet)

        # Optional: rollback to best weights if the raw H1 worsens (robust monotone trajectory)
        if int(getattr(cfg, "MONOTONE_H1_RESTORE", 0)) == 1:
            tol = float(getattr(cfg, "MONOTONE_H1_TOL", 0.0))
            if h1_raw0 > best_h1 * (1.0 + tol):
                unpack_weights(unet, best_weights)
                h1_raw = float(best_h1)

        # Report either raw or best-so-far (monotone) per user's choice
        if int(getattr(cfg, "MONOTONE_H1_REPORT", 0)) == 1:
            h1_rep = float(best_h1)
        else:
            h1_rep = float(h1_raw)

        M = int(anchors.shape[0])
        x_hist.append(M)
        h1_hist.append(float(h1_raw))
        h1_raw_hist.append(float(h1_raw0))
        h1_hold_hist.append(float(h1_rep))
        h1_best_hist.append(float(best_h1))
        loss_hist.append(float(ev["loss"]))

        dt = float(time.time() - t0)
        time_hist.append(dt)
        print(
            f"Step {it:02d}: M={M:4d} | loss={ev['loss']:.3e} | H1_raw={h1_raw0:.3e} | H1_end={h1_rep:.3e} | H1_best={best_h1:.3e} | ell≈{float(np.median(ell)):.3g} | time[s]={dt:.1f}")

        # per-iteration growth diagnostics (filled after grow; keep alignment even if grow is skipped)
        add_hist.append(0)
        rej_hist.append(float("nan"))
        minsep_hist.append(float("nan"))
        baseell_hist.append(float("nan"))

        # ------------------------------------------------------------
        # Snapshot capture (Fig.4/5-compatible schema)
        # Keys: P{M}_centers, P{M}_h, P{M}_eta
        # ------------------------------------------------------------
        if snapshot_every and (it % int(snapshot_every) == 0):
            # per-anchor residual magnitude (fallback to zeros if missing)
            Rabs = np.asarray(ev.get("Rabs", np.zeros((M,), dtype=np.float64)), dtype=np.float64).reshape(-1)

            # denominator for whitening (matches your strategy1 eta_a logic)
            if str(getattr(cfg, "WHITEN_MODE", "energy")) == "energy":
                denom = np.sqrt(
                    np.maximum(np.asarray(G_full, dtype=np.float64).reshape(-1), 0.0)
                    + float(getattr(cfg, "WHITEN_EPS", 1e-12))
                )
            else:
                denom = np.ones_like(Rabs, dtype=np.float64)

            eta_a = (Rabs / np.maximum(denom, 1e-30)) ** 2  # shape (M,)

            snaps[f"P{M}"] = {
                "centers": np.asarray(anchors, dtype=np.float64).copy(),  # (M,2)
                "h": np.asarray(ell, dtype=np.float64).copy().reshape(-1),  # (M,)
                "eta": np.asarray(eta_a, dtype=np.float64).copy().reshape(-1),  # (M,)
            }

        # stop if reached target
        if cfg.TARGET_NANCHOR and M >= int(cfg.TARGET_NANCHOR):
            break

        # Grow anchors: choose add_count using selected policy.
        room = max(0, int(cfg.TARGET_NANCHOR) - M) if cfg.TARGET_NANCHOR else int(cfg.MAX_ADD_PER_ITER)
        if room <= 0:
            break

        grow_policy = str(getattr(cfg, "ANCHOR_GROW_POLICY", "fraction")).strip().lower()
        child_policy = str(getattr(cfg, "ANCHOR_CHILD_POLICY", grow_policy)).strip().lower()
        mark_stats = None
        marked_ids = None
        add_count = 0

        # (A) Explicit schedule for *total* M (if provided)
        seq = getattr(cfg, "ANCHOR_GROW_SEQ", [])
        if grow_policy == "seq" and seq:
            # Step it uses current M; next target is seq[it+1] if available
            if (it + 1) < len(seq):
                M_target = int(seq[it + 1])
                add_count = max(0, M_target - M)
            else:
                # schedule ended: gracefully fall back to strategy1 marking
                grow_policy = "strategy1"

        # (B) Strategy-1 marking rule (fig3_strategy1-style / extended for strategy3)
        if grow_policy in ("strategy1", "strategy2", "strategy3"):
            # Indicator per existing anchor: normalized residual energy
            Rabs = np.asarray(ev.get("Rabs", np.zeros((M,), dtype=np.float64))).reshape(-1)
            if str(getattr(cfg, "WHITEN_MODE", "energy")) == "energy":
                denom = np.sqrt(
                    np.maximum(np.asarray(G_full).reshape(-1), 0.0) + float(getattr(cfg, "WHITEN_EPS", 1e-12)))
            else:
                denom = np.ones_like(Rabs)

            eta_a = (Rabs / np.maximum(denom, 1e-30)) ** 2

            # ===== growth score: eta magnitude * directional novelty =====
            gsm = str(getattr(cfg, "GROWTH_SCORE_MODE", "eta"))
            alpha = float(getattr(cfg, "GROWTH_ALPHA", 0.7))
            beta = float(getattr(cfg, "GROWTH_BETA", 0.3))
            rho_eps = float(getattr(cfg, "GROWTH_RHO_EPS", 1e-12))

            weights = eta_a
            if (gsm != "eta") and (beta > 0.0):
                # Cheap coefficient-space correlation proxy:
                # r_i = |R_i| / ||phi_i||   with  ||phi_i|| ~ denom_i
                r_coeff = np.sqrt(np.maximum(eta_a, 0.0))
                r_norm = float(np.linalg.norm(r_coeff) + rho_eps)
                novelty = np.clip(r_coeff / r_norm, 0.0, None)

                # desired number of new anchors this iteration (used for shortlist sizing; needed even when gsm != projection)
                nwant = int(min(room, max(int(getattr(cfg, 'MIN_ADD_PER_ITER', 1)),
                                          math.ceil(float(getattr(cfg, 'ADD_FRAC_MAX', 0.0)) * M))))
                nwant = int(min(nwant, int(getattr(cfg, 'MAX_ADD_PER_ITER', nwant))))
                if gsm == "projection":
                    # Local Gram projection novelty on a shortlist only (keeps overhead low)
                    sf = float(getattr(cfg, "GROWTH_SHORTLIST_FACTOR", 4.0))
                    # desired number of new anchors this iteration (used for shortlist sizing)
                    nwant = int(min(room, max(int(getattr(cfg, 'MIN_ADD_PER_ITER', 1)),
                                              math.ceil(float(getattr(cfg, 'ADD_FRAC_MAX', 0.0)) * M))))
                    nwant = int(min(nwant, int(getattr(cfg, 'MAX_ADD_PER_ITER', nwant))))
                    n_short = int(max(1, min(M, math.ceil(sf * float(max(1, nwant))))))
                    shortlist = np.argsort(-eta_a)[:n_short]
                    try:
                        novelty_proj = _basis_novelty_projection(anchors, ell, shortlist, cfg)
                        novelty[shortlist] = np.maximum(novelty_proj, rho_eps)
                    except Exception as _ex:
                        print(f"[GROWTH] WARNING: projection novelty failed; fallback to eta_rho. ({_ex})")

                weights = (np.power(np.maximum(eta_a, 0.0), alpha) *
                           np.power(np.maximum(novelty, rho_eps), beta))

                try:
                    pe = np.percentile(eta_a, [50, 90, 99])
                    ps = np.percentile(weights, [50, 90, 99])
                    print(f"[GROWTH] mode={gsm} eta[p50,p90,p99]={pe} score[p50,p90,p99]={ps}")
                except Exception:
                    pass

            # Conditioning-aware growth control: score *= (lam_min/lam_max)^gamma
            gamma_cond = float(getattr(cfg, "CONDCTRL_GROWTH_GAMMA", 0.0))
            if gamma_cond > 0.0 and int(getattr(cfg, "CONDCTRL_ENABLE", 1)) != 0:
                sf = float(getattr(cfg, "GROWTH_SHORTLIST_FACTOR", 4.0))
                n_short = int(max(1, min(M, _math.ceil(sf * float(max(1, nwant))))))
                shortlist = _np.argsort(-eta_a)[:n_short]
                try:
                    conf_ratio = _basis_condition_confidence(anchors, ell, shortlist, cfg)
                    conf = _np.power(_np.maximum(conf_ratio, rho_eps), gamma_cond)
                    weights[shortlist] *= conf
                except Exception:
                    pass

                    # --- New: Strategy3 level-based penalty on weights (optional) ---
                    if grow_policy == "strategy3":
                        penalty = float(getattr(cfg, "SLG_LEVEL_PENALTY", 0.0))
                        if penalty > 0.0 and M > 0:
                            # فقط اگر weights قبلاً به صورت residual-based ساخته شده باشد
                            if "weights" in locals():
                                # reference level: e.g., max anchor level; higher level => smaller penalty
                                L_ref = int(np.max(anchor_lvl)) if anchor_lvl.size > 0 else 0
                                lvl_gap = (L_ref - anchor_lvl).astype(np.float64)
                                # exp(-penalty * positive gap): penalize low-level anchors
                                weights *= np.exp(-penalty * np.maximum(0.0, lvl_gap))

                                # اختیاری: renormalize برای حفظ interpretation
                                s = float(np.sum(weights))
                                if s > 0.0:
                                    weights /= s
            # marking: smallest set capturing MARK_FRAC of total weight (Doerfler)
            idx = np.argsort(-weights)
            total = float(np.sum(weights))
            if total <= 0.0 or not np.isfinite(total):
                tau_tilde = 1
            else:
                cumsum = np.cumsum(weights[idx])
                tau_tilde = int(np.searchsorted(cumsum, float(getattr(cfg, "MARK_FRAC", 0.75)) * total) + 1)
            tau_cap = int(math.ceil(float(getattr(cfg, "MARK_CAP", 0.30)) * M))
            tau_m = max(1, min(tau_tilde, tau_cap))
            cm = int(getattr(cfg, "ANCHOR_CM", 4))

            # ------------------------------------------------------------------
            # IMPORTANT: do NOT let growth be bottlenecked by tau_m (cm*tau_m).
            # We still compute Doerfler parents (tau_m) for *where* to refine,
            # but the *amount* of refinement should follow the user add-budget:
            #   add_budget ~= clamp( ceil(ADD_FRAC_MAX*M), MIN_ADD, MAX_ADD )
            # Then we *fill parents* up to parents_needed = ceil(add_budget/cm)
            # by taking additional high-eta anchors globally (if enabled).
            # ------------------------------------------------------------------
            add_budget = int(min(cfg.MAX_ADD_PER_ITER, max(cfg.MIN_ADD_PER_ITER, math.ceil(cfg.ADD_FRAC_MAX * M))))
            add_budget = int(min(add_budget, room))
            parents_needed = int(math.ceil(float(add_budget) / float(max(1, cm))))
            parents_needed = max(1, min(M, parents_needed))

            marked_core = idx[:tau_m].astype(np.int32, copy=False)

            # ---------------------------------------------------------
            # Strategy3: level-gap marking (extra Dörfler on low-level)
            # ---------------------------------------------------------
            marked_L = np.zeros((0,), dtype=np.int64)
            if grow_policy == "strategy3":
                # user-tunable level-gap L_max
                L_max = int(getattr(cfg, "ANCHOR_SLG_GAP", 3))

                # mask for low-level anchors
                low_mask = (anchor_lvl <= L_max)
                low_idx = np.nonzero(low_mask)[0]
                M_low = int(low_idx.size)

                if M_low > 0:
                    weights_low = weights[low_mask]
                    idx_low = np.argsort(-weights_low)
                    total_low = float(np.sum(weights_low))
                    if total_low > 0.0:
                        cumsum_low = np.cumsum(weights_low[idx_low])
                        tau_low = int(
                            np.searchsorted(
                                cumsum_low,
                                float(getattr(cfg, "MARK_FRAC_L", 0.50)) * total_low
                            ) + 1
                        )
                        M_low_eff = M_low
                        tau_cap_L = int(math.ceil(
                            float(getattr(cfg, "MARK_CAP_L", 0.20)) * M_low_eff
                        ))
                        tau_L = max(1, min(tau_low, tau_cap_L))
                        local_ids = low_idx[idx_low[:tau_L]]
                        marked_L = local_ids
                    else:
                        marked_L = np.zeros((0,), dtype=np.int64)
                else:
                    marked_L = np.zeros((0,), dtype=np.int64)

                # We still compute marked_ids for diagnostics, but *not* use plain union as parents
                marked_ids = np.unique(
                    np.concatenate([marked_core, marked_L])
                )
            else:
                marked_L = np.zeros((0,), dtype=np.int64)
                marked_ids = marked_core

            # --- New: HARD FILTER for SLG parents / parent_ids selection ---
            if grow_policy == "strategy3":
                # 1) Start from core Doerfler marks as candidates
                # (marked_L is kept only for diagnostics; if you want it as candidates, use marked_ids instead)
                cand_ids = marked_core.copy()

                if cand_ids.size > 0:
                    # 2) Level constraint: only anchors with level >= L_MIN_GROW can spawn children
                    L_min = int(getattr(cfg, "L_MIN_GROW", 0))
                    lvl_ok = (anchor_lvl[cand_ids] >= L_min)

                    # 3) Refine-count constraint: cap number of times an anchor can act as parent
                    max_ref = int(getattr(cfg, "ANCHOR_MAX_REFINES", 1000000))  # very large default
                    refine_ok = (anchor_refine_count[cand_ids] < max_ref)

                    mask_ok = lvl_ok & refine_ok
                    base_set = cand_ids[mask_ok]
                else:
                    base_set = np.array([], dtype=np.int64)

                # Fallback: If filtering removes all candidates, use original core marks
                if base_set.size == 0:
                    base_set = marked_core.copy()
                    # اگر ترجیح بدهی union(core, L) باشد:
                    # base_set = marked_ids.copy()

            else:
                # Strategy1/2: original behavior = core marks
                base_set = marked_core

            # Now, select parents_needed from the filtered base_set
            if base_set.size >= parents_needed:
                parent_ids = base_set[:parents_needed]
            else:
                # SLG should *refuse* to over-refine; do NOT use global fill here.
                parent_ids = base_set.astype(np.int32, copy=False)

            # برای سازگاری با کد موجود که از marked_ids به عنوان والد استفاده می‌کند:
            if grow_policy == "strategy3":
                marked_ids = parent_ids.copy()
            else:
                marked_ids = parent_ids.copy()
            tau_eff = int(marked_ids.size)

            # Final requested add_count (cannot exceed cm*tau_eff by design)
            add_count = int(min(add_budget, cm * max(1, tau_eff)))

            mark_stats = {
                "tau_tilde": tau_tilde,
                "tau_cap": tau_cap,
                "tau_m": tau_m,
                "tau_eff": tau_eff,
                "cm": cm,
                "add_budget": add_budget,
                "parents_needed": parents_needed,
            }

        # (C) Default: old fraction-based add_count
        if grow_policy == "fraction":
            add_max = int(
                min(
                    cfg.MAX_ADD_PER_ITER,
                    max(cfg.MIN_ADD_PER_ITER, math.ceil(cfg.ADD_FRAC_MAX * M)),
                )
            )
            add_count = int(add_max)

        # Final clamp to available room (do NOT force MIN_ADD in seq mode)
        add_count = int(min(add_count, room))
        if add_count <= 0:
            continue

        if marked_ids is not None:
            # CM-style local refinement around marked *existing* anchors (meshfree, patchless).
            anchors, ell, aw, info = add_anchors_strategy1_cm_meshfree(
                unet, gnet, lift_mode_int,
                anchors, ell, aw,
                marked_ids=marked_ids,
                cm=int(getattr(cfg, "ANCHOR_CM", 4)),
                add_count=add_count,
                child_policy=child_policy,
                cfg=cfg,
                bubble_mode_tf=bubble_mode_tf,
                softmin_tau_tf=softmin_tau_tf,
                bubble_power_tf=bubble_power_tf,
                seed=seed_base + 7777 + it,
                use_eta=(str(getattr(cfg, "ADAPTIVE_MODE", "eta")) == "eta"),
            )
        else:
            # Refresh candidate pool each iteration (prevents growth deadlock when Poisson min-sep
            # makes most of a fixed pool unusable)
            cand_seed = (
                int(cfg.CAND_SEED + seed_base)
                if int(getattr(cfg, "FREEZE_CANDIDATES", 0)) == 1
                else int(cfg.CAND_SEED + seed_base + 31 * it)
            )
            cand_np, _ = build_ref_cloud_square(int(cfg.NCAND), seed=cand_seed, cfg=cfg, dim=2)

            anchors, ell, aw, info = add_anchors_adaptive(
                unet, gnet, lift_mode_int,
                anchors, ell, aw,
                cand_np,
                add_count=add_count,
                cfg=cfg,
                bubble_mode_tf=bubble_mode_tf,
                softmin_tau_tf=softmin_tau_tf,
                bubble_power_tf=bubble_power_tf,
                seed=seed_base + 7777 + it,
                use_eta=(str(getattr(cfg, "ADAPTIVE_MODE", "eta")) == "eta"),
            )

        # Update KDTree neighborhoods for newly added anchors (nestedness: old ones stay frozen).
        n_new = int(info.get("added", 0))

        # --- مهم: افزایش شمارنده‌ی والدها در Strategy3 ---
        if grow_policy == "strategy3" and n_new > 0:
            parent_ids_used = info.get("parent_ids", None)
            if parent_ids_used is None:
                parent_ids_used = marked_ids
            if parent_ids_used is not None and len(parent_ids_used) > 0:
                unique_parents = np.unique(parent_ids_used)
                anchor_refine_count[unique_parents] += 1

        # Strategy3: update anchor levels for newly added anchors
        if n_new > 0:
            lvl_new = np.full((n_new,), int(it), dtype=np.int32)
            anchor_lvl = np.concatenate([anchor_lvl, lvl_new], axis=0)

            # refinement counters for new anchors start at 0
            refine_new = np.zeros((n_new,), dtype=np.int32)
            anchor_refine_count = np.concatenate([anchor_refine_count, refine_new], axis=0)

        if pruner is not None and n_new > 0:
            anchors_new = anchors[-n_new:]
            ell_new = ell[-n_new:]
            pruner.append(anchors_new, ell_new)
            if int(getattr(cfg, 'KDTREE_REPORT', 1)) == 1:
                print(pruner.summary_str(), flush=True)

        # Extend frozen G_j / denom for the newly added anchors only (strict nestedness).
        if use_energy and (G_full is not None) and int(getattr(cfg, "NESTED_ENERGY", 1)) == 1:
            if n_new > 0:
                if pruner is not None:
                    M_now = int(anchors.shape[0])
                    new_ids = np.arange(M_now - n_new, M_now, dtype=np.int32)
                    G_new = compute_G_for_anchor_ids_mixed(
                        Xq_tf, b0_all_tf, db0_all_tf,
                        anchors, ell,
                        anchor_ids=new_ids,
                        pruner=pruner, cfg=cfg,
                        cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                    )
                else:
                    anchors_new = anchors[-n_new:]
                    ell_new = ell[-n_new:]
                    G_new = compute_G_full_cached(
                        Xq_tf, b0_all_tf, db0_all_tf,
                        anchors_new, ell_new, cfg,
                        cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                    ).reshape(-1)
                G_full = np.concatenate([np.asarray(G_full).reshape(-1), np.asarray(G_new).reshape(-1)], axis=0)
        rej_rate = info["rejects"] / max(1, info["tries"])
        extra = ""
        if mark_stats is not None:
            extra = f"  mark:(tau~={mark_stats['tau_tilde']},cap={mark_stats['tau_cap']},tau={mark_stats['tau_m']},parents={mark_stats.get('tau_eff', '?')},req={mark_stats.get('add_budget', '?')})"
        print(
            f"         grow[{getattr(cfg, 'ANCHOR_GROW_POLICY', 'fraction')}]: req={add_count} added={info['added']} tries={info['tries']} reject_rate={rej_rate * 100:.1f}%  min_sep={info.get('min_sep', float('nan')):.3g}  base_ell={info.get('base_ell', float('nan')):.3g}  eta_mean={info.get('eta_mean', float('nan')):.2e}  eta_p95={info.get('eta_p95', float('nan')):.2e}{extra}")
        add_hist[-1] = int(info.get("added", 0))
        rej_hist[-1] = float(rej_rate)
        minsep_hist[-1] = float(info.get("min_sep", float("nan")))
        baseell_hist[-1] = float(info.get("base_ell", float("nan")))

        # ------------------------------------------------------------
        # Lightweight checkpoint of anchor state (including anchor_lvl)
        # ------------------------------------------------------------
        if int(getattr(cfg, "SAVE_STATE_EVERY", 0)) > 0:
            every = int(getattr(cfg, "SAVE_STATE_EVERY", 0))
            if (it + 1) % every == 0:
                state_path = os.path.join(outdir, "anchor_state_latest.npz")
                ckpt = {
                    "anchors": anchors,
                    "ell": ell,
                    "aw": aw,
                    "anchor_lvl": anchor_lvl,
                    # --- New ---
                    "anchor_refine_count": anchor_refine_count,
                    "it": np.array([it], dtype=np.int32),
                }
                np.savez_compressed(state_path, **ckpt)

        # Optional: restore best weights after growth (can be enabled for stability, but disabled by default
        # so that reported H1 reflects the actual training trajectory).
        if bool(getattr(cfg, "RESTORE_BEST_AFTER_GROW", False)):
            unpack_weights(unet, best_weights)

    # -------------------------
    # Final L-BFGS polish (once per CM, at the very end)
    # -------------------------
    # User-requested behavior: NEVER run L-BFGS inside the outer loop.
    # Only run one final polish after all growth/training is complete for this CM.
    final_lbfgs_iters = 0
    try:
        _nxt = int(getattr(cfg, "LBFGS_MAXITER_NEXT", 0))
        _fst = int(getattr(cfg, "LBFGS_MAXITER_FIRST", 0))
        final_lbfgs_iters = _nxt if _nxt > 0 else (_fst if _fst > 0 else 0)
    except Exception:
        final_lbfgs_iters = 0

    if final_lbfgs_iters > 0:
        t_pol0 = time.time()
        print(f"\n[FINAL POLISH] Running one L-BFGS (maxiter={final_lbfgs_iters}) at end of CM...", flush=True)
        try:
            lbfgs_optimize(
                unet, gnet, lift_mode_int,
                Xq_tf, anchors, ell, aw,
                cfg,
                bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                maxiter=int(final_lbfgs_iters),
                callback_every=10,
                b0_all_tf=b0_all_tf,
                db0_all_tf=db0_all_tf,
                denom_tf=denom_tf,
                cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                pruner=pruner,
            )
        except Exception as e:
            print(f"[FINAL POLISH] L-BFGS failed: {e}", flush=True)

        # Re-evaluate objective + raw H1 after polish and append one extra log entry
        try:
            if pruner is not None:
                ev_pol = eval_full_cached_mixed(
                    unet, gnet, lift_mode_int,
                    Xq_tf, b0_all_tf, db0_all_tf,
                    anchors, ell, aw,
                    G_full,
                    cfg,
                    bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                    pruner=pruner,
                    cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                )
            else:
                ev_pol = eval_full_cached(
                    unet, gnet, lift_mode_int,
                    Xq_tf, b0_all_tf, db0_all_tf,
                    anchors, ell, aw,
                    G_full,
                    cfg,
                    bubble_mode_tf, softmin_tau_tf, bubble_power_tf, whiten_mode_tf,
                    cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
                )

            h1_raw0_pol = float(h1_eval.rel_H1(
                unet, gnet, lift_mode_int,
                bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
                ref_energy_const=h1_ref_energy
            ))
            # No rollback here: just report and record the new value.
            h1_rep_pol = float(h1_raw0_pol)

            improved = False
            if h1_raw0_pol < best_h1 * (1.0 - 1e-12):
                best_h1 = float(h1_raw0_pol)
                best_weights = pack_weights(unet)
                improved = True

            M_pol = int(anchors.shape[0])
            x_hist.append(M_pol)
            h1_hist.append(float(h1_rep_pol))
            h1_raw_hist.append(float(h1_raw0_pol))
            h1_hold_hist.append(float(h1_rep_pol))
            h1_best_hist.append(float(best_h1))
            loss_hist.append(float(ev_pol["loss"]))
            dt_pol = float(time.time() - t_pol0)
            time_hist.append(dt_pol)
            add_hist.append(0)
            rej_hist.append(float("nan"))
            minsep_hist.append(float("nan"))
            baseell_hist.append(float("nan"))

            print(
                f"[FINAL POLISH] M={M_pol:4d} | loss={ev_pol['loss']:.3e} | H1_raw={h1_raw0_pol:.3e} | best={best_h1:.3e} | improved={improved} | time[s]={dt_pol:.1f}",
                flush=True)
        except Exception as e:
            print(f"[FINAL POLISH] Post-eval failed: {e}", flush=True)

    # Save diagnostics
    np.savez(
        os.path.join(outdir, "diag_patchless.npz"),
        M=np.asarray(x_hist, np.int32),
        H1=np.asarray(h1_hist, np.float64),
        H1_raw=np.asarray(h1_raw_hist, np.float64),
        H1_hold=np.asarray(h1_hold_hist, np.float64),
        H1_best=np.asarray(h1_best_hist, np.float64),
        loss=np.asarray(loss_hist, np.float64),
        time_step_s=np.asarray(time_hist, np.float64),
        time_cum_s=np.cumsum(np.asarray(time_hist, np.float64)),
        added=np.asarray(add_hist, np.int32),
        reject_rate=np.asarray(rej_hist, np.float64),
        min_sep=np.asarray(minsep_hist, np.float64),
        base_ell=np.asarray(baseell_hist, np.float64),
    )
    # --- write snapshots to disk (separate file, used by Fig.4/5 plotters) ---
    save_snapshots_npz(os.path.join(outdir, "snapshots_patchless.npz"), snaps)
    return {"x": x_hist, "h1_raw": h1_raw_hist, "h1": h1_hold_hist, "h1_best": h1_best_hist, "loss": loss_hist,
            "time": time_hist, "snaps": snaps}


def save_snapshots_npz(path, snaps: Dict[str, Any]):
    """
    Save snapshots in Fig.4/5 expected schema:
      P{M}_centers : (M,2)
      P{M}_h       : (M,)
      P{M}_eta     : (M,)
    """
    out = {}
    for k, v in snaps.items():
        # k is like "P{M}"
        out[f"{k}_centers"] = np.asarray(v["centers"], dtype=np.float64)
        out[f"{k}_h"] = np.asarray(v["h"], dtype=np.float64).reshape(-1)
        out[f"{k}_eta"] = np.asarray(v["eta"], dtype=np.float64).reshape(-1)

    np.savez(path, **out)


# -------------------------
# CLI
# -------------------------
def main(argv=None):
    global KERNEL_KIND
    cfg = Config()  # instantiate early so argparse defaults can reference cfg safely
    parser = argparse.ArgumentParser(description="Patchless kernelized MF-VPINN (Figure 6 benchmark)")

    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--outer-iters", type=int, default=12)
    parser.add_argument("--target-nanchor", type=int, default=1000)

    parser.add_argument("--lift", type=str, default="exact", choices=["exact", "gnet", "coons"])
    parser.add_argument("--gnet-steps", type=int, default=20000)

    # net
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--width", type=int, default=50)

    # training
    parser.add_argument("--adam-steps-first", type=int, default=2000)
    parser.add_argument("--adam-steps-after-grow", type=int, default=1200)
    parser.add_argument("--lbfgs-maxiter-first", type=int, default=200)
    parser.add_argument("--lbfgs-maxiter-next", type=int, default=600)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-14)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    parser.add_argument("--lbfgs-maxcor", type=int, default=50)
    parser.add_argument("--lbfgs-maxfun", type=int, default=0)

    parser.add_argument("--adam-point-batch", type=int, default=4096)
    parser.add_argument("--adam-anchor-batch", type=int, default=64)
    parser.add_argument("--adam-grad-accum", type=int, default=16)  # integration / candidate pools
    parser.add_argument("--nq-global", type=int, default=65536)
    parser.add_argument("--ncand", type=int, default=65536)
    parser.add_argument("--nq-factor", type=float, default=4.0, help="ensure NQ_GLOBAL >= nq_factor*target_nanchor")
    parser.add_argument("--domain-normalize", type=int, default=1,
                        help="map NN inputs from [0,1]^d to [-1,1]^d internally")
    parser.add_argument("--ell-from-h", type=int, default=1, help="set base ell from mean NN spacing h")
    parser.add_argument("--ell-c-rho", type=float, default=2.0, help="rho=C_rho*h with rho=cutoff*ell")
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=5.0)
    parser.add_argument("--w-energy", type=float, default=0.1)
    parser.add_argument("--eta-power", type=float, default=1.5)
    parser.add_argument("--laplace-h", type=float, default=1e-3,
                        help="FD step h for Laplacian indicator eta (only used when --adaptive eta).")
    parser.add_argument("--min-sep-alpha", type=float, default=0.7)
    parser.add_argument("--adaptive", type=str, default="eta", choices=["eta", "uniform"],
                        help="eta: adaptive sampling via |Δu|; uniform: disable adaptive weighting (ablation)")

    # kernel lengthscale
    parser.add_argument("--ell0", type=float, default=0.15)
    parser.add_argument("--ell-min", type=float, default=0.015)
    parser.add_argument("--ell-sched-p", type=float, default=0.5)
    parser.add_argument("--whiten-eps", type=float, default=1e-12)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--kernel-cutoff", type=float, default=4.0,
                        help="Ignore kernel contributions when r/ell > C (C~3-5) for speed")
    parser.add_argument("--kernel", type=str, default="gauss", choices=["gauss", "wendland_c2"],
                        help="RBF kernel family. gauss: exp(-(r/ell)^2). wendland_c2: compactly-supported Wendland C2 (polynomial, real sparsity).")
    parser.add_argument("--kdtree-prune", type=int, default=1, choices=[0, 1],
                        help="KDTree pruning: evaluate each test function only on its local quadrature subset (nested, meshfree).")
    parser.add_argument("--kdtree-leaf", type=int, default=40)
    parser.add_argument("--kdtree-max-local", type=int, default=0,
                        help="If a local neighborhood exceeds this many points, fall back to dense evaluation. 0 disables this cap.")
    parser.add_argument("--kdtree-dense-ratio", type=float, default=0.75,
                        help="If |I_j| > dense_ratio*Nq, fall back to dense evaluation to avoid memory blowups.")
    # Stencil bank (pre-sampled per-anchor point indices, avoids Python edge-list build every step)
    parser.add_argument("--use-stencil-bank", type=int, default=1, choices=[0, 1],
                        help="If 1, build a per-anchor stencil-bank once and gather edges in TF (big speed win on CPU).")
    parser.add_argument("--stencil-bank-k", type=int, default=4,
                        help="Number of stencils per anchor (cycled during training).")
    parser.add_argument("--stencil-smax", type=int, default=0,
                        help="Stencil length per anchor. 0 => auto = adam_point_batch//adam_anchor_batch (>=8).")
    parser.add_argument("--stencil-seed", type=int, default=1234, help="Seed used to pre-sample the stencil bank.")
    parser.add_argument("--kdtree-report", type=int, default=1, choices=[0, 1],
                        help="Print KDTree prune/dense neighborhood stats (helps tune dense_ratio/max_local).")
    parser.add_argument("--cache-energy", type=int, default=1, choices=[0, 1],
                        help="Cache G_j once per outer iteration when WHITEN_MODE=energy")
    parser.add_argument("--nested-energy", type=int, default=1,
                        help="If 1, keep old anchor energies G_j fixed and compute G only for newly added anchors.")
    parser.add_argument("--eval-point-batch", type=int, default=8192)
    parser.add_argument("--eval-edge-batch", type=int, default=2048)
    parser.add_argument("--eval-anchor-batch", type=int, default=64)
    parser.add_argument("--lbfgs-point-batch", type=int, default=0,
                        help="If >0, override point batch size inside L-BFGS (else uses --eval-point-batch).")
    parser.add_argument("--lbfgs-edge-batch", type=int, default=0,
                        help="If >0, override edge chunk size inside L-BFGS KDTree path (else uses --eval-edge-batch).")
    parser.add_argument("--lbfgs-anchor-batch", type=int, default=0,
                        help="If >0, override anchor batch size inside L-BFGS (else uses --eval-anchor-batch).")
    parser.add_argument("--whiten-mode", type=str, default="energy", choices=["energy", "none"],
                        help="energy: normalize residual by sqrt(G+eps); none: no normalization (ablation)")

    # growth
    parser.add_argument("--anchor-init", type=int, default=25)
    parser.add_argument("--add-frac-max", type=float, default=1.0)
    parser.add_argument("--min-add-per-iter", type=int, default=16)
    parser.add_argument("--max-add-per-iter", type=int, default=4096)
    parser.add_argument(
        "--anchor-growth-policy", type=str, default="strategy1",
        choices=["fraction", "strategy1", "strategy2", "strategy3", "seq"],
        help=(
            "How many anchors to add each outer step. "
            "'strategy1' uses random CM refinement; "
            "'strategy2' uses fixed CM refinement; "
            "'strategy3' uses small-level-gap marking; "
            "'seq' forces a fixed total-anchor schedule."
        )
    )
    parser.add_argument("--mark-frac", type=float, default=0.75,
                        help="Strategy1 marking fraction (cumulative weight target).")
    parser.add_argument("--mark-cap", type=float, default=0.30,
                        help="Strategy1 marking cap as a fraction of current M.")
    parser.add_argument("--anchor-cm", type=int, default=4,
                        help="Strategy1: number of children anchors per marked anchor.")
    parser.add_argument("--anchor-slg-gap", type=int, default=None,
                        help="Strategy3 small-level gap L_max for low-level anchors (default: Config.ANCHOR_SLG_GAP).")
    parser.add_argument("--mark-frac-l", type=float, default=None,
                        help="Dörfler theta for low-level anchor subset in Strategy3.")
    parser.add_argument("--mark-cap-l", type=float, default=None,
                        help="Cap (fraction of low-level anchors) for Strategy3 low-level marking.")

    # --- New: SLG hard constraints / weighting ---
    parser.add_argument("--l-min-grow", type=int, default=None,
                        help="Minimum anchor level allowed to act as a parent in Strategy3. "
                             "(Default: Config.L_MIN_GROW).")
    parser.add_argument("--slg-level-penalty", type=float, default=None,
                        help="Exponential penalty coefficient for low-level anchors in Strategy3 "
                             "(0 disables penalty).")
    parser.add_argument("--anchor-max-refines", type=int, default=None,
                        help="Maximum number of times a single anchor can be used as parent in Strategy3. "
                             "(Very large value ~= no cap).")
    parser.add_argument(
        "--anchor-child-policy",
        type=str,
        default=cfg.ANCHOR_CHILD_POLICY,
        help="Child anchor refinement policy; e.g. 'strategy2', 'strategy3'. "
             "Defaults to same as ANCHOR_GROW_POLICY if not set explicitly.",
    )
    parser.add_argument("--run-both-cm", type=int, default=1, choices=[0, 1],
                        help="If 1, run multiple CM values sequentially inside the script and make a shared comparison plot.")
    parser.add_argument("--cm-list", type=str, default="4,9",
                        help="Comma-separated CM list used when --run-both-cm=1 (default: 4,9).")
    parser.add_argument("--outer-iters-cm9", type=int, default=10,
                        help="Outer iterations for the CM=9 run when --run-both-cm=1 (keeps a single CLI for both CMs).")
    parser.add_argument("--cm-seed-offset", type=int, default=1000,
                        help="Seed offset added for the 2nd/3rd/... CM run when --run-both-cm=1. Set 0 to reuse the same seed.")
    parser.add_argument("--compare-metric", type=str, default="H1", choices=["H1", "H1_raw", "H1_hold"],
                        help="Which diagnostic series to plot for the CM comparison.")

    parser.add_argument("--anchor-child-a-ratio", type=float, default=1.25,
                        help="Strategy1: geometry spread factor A_RATIO (patch-inspired).")
    parser.add_argument("--anchor-child-lam-lo", type=float, default=0.9,
                        help="Strategy1: lower bound for ell_child multiplier (patch-inspired).")
    parser.add_argument("--anchor-child-lam-hi", type=float, default=(10.0 / 9.0),
                        help="Strategy1: upper bound for ell_child multiplier (patch-inspired).")
    parser.add_argument("--anchor-child-oversample", type=int, default=2,
                        help="Strategy1: propose CM*oversample candidates per marked anchor, then pick best CM.")
    parser.add_argument("--strategy1-fill-global", type=int, default=1, choices=[0, 1],
                        help="If 1, backfill using global adaptive candidates when local CM proposals cannot fill.")

    parser.add_argument("--anchor-growth-seq", type=str, default="",
                        help="Comma-separated total-anchor counts per outer step (e.g. '5,13,21,...'). Used when --anchor-growth-policy=seq; if shorter than outer-iters, it falls back to 'strategy1'.")
    parser.add_argument("--anchor-weight-new", type=float, default=0.0)
    parser.add_argument("--anchor-weight-ramp", type=float, default=0.25)
    parser.add_argument("--reject-max-tries", type=int, default=200000)
    parser.add_argument("--restore-best-after-grow", type=int, default=0,
                        help="If 1, restore best network weights after anchor growth (stability; reporting still uses raw H1).")

    # --- no-rollback monotone H1 (stability pack) ---
    parser.add_argument("--deterministic-batching", type=int, default=1, choices=[0, 1])
    parser.add_argument("--freeze-candidates", type=int, default=1, choices=[0, 1])
    parser.add_argument("--disable-xla", type=int, default=1, choices=[0, 1],
                        help="Disable TF XLA JIT to avoid duplicate-variable XLA crashes")
    parser.add_argument("--ramp-stages", type=int, default=4)
    parser.add_argument("--h1-mono-enforce", type=int, default=1, choices=[0, 1])
    parser.add_argument("--h1-mono-tol", type=float, default=0.0)
    parser.add_argument("--h1-mono-max-phases", type=int, default=8)
    parser.add_argument("--h1-mono-extra-steps", type=int, default=200)
    parser.add_argument("--h1-mono-lr-decay", type=float, default=0.5)
    parser.add_argument("--h1-mono-min-lr-scale", type=float, default=0.05)
    parser.add_argument("--h1-mono-verbose", type=int, default=1, choices=[0, 1])

    # Hard monotone mode (train-only, no rollback): if H1 still fails after guard phases, keep training with smaller LR;
    # if still cannot reach target within budget, stop the run (so logged H1_raw is strictly monotone).
    parser.add_argument("--h1-mono-hard", type=int, default=1, choices=[0, 1])
    parser.add_argument("--h1-mono-hard-budget", type=int, default=4000,
                        help="Max additional Adam steps beyond --h1-mono-max-phases loop.")
    parser.add_argument("--h1-mono-hard-extra-steps", type=int, default=200, help="Adam steps per hard-enforce round.")
    parser.add_argument("--h1-mono-hard-lr-decay", type=float, default=0.5,
                        help="Extra LR decay factor used in hard-enforce mode.")
    parser.add_argument("--h1-mono-hard-min-lr-scale", type=float, default=0.02,
                        help="Minimum LR scale allowed in hard-enforce mode.")
    parser.add_argument("--h1-mono-hard-verbose", type=int, default=1, choices=[0, 1])
    # bubble for u
    parser.add_argument("--bubble", type=str, default="product", choices=["product", "softmin"])
    parser.add_argument("--softmin-tau", type=float, default=1e-3)
    parser.add_argument("--bubble-power", type=float, default=1.0)

    # H1
    parser.add_argument("--h1-samples", type=int, default=65536)
    parser.add_argument("--h1-holdout", type=int, default=32768)
    parser.add_argument("--h1-use-giqmc", type=int, default=1,
                        help="If 1, GI-QMC (graded + importance) H1 estimator. If 0, plain graded QMC.")
    parser.add_argument("--h1-import-alpha", type=float, default=0.75, help="GI-QMC importance exponent alpha.")
    parser.add_argument("--h1-import-eps", type=float, default=1e-10, help="GI-QMC epsilon in q(x) ∝ (eta+eps)^alpha.")
    parser.add_argument("--h1-eta-ema", type=float, default=0.0, help="EMA beta for eta weights (0 disables).")

    parser.add_argument("--h1-freeze-importance", type=int, default=0, choices=[0, 1],
                        help="If 1, freeze GI-QMC importance weights after a few calls so H1 is comparable across iterations.")
    parser.add_argument("--h1-freeze-after-calls", type=int, default=1,
                        help="Freeze importance weights after this many H1 GI-QMC weight builds (>=1).")
    parser.add_argument("--monotone-h1-report", type=int, default=0, choices=[0, 1],
                        help="If 1, store/report best-so-far H1 as H1_hold each step (monotone curve). Raw is saved as H1_raw.")
    parser.add_argument("--monotone-h1-restore", type=int, default=0, choices=[0, 1],
                        help="If 1, when raw H1 worsens by more than tol, restore best network weights (stability).")
    parser.add_argument("--monotone-h1-tol", type=float, default=0.0,
                        help="Relative tolerance for monotone restore: allow raw <= best*(1+tol) without restoring.")
    parser.add_argument("--marking", type=str, default="eta", choices=["eta", "sqrtgamma_eta", "gamma_eta"],
                        help="Marking quantity for adaptive growth (strategy1-style).")

    # ===== strategy1 growth scoring (direction-aware) =====
    parser.add_argument(
        "--growth-score-mode",
        type=str,
        default="eta",
        choices=["eta", "eta_rho", "projection"],
        help="Strategy1 parent scoring: eta (baseline), eta_rho (cheap directional), projection (local Gram projection).",
    )
    parser.add_argument("--growth-alpha", type=float, default=0.7, help="Exponent for eta magnitude in growth score.")
    parser.add_argument("--growth-beta", type=float, default=0.3, help="Exponent for novelty term in growth score.")
    parser.add_argument("--growth-proj-reg", type=float, default=1e-6,
                        help="Regularizer for local Gram solve in projection mode.")
    parser.add_argument("--growth-local-k", type=int, default=8,
                        help="Nearest-neighbor count for local Gram (projection mode).")
    parser.add_argument("--growth-rho-eps", type=float, default=1e-12, help="Epsilon for rho normalization.")
    parser.add_argument("--growth-loc-nq", type=int, default=512,
                        help="Local probe points per shortlisted anchor (projection mode).")
    parser.add_argument("--growth-shortlist-factor", type=float, default=4.0,
                        help="Compute projection novelty only for top (factor * requested_add) anchors.")
    # Conditioning control (Gram stability / whitening)
    parser.add_argument("--condctrl-enable", type=int, default=cfg.CONDCTRL_ENABLE,
                        help="Enable conditioning control for local Gram matrices (0/1)")
    parser.add_argument("--condctrl-kappa-unstable", type=float, default=cfg.CONDCTRL_KAPPA_UNSTABLE,
                        help="Condition number threshold for 'unstable' region")
    parser.add_argument("--condctrl-kappa-reject", type=float, default=cfg.CONDCTRL_KAPPA_REJECT,
                        help="Condition number threshold for hard reject/repair")
    parser.add_argument("--condctrl-shrink-gamma", type=float, default=cfg.CONDCTRL_SHRINK_GAMMA,
                        help="Support shrink factor per repair step (0<gamma<1)")
    parser.add_argument("--condctrl-max-shrink", type=int, default=cfg.CONDCTRL_MAX_SHRINK,
                        help="Max number of support-shrink repair attempts")
    parser.add_argument("--condctrl-eps0", type=float, default=cfg.CONDCTRL_EPS0,
                        help="Minimum numerical floor used in conditioning control")
    parser.add_argument("--condctrl-alpha-lammax", type=float, default=cfg.CONDCTRL_ALPHA_LAMMAX,
                        help="Adaptive ridge term scale relative to lam_max")
    parser.add_argument("--condctrl-alpha-tracemean", type=float, default=cfg.CONDCTRL_ALPHA_TRACEM,
                        help="Adaptive ridge term scale relative to mean(trace)")
    parser.add_argument("--condctrl-whiten", type=int, default=cfg.CONDCTRL_WHITEN,
                        help="Use whitening transform for stable projection (0/1)")
    parser.add_argument("--condctrl-growth-gamma", type=float, default=cfg.CONDCTRL_GROWTH_GAMMA,
                        help="Multiply projection novelty by (lam_min/lam_max)^gamma (0 disables)")

    # misc
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--plot-best", type=int, default=0,
                        help="If 1, also plot best-so-far curve as dashed line (for reference).")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")

    args = parser.parse_args(argv)

    # --- XLA toggle ---
    # Some TF+XLA builds (esp. under WSL/GPU) can crash with:
    #   "Duplicate variable passed to XLA cluster"
    # Disabling JIT avoids that and is often faster/more stable on small VRAM GPUs.
    if getattr(args, "disable_xla", 0) == 1:
        try:
            tf.config.optimizer.set_jit(False)
        except Exception:
            pass
        try:
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
        except Exception:
            pass

    # --- dtype selection (speed vs accuracy) ---
    global DTYPE
    if args.dtype == "float32":
        tf.keras.backend.set_floatx("float32")
        DTYPE = tf.float32
    else:
        tf.keras.backend.set_floatx("float64")
        DTYPE = tf.float64

    # cfg already instantiated above
    cfg.SEED = int(args.seed)
    cfg.OUTER_ITERS = int(args.outer_iters)
    cfg.TARGET_NANCHOR = int(args.target_nanchor)
    # make quadrature/candidate pools large enough for the final anchor count (keeps under-integration away)
    min_nq = int(max(1024, math.ceil(cfg.NQ_FACTOR * cfg.TARGET_NANCHOR)))
    if cfg.NQ_GLOBAL < min_nq:
        cfg.NQ_GLOBAL = min_nq
    if cfg.NCAND < min_nq:
        cfg.NCAND = min_nq

    cfg.L_LAYERS = int(args.layers)
    cfg.WIDTH = int(args.width)

    cfg.GNET_STEPS = int(args.gnet_steps)

    cfg.ADAM_STEPS_FIRST = int(args.adam_steps_first)
    cfg.ADAM_STEPS_AFTER_GROW = int(args.adam_steps_after_grow)
    cfg.LBFGS_MAXITER_FIRST = int(args.lbfgs_maxiter_first)
    cfg.LBFGS_MAXITER_NEXT = int(args.lbfgs_maxiter_next)

    cfg.LBFGS_FTOL = float(args.lbfgs_ftol)
    cfg.LBFGS_GTOL = float(args.lbfgs_gtol)
    cfg.LBFGS_MAXCOR = int(args.lbfgs_maxcor)
    cfg.LBFGS_MAXFUN = int(args.lbfgs_maxfun)

    cfg.ADAM_POINT_BATCH = int(args.adam_point_batch)
    cfg.ADAM_ANCHOR_BATCH = int(args.adam_anchor_batch)
    cfg.ADAM_GRAD_ACCUM = int(args.adam_grad_accum)

    cfg.NQ_GLOBAL = int(args.nq_global)
    cfg.NCAND = int(args.ncand)
    cfg.NQ_FACTOR = float(args.nq_factor)
    cfg.DOMAIN_NORMALIZE = int(args.domain_normalize)
    cfg.ELL_FROM_H = int(args.ell_from_h)
    cfg.ELL_C_RHO = float(args.ell_c_rho)
    cfg.W_PDE = float(args.w_pde)
    cfg.W_BC = float(args.w_bc)
    cfg.W_ENERGY = float(args.w_energy)
    cfg.ETA_POWER = float(args.eta_power)
    cfg.MIN_SEP_ALPHA = float(args.min_sep_alpha)

    cfg.ELL0 = float(args.ell0)
    cfg.ELL_MIN = float(args.ell_min)
    cfg.ELL_SCHED_P = float(args.ell_sched_p)
    cfg.WHITEN_EPS = float(args.whiten_eps)

    cfg.WHITEN_MODE = str(args.whiten_mode)
    cfg.ADAPTIVE_MODE = str(args.adaptive)

    cfg.ANCHOR_INIT = int(args.anchor_init)
    cfg.ADD_FRAC_MAX = float(args.add_frac_max)
    cfg.MIN_ADD_PER_ITER = int(args.min_add_per_iter)
    cfg.MAX_ADD_PER_ITER = int(args.max_add_per_iter)

    cfg.ANCHOR_GROW_POLICY = str(args.anchor_growth_policy)
    cfg.ANCHOR_CHILD_POLICY = str(args.anchor_child_policy)
    cfg.MARK_FRAC = float(args.mark_frac)
    cfg.MARK_CAP = float(args.mark_cap)
    cfg.ANCHOR_CM = int(args.anchor_cm)
    if args.anchor_slg_gap is not None:
        cfg.ANCHOR_SLG_GAP = int(args.anchor_slg_gap)
    if args.mark_frac_l is not None:
        cfg.MARK_FRAC_L = float(args.mark_frac_l)
    if args.mark_cap_l is not None:
        cfg.MARK_CAP_L = float(args.mark_cap_l)

    # --- New: SLG-related config options ---
    if args.l_min_grow is not None:
        cfg.L_MIN_GROW = int(args.l_min_grow)
    if args.slg_level_penalty is not None:
        cfg.SLG_LEVEL_PENALTY = float(args.slg_level_penalty)
    if args.anchor_max_refines is not None:
        cfg.ANCHOR_MAX_REFINES = int(args.anchor_max_refines)

    cfg.ANCHOR_CHILD_A_RATIO = float(args.anchor_child_a_ratio)
    cfg.ANCHOR_CHILD_LAM_LO = float(args.anchor_child_lam_lo)
    cfg.ANCHOR_CHILD_LAM_HI = float(args.anchor_child_lam_hi)
    cfg.ANCHOR_CHILD_OVERSAMPLE = int(args.anchor_child_oversample)
    cfg.STRATEGY1_FILL_GLOBAL = int(args.strategy1_fill_global)

    # Optional explicit schedule: comma-separated list of total anchor counts per outer step.
    # If shorter than outer-iters, we automatically fall back to strategy1 policy for the rest.
    if isinstance(args.anchor_growth_seq, str) and args.anchor_growth_seq.strip():
        _parts = [p.strip() for p in args.anchor_growth_seq.split(',') if p.strip()]
        seq = []
        for p in _parts:
            try:
                seq.append(int(p))
            except Exception:
                pass
        cfg.ANCHOR_GROW_SEQ = seq
        if seq and cfg.ANCHOR_INIT != seq[0]:
            print(
                f"[warn] anchor-init={cfg.ANCHOR_INIT} but anchor-growth-seq starts with {seq[0]}; using {seq[0]} as anchor-init.")
            cfg.ANCHOR_INIT = int(seq[0])
    cfg.ANCHOR_WEIGHT_NEW = float(args.anchor_weight_new)
    cfg.ANCHOR_WEIGHT_RAMP = float(args.anchor_weight_ramp)
    cfg.RESTORE_BEST_AFTER_GROW = bool(int(args.restore_best_after_grow))
    cfg.REJECT_MAX_TRIES = int(args.reject_max_tries)

    cfg.EVAL_POINT_BATCH = int(args.eval_point_batch)
    cfg.EVAL_EDGE_BATCH = int(args.eval_edge_batch)
    cfg.EVAL_ANCHOR_BATCH = int(args.eval_anchor_batch)
    cfg.LBFGS_POINT_BATCH = int(args.lbfgs_point_batch)
    cfg.LBFGS_EDGE_BATCH = int(args.lbfgs_edge_batch)
    cfg.LBFGS_ANCHOR_BATCH = int(args.lbfgs_anchor_batch)
    cfg.KERNEL_CUTOFF = float(args.kernel_cutoff)
    cfg.KERNEL_KIND = str(getattr(args, "kernel", "gauss"))
    KERNEL_KIND = cfg.KERNEL_KIND
    KERNEL_CUTOFF_FACTOR = cfg.KERNEL_CUTOFF
    cfg.USE_KDTREE_PRUNING = int(args.kdtree_prune)
    cfg.KDTREE_LEAF = int(args.kdtree_leaf)
    cfg.KDTREE_MAX_LOCAL = int(args.kdtree_max_local)
    cfg.KDTREE_DENSE_RATIO = float(args.kdtree_dense_ratio)
    cfg.USE_STENCIL_BANK = int(args.use_stencil_bank)
    cfg.STENCIL_BANK_K = int(args.stencil_bank_k)
    cfg.STENCIL_SMAX = int(args.stencil_smax)
    cfg.STENCIL_SEED = int(args.stencil_seed)
    cfg.KDTREE_REPORT = int(args.kdtree_report)
    cfg.CACHE_ENERGY = int(args.cache_energy)
    cfg.NESTED_ENERGY = int(args.nested_energy)

    cfg.BUBBLE_KIND = str(args.bubble)
    cfg.SOFTMIN_TAU = float(args.softmin_tau)
    cfg.BUBBLE_POWER = float(args.bubble_power)

    cfg.DETERMINISTIC_BATCHING = int(args.deterministic_batching)
    cfg.FREEZE_CANDIDATES = int(args.freeze_candidates)
    cfg.RAMP_STAGES = int(args.ramp_stages)
    cfg.H1_MONO_ENFORCE = int(args.h1_mono_enforce)
    cfg.H1_MONO_TOL_REL = float(args.h1_mono_tol)
    cfg.H1_MONO_MAX_PHASES = int(args.h1_mono_max_phases)
    cfg.H1_MONO_EXTRA_STEPS = int(args.h1_mono_extra_steps)
    cfg.H1_MONO_LR_DECAY = float(args.h1_mono_lr_decay)
    cfg.H1_MONO_MIN_LR_SCALE = float(args.h1_mono_min_lr_scale)
    cfg.H1_MONO_VERBOSE = int(args.h1_mono_verbose)

    cfg.H1_MONO_HARD = int(getattr(args, "h1_mono_hard", 1))
    cfg.H1_MONO_HARD_BUDGET = int(getattr(args, "h1_mono_hard_budget", 4000))
    cfg.H1_MONO_HARD_EXTRA_STEPS = int(getattr(args, "h1_mono_hard_extra_steps", cfg.H1_MONO_EXTRA_STEPS))
    cfg.H1_MONO_HARD_LR_DECAY = float(getattr(args, "h1_mono_hard_lr_decay", cfg.H1_MONO_LR_DECAY))
    cfg.H1_MONO_HARD_MIN_LR_SCALE = float(getattr(args, "h1_mono_hard_min_lr_scale", 0.02))
    cfg.H1_MONO_HARD_VERBOSE = int(getattr(args, "h1_mono_hard_verbose", 1))
    cfg.LAPLACE_H = float(args.laplace_h)
    cfg.H1_NSAMPLES = int(args.h1_samples)
    cfg.H1_HOLDOUT_SAMPLES = int(args.h1_holdout)

    # H1 evaluator controls (GI-QMC + robustness)
    cfg.H1_USE_GIQMC = int(args.h1_use_giqmc)
    cfg.H1_IMPORT_ALPHA = float(args.h1_import_alpha)
    cfg.H1_IMPORT_EPS = float(args.h1_import_eps)
    cfg.H1_ETA_EMA_BETA = float(args.h1_eta_ema)
    cfg.H1_FREEZE_IMPORTANCE = int(args.h1_freeze_importance)
    cfg.H1_FREEZE_AFTER_CALLS = int(args.h1_freeze_after_calls)

    # Reporting/stability controls
    cfg.MONOTONE_H1_REPORT = int(args.monotone_h1_report)
    cfg.MONOTONE_H1_RESTORE = int(args.monotone_h1_restore)
    cfg.MONOTONE_H1_TOL = float(args.monotone_h1_tol)

    cfg.MARKING = str(args.marking)

    # growth scoring config (strategy1)
    cfg.GROWTH_SCORE_MODE = str(args.growth_score_mode)
    cfg.GROWTH_ALPHA = float(args.growth_alpha)
    cfg.GROWTH_BETA = float(args.growth_beta)
    cfg.GROWTH_PROJ_REG = float(args.growth_proj_reg)
    cfg.GROWTH_LOCAL_K = int(args.growth_local_k)
    cfg.GROWTH_RHO_EPS = float(args.growth_rho_eps)
    cfg.GROWTH_LOC_NQ = int(args.growth_loc_nq)
    cfg.GROWTH_SHORTLIST_FACTOR = float(args.growth_shortlist_factor)

    # Conditioning control
    cfg.CONDCTRL_ENABLE = int(args.condctrl_enable)
    cfg.CONDCTRL_KAPPA_UNSTABLE = float(args.condctrl_kappa_unstable)
    cfg.CONDCTRL_KAPPA_REJECT = float(args.condctrl_kappa_reject)
    cfg.CONDCTRL_SHRINK_GAMMA = float(args.condctrl_shrink_gamma)
    cfg.CONDCTRL_MAX_SHRINK = int(args.condctrl_max_shrink)
    cfg.CONDCTRL_EPS0 = float(args.condctrl_eps0)
    cfg.CONDCTRL_ALPHA_LAMMAX = float(args.condctrl_alpha_lammax)
    cfg.CONDCTRL_ALPHA_TRACEM = float(args.condctrl_alpha_tracemean)
    cfg.CONDCTRL_WHITEN = int(args.condctrl_whiten)
    cfg.CONDCTRL_GROWTH_GAMMA = float(args.condctrl_growth_gamma)

    outdir = args.outdir or os.getcwd()
    os.makedirs(outdir, exist_ok=True)

    print(f"CWD: {os.getcwd()}")
    print(f"Output dir: {outdir}")
    print(f"Lift: {args.lift}  gnet_steps={int(args.gnet_steps)}")
    print(f"Net: layers={cfg.L_LAYERS} width={cfg.WIDTH}")
    print(
        f"Kernel: {cfg.KERNEL_KIND}  ell0={cfg.ELL0:g} ell_min={cfg.ELL_MIN:g} sched_p={cfg.ELL_SCHED_P:g}  cutoff={cfg.KERNEL_CUTOFF:g}")
    print(
        f"Anchors: init={cfg.ANCHOR_INIT} target={cfg.TARGET_NANCHOR} outer={cfg.OUTER_ITERS} add_frac_max={cfg.ADD_FRAC_MAX:g} min_add={cfg.MIN_ADD_PER_ITER}")
    print(f"QMC: {cfg.QMC_KIND} scramble={cfg.QMC_SCRAMBLE}  NQ_GLOBAL={cfg.NQ_GLOBAL}  NCAND={cfg.NCAND}")
    print(
        f"Adam: pb={cfg.ADAM_POINT_BATCH} ab={cfg.ADAM_ANCHOR_BATCH} grad_accum={cfg.ADAM_GRAD_ACCUM}  steps_first={cfg.ADAM_STEPS_FIRST} steps_after={cfg.ADAM_STEPS_AFTER_GROW}")
    print(f"L-BFGS: first={cfg.LBFGS_MAXITER_FIRST} next={cfg.LBFGS_MAXITER_NEXT}  (set to 0 to disable)")
    print(f"Bubble for u: {cfg.BUBBLE_KIND} tau={cfg.SOFTMIN_TAU:g} power={cfg.BUBBLE_POWER:g}")

    set_seed(1234)
    t_all = time.time()

    # Run both CM values in one go (like fig3_strategy1.py), to get a single joint plot.
    if int(getattr(args, "run_both_cm", 0)) == 1:
        cm_list_str = str(getattr(args, "cm_list", "4,9"))
        cm_list = [int(x) for x in cm_list_str.split(",") if x.strip() != ""]
        if len(cm_list) == 0:
            cm_list = [4, 9]

        seed0 = int(getattr(args, "seed", 0))
        seed_off = int(getattr(args, "cm_seed_offset", 1000))

        results = []  # (cm, outdir_cm)
        for j, cm in enumerate(cm_list):
            outdir_cm = os.path.join(outdir, f"CM{cm}")
            os.makedirs(outdir_cm, exist_ok=True)

            cfg_cm = copy.deepcopy(cfg)
            cfg_cm.ANCHOR_CM = int(cm)

            # Allow CM=9 to optionally have different outer_iters, but keep everything else identical.
            if (cm == 9) and int(getattr(args, "outer_iters_cm9", 0)) > 0:
                cfg_cm.OUTER_ITERS = int(args.outer_iters_cm9)
            else:
                cfg_cm.OUTER_ITERS = int(args.outer_iters)

            cfg_cm.SEED = seed0 + j * seed_off

            print("=" * 88)
            print(f"RUN: CM={cm}  outer_iters={cfg_cm.OUTER_ITERS}  seed={cfg_cm.SEED}")
            print(f"OUT: {outdir_cm}")

            _ = run_patchless(
                cfg_cm,
                outdir_cm,
                lift=str(args.lift),
                seed_base=cfg_cm.SEED,
                snapshot_every=int(args.snapshot_every),
            )
            results.append((cm, outdir_cm))

            # Free TF graphs/GPU memory before next CM.
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()

        # Build a shared plot from diag_patchless.npz files.
        if not args.no_plot:
            try:
                curves = []
                for cm, od in results:
                    fn = os.path.join(od, "diag_patchless.npz")
                    d = np.load(fn)
                    # Expect one monotone-accepted H1 per step in d["H1"].
                    curves.append((d["M"], d["H1"], rf"$C_M={cm}$"))
                out_png = os.path.join(outdir, "figure6_patchless.png")
                plot_figure3_compare(curves, out_png=out_png, title="PATCHLESS MF-VPINN")
                print(f"Saved plot: {out_png}")
            except Exception as e:
                print(f"[WARN] Could not build shared plot: {e}")
        else:
            print("Plot disabled (--no-plot).")

        print(f"Elapsed [s]: {time.time() - t_all:.2f}")
        return

    # Single-run mode (one CM only)
    _ = run_patchless(cfg, outdir, lift=str(args.lift), seed_base=cfg.SEED, snapshot_every=int(args.snapshot_every))
    print(f"Elapsed [s]: {time.time() - t_all:.2f}")


if __name__ == "__main__":
    main()