#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
patchless_fig17_problem35_strat12.py

Patchless / anchor-based reproduction for Problem (35) in Sec. 3.4:
- runs a common two-stage base construction in Ω₂ (square with 4 rectangular holes),
- branches into Strategy #1 and Strategy #2 for C_M = 4 and C_M = 9,
- saves the Figure 17-style convergence plot,
- saves the final anchor snapshots needed by patchless_fig18_problem35.py.

Design principles:
- Reuse the kernelized patchless MF-VPINN machinery from patchless_mfvpinn_strategy2.py
  whenever it remains mathematically correct for Problem (35).
- Override only the problem-dependent pieces:
    * domain sampler Ω₂,
    * boundary ADF bubble φ on ∂Ω₂,
    * Poisson weak residual  ∫Ω₂ (∇u·∇v - f v),
    * strong residual indicator via autograd Laplacian,
    * local child proposals that stay outside the holes.
- Keep the Figure 17/18 plotting schema as close as possible to the original patch-based code.

Output files:
  - figure17_problem35_patchless.png
  - figure17_problem35_patchless_curves.npz
  - figure18_problem35_patchless.npz

Notes:
- We intentionally disable KDTree-pruned residual assembly here, because the original pruned
  fast-path in the base patchless code is specialized to the homogeneous Laplace benchmark and
  does not include the source term f. The dense path is mathematically correct for Problem (35).
- Dirichlet data for Problem (35) are homogeneous on the full boundary ∂Ω₂, so the trial space
  is simply u_θ = φ w_θ with φ the ADF boundary factor.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import patchless_mfvpinn_strategy2 as pf
import fig17_problem35_strat12 as p35


# -----------------------------------------------------------------------------
# TensorFlow runtime
# -----------------------------------------------------------------------------
try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

DTYPE = pf.DTYPE


def apply_tf_optimizer_options_from_args(args: argparse.Namespace) -> None:
    opts = {
        "disable_meta_optimizer": bool(int(getattr(args, "disable_meta_optimizer", 0))),
        "arithmetic_optimization": not bool(int(getattr(args, "disable_arithmetic_optimizer", 0))),
        "dependency_optimization": not bool(int(getattr(args, "disable_dependency_optimizer", 0))),
    }
    try:
        tf.config.optimizer.set_experimental_options(opts)
    except Exception as exc:
        print(f"[warn] could not set TF optimizer experimental options: {exc}", flush=True)


# -----------------------------------------------------------------------------
# Small config for ADF / H1 utilities borrowed from the patch-based Problem (35)
# -----------------------------------------------------------------------------
@dataclass
class Problem35AuxConfig:
    ADF_M: int = 2
    ADF_EPS: float = 1e-12
    H1_NCELLS: int = 221
    H1_GAUSS_N: int = 4
    H1_EVAL_CHUNK: int = 65536


# -----------------------------------------------------------------------------
# Domain helpers for Ω₂
# -----------------------------------------------------------------------------
class Problem35Domain:
    def __init__(self, aux_cfg: Problem35AuxConfig):
        self.aux_cfg = aux_cfg
        self.holes = p35.holes_problem35()
        self.segments = p35.boundary_segments_omega2(self.holes)
        self.phi_and_grad = p35.build_phi_adf_tf(self.segments, aux_cfg)

        hole_area = 0.0
        hole_boxes = []
        for H in self.holes:
            xmin, xmax, ymin, ymax = p35.hole_bounds(H)
            hole_boxes.append([xmin, ymin, xmax, ymax])
            hole_area += (xmax - xmin) * (ymax - ymin)
        self.holes_xyxy = np.asarray(hole_boxes, dtype=np.float64)
        self.area = float(1.0 - hole_area)

    def point_in_holes(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        if xy.size == 0:
            return np.zeros((0,), dtype=bool)
        return p35.point_in_holes(xy[:, 0], xy[:, 1], self.holes)

    def sample_domain_qmc(self, n: int, seed: int, cfg: pf.Config) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw n QMC points in Ω₂ by rejection from the original square sampler.
        We keep the same sampling style as the base patchless code and only filter the holes.
        """
        n = int(max(1, n))
        pts_keep: List[np.ndarray] = []
        total = 0
        local_seed = int(seed)

        # Rejection overhead is tiny here because |Ω₂| = 217/221 ≈ 0.9819.
        while total < n:
            need = n - total
            m = max(int(math.ceil(1.10 * need / max(self.area, 1e-12))) + 64, need + 32)
            pts_sq, _ = ORIG_BUILD_REF_CLOUD(m, seed=local_seed, cfg=cfg, dim=2)
            local_seed += 97
            mask = ~self.point_in_holes(pts_sq)
            pts_ok = np.asarray(pts_sq[mask], dtype=np.float64)
            if pts_ok.size > 0:
                pts_keep.append(pts_ok)
                total += int(pts_ok.shape[0])

        pts = np.concatenate(pts_keep, axis=0)[:n]
        w = np.full((n,), 1.0 / float(n), dtype=np.float64)
        return pts.astype(np.float64), w

    def project_points_outside_holes(self, xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Reflect points to [0,1]^2 first, then if a point falls inside a hole,
        push it outside through the nearest hole side by a tiny epsilon.
        """
        pts = pf._reflect_unit_square_np(np.asarray(xy, dtype=np.float64).copy())
        eps = float(max(eps, 1e-12))

        if pts.size == 0:
            return pts

        for i in range(int(pts.shape[0])):
            x = float(pts[i, 0])
            y = float(pts[i, 1])

            for _ in range(8):
                moved = False
                for H in self.holes:
                    xmin, xmax, ymin, ymax = p35.hole_bounds(H)
                    if (xmin < x < xmax) and (ymin < y < ymax):
                        d_left = x - xmin
                        d_right = xmax - x
                        d_bot = y - ymin
                        d_top = ymax - y
                        side = int(np.argmin([d_left, d_right, d_bot, d_top]))
                        if side == 0:
                            x = xmin - eps
                        elif side == 1:
                            x = xmax + eps
                        elif side == 2:
                            y = ymin - eps
                        else:
                            y = ymax + eps
                        x = float(np.clip(x, 0.0, 1.0))
                        y = float(np.clip(y, 0.0, 1.0))
                        moved = True
                        break
                if not moved:
                    break

            pts[i, 0] = x
            pts[i, 1] = y

        pts = pf._reflect_unit_square_np(pts)
        return pts

    @staticmethod
    def distance_to_outer_boundary(xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.minimum(
            np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
            np.minimum(pts[:, 1], 1.0 - pts[:, 1]),
        )

    def distance_to_holes(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0,), dtype=np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        best = np.full((pts.shape[0],), np.inf, dtype=np.float64)
        for H in self.holes:
            xmin, xmax, ymin, ymax = p35.hole_bounds(H)
            dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
            dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
            best = np.minimum(best, np.hypot(dx, dy))
        return best

    def feature_frames(self, xy: np.ndarray, hole_priority: float = 1.15) -> Dict[str, np.ndarray]:
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        n = int(pts.shape[0])
        if n == 0:
            return {
                'normal': np.zeros((0, 2), dtype=np.float64),
                'tangent': np.zeros((0, 2), dtype=np.float64),
                'kind': np.zeros((0,), dtype=np.int32),
                'hole_dist': np.zeros((0,), dtype=np.float64),
                'outer_dist': np.zeros((0,), dtype=np.float64),
            }

        x = pts[:, 0]
        y = pts[:, 1]

        outer_d_all = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        outer_side = np.argmin(outer_d_all, axis=1)
        outer_dist = np.min(outer_d_all, axis=1)
        outer_n = np.zeros((n, 2), dtype=np.float64)
        outer_n[outer_side == 0] = np.array([1.0, 0.0])
        outer_n[outer_side == 1] = np.array([-1.0, 0.0])
        outer_n[outer_side == 2] = np.array([0.0, 1.0])
        outer_n[outer_side == 3] = np.array([0.0, -1.0])

        hole_dist = np.full((n,), np.inf, dtype=np.float64)
        hole_n = np.zeros((n, 2), dtype=np.float64)
        eps = 1e-14

        for H in self.holes:
            xmin, xmax, ymin, ymax = p35.hole_bounds(H)
            px = np.clip(x, xmin, xmax)
            py = np.clip(y, ymin, ymax)
            vx = x - px
            vy = y - py
            d = np.hypot(vx, vy)

            cand_n = np.zeros((n, 2), dtype=np.float64)
            mask = d > eps
            cand_n[mask, 0] = vx[mask] / d[mask]
            cand_n[mask, 1] = vy[mask] / d[mask]

            if np.any(~mask):
                dl = np.abs(x - xmin)
                dr = np.abs(xmax - x)
                db = np.abs(y - ymin)
                dt = np.abs(ymax - y)
                side = np.argmin(np.stack([dl, dr, db, dt], axis=1), axis=1)
                cand_n[(~mask) & (side == 0)] = np.array([-1.0, 0.0])
                cand_n[(~mask) & (side == 1)] = np.array([1.0, 0.0])
                cand_n[(~mask) & (side == 2)] = np.array([0.0, -1.0])
                cand_n[(~mask) & (side == 3)] = np.array([0.0, 1.0])

            upd = d < hole_dist
            hole_dist[upd] = d[upd]
            hole_n[upd] = cand_n[upd]

        use_hole = hole_dist <= float(hole_priority) * outer_dist
        normal = outer_n.copy()
        normal[use_hole] = hole_n[use_hole]
        tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
        kind = np.where(use_hole, 1, 0).astype(np.int32)
        return {
            'normal': normal.astype(np.float64),
            'tangent': tangent.astype(np.float64),
            'kind': kind,
            'hole_dist': hole_dist.astype(np.float64),
            'outer_dist': outer_dist.astype(np.float64),
        }

    def obstacle_bias(self, xy: np.ndarray, ell: np.ndarray, cfg: Any) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0,), dtype=np.float64)
        ellv = np.maximum(np.asarray(ell, dtype=np.float64).reshape(-1), 1e-12)
        radius = np.maximum(
            float(getattr(cfg, 'GEO_BIAS_RADIUS_MULT', 2.5)) * ellv,
            float(getattr(cfg, 'ELL_MIN', 1e-6)),
        )
        d_h = self.distance_to_holes(pts)
        d_o = self.distance_to_outer_boundary(pts)
        w = np.ones((pts.shape[0],), dtype=np.float64)
        w += float(getattr(cfg, 'GEO_BIAS_HOLE', 1.5)) * np.exp(-d_h / np.maximum(radius, 1e-12))
        w += float(getattr(cfg, 'GEO_BIAS_OUTER', 0.75)) * np.exp(-d_o / np.maximum(radius, 1e-12))
        return w

    def sample_hole_band(self, n: int, seed: int, band_width: float) -> np.ndarray:
        n = int(max(0, n))
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        band_width = float(max(band_width, 1e-8))
        pts = np.empty((n, 2), dtype=np.float64)

        for i in range(n):
            H = self.holes[int(rng.integers(0, len(self.holes)))]
            xmin, xmax, ymin, ymax = p35.hole_bounds(H)
            if float(rng.random()) < 0.25:
                cid = int(rng.integers(0, 4))
                corners = np.array([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]], dtype=np.float64)
                normals = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
                off_n = band_width * (0.05 + 0.95 * (rng.random() ** 2))
                off_t = 0.35 * band_width * (2.0 * rng.random() - 1.0)
                pts[i] = corners[cid] + off_n * normals[cid] + off_t * tangents[cid]
            else:
                side = int(rng.integers(0, 4))
                t = float(rng.random())
                off = band_width * (0.05 + 0.95 * (rng.random() ** 2))
                if side == 0:
                    pts[i] = np.array([xmin - off, ymin + t * (ymax - ymin)], dtype=np.float64)
                elif side == 1:
                    pts[i] = np.array([xmax + off, ymin + t * (ymax - ymin)], dtype=np.float64)
                elif side == 2:
                    pts[i] = np.array([xmin + t * (xmax - xmin), ymin - off], dtype=np.float64)
                else:
                    pts[i] = np.array([xmin + t * (xmax - xmin), ymax + off], dtype=np.float64)

        pts = self.project_points_outside_holes(np.clip(pts, 0.0, 1.0))
        return pts.astype(np.float64)

    def sample_outer_band(self, n: int, seed: int, band_width: float) -> np.ndarray:
        n = int(max(0, n))
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        band_width = float(max(band_width, 1e-8))
        pts = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            side = int(rng.integers(0, 4))
            t = float(rng.random())
            off = band_width * (0.05 + 0.95 * (rng.random() ** 2))
            if side == 0:
                pts[i] = np.array([off, t], dtype=np.float64)
            elif side == 1:
                pts[i] = np.array([1.0 - off, t], dtype=np.float64)
            elif side == 2:
                pts[i] = np.array([t, off], dtype=np.float64)
            else:
                pts[i] = np.array([t, 1.0 - off], dtype=np.float64)
        pts = self.project_points_outside_holes(np.clip(pts, 0.0, 1.0))
        return pts.astype(np.float64)

    def build_growth_candidate_cloud(self, n: int, seed: int, cfg: Any, ref_ell: Optional[float] = None) -> np.ndarray:
        n = int(max(1, n))
        rng = np.random.default_rng(int(seed))
        if ref_ell is None:
            ref_ell = float(max(getattr(cfg, 'ELL_MIN', 1e-6), getattr(cfg, 'ELL0', 0.15)))
        ref_ell = float(max(ref_ell, getattr(cfg, 'ELL_MIN', 1e-6)))

        fg = float(getattr(cfg, 'CAND_GLOBAL_FRAC', 0.45))
        fh = float(getattr(cfg, 'CAND_HOLE_FRAC', 0.35))
        fo = float(getattr(cfg, 'CAND_OUTER_FRAC', 0.20))
        s = max(fg + fh + fo, 1e-12)
        fg, fh, fo = fg / s, fh / s, fo / s

        ng = int(round(n * fg))
        nh = int(round(n * fh))
        no = int(max(0, n - ng - nh))

        band_h = max(float(getattr(cfg, 'ELL_MIN', 1e-6)), float(getattr(cfg, 'HOLE_BAND_MULT', 2.5)) * ref_ell)
        band_o = max(float(getattr(cfg, 'ELL_MIN', 1e-6)), float(getattr(cfg, 'OUTER_BAND_MULT', 2.0)) * ref_ell)

        parts = []
        if ng > 0:
            pts_g, _ = self.sample_domain_qmc(ng, seed=int(seed) + 17, cfg=cfg)
            parts.append(pts_g)
        if nh > 0:
            parts.append(self.sample_hole_band(nh, seed=int(seed) + 29, band_width=band_h))
        if no > 0:
            parts.append(self.sample_outer_band(no, seed=int(seed) + 41, band_width=band_o))

        Z = np.concatenate(parts, axis=0)[:n]
        rng.shuffle(Z)
        return Z.astype(np.float64)




def fixed_add_anchors_strategy1_cm_meshfree(
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
    Fixed wrapper around pf.add_anchors_strategy1_cm_meshfree.

    The upstream implementation leaves Z undefined when add_count >= cm * n_marked,
    which happens often in short smoke tests and early branch stages. Here we always
    construct the child proposal cloud before consuming it.
    """
    rng = np.random.default_rng(int(seed))
    marked_ids = np.asarray(marked_ids, dtype=np.int32).reshape(-1)
    M = int(anchors.shape[0])
    add_count = int(add_count)
    cm = int(max(1, cm))

    if add_count <= 0 or marked_ids.size == 0:
        return anchors, ell, aw, {"added": 0, "tries": 0, "rejects": 0, "base_ell": float("nan"),
                                  "min_sep": float("nan")}

    M0 = int(getattr(cfg, "ANCHOR_INIT", max(1, M)))
    base_ell = float(pf.ell_schedule(cfg, M=M + add_count, M0=M0))

    A = int(marked_ids.size)
    total_full = cm * A
    if add_count >= total_full:
        parents_eff = marked_ids
        per_parent = [cm] * A
        add_target = total_full
    else:
        if add_count < cm:
            parents_eff = marked_ids[:1]
            per_parent = [add_count]
            add_target = add_count
        else:
            A_full = int(add_count // cm)
            remainder = int(add_count - A_full * cm)
            parents_eff = marked_ids[:max(1, A_full + (1 if remainder > 0 else 0))]
            per_parent = [cm] * int(min(A_full, parents_eff.size))
            if remainder > 0 and len(per_parent) < int(parents_eff.size):
                per_parent.append(remainder)
            add_target = int(sum(per_parent))

    grow_policy = str(child_policy).strip().lower()
    if grow_policy == "strategy2":
        Z, ell_Z, _parent_anchor_id = pf._strategy2_children_candidates_np(
            rng, anchors, ell, parents_eff, cm=cm, cfg=cfg, base_ell=base_ell
        )
        npp = cm
    else:
        Z, ell_Z, _parent_anchor_id = pf._strategy1_children_candidates_np(
            rng, anchors, ell, parents_eff, cm=cm, cfg=cfg, base_ell=base_ell
        )
        overs = int(max(1, getattr(cfg, "ANCHOR_CHILD_OVERSAMPLE", 2)))
        npp = cm * overs

    N = int(Z.shape[0])
    if N <= 0:
        return anchors, ell, aw, {"added": 0, "tries": 0, "rejects": 0, "base_ell": base_ell,
                                  "min_sep": float(cfg.MIN_SEP_ALPHA) * base_ell}

    if not bool(use_eta):
        eta = np.ones((N,), dtype=np.float64)
    else:
        lift_mode_tf = tf.constant(int(lift_mode), dtype=tf.int32)
        chunk = int(min(8192, max(1024, getattr(cfg, "EVAL_POINT_BATCH", 8192))))
        laps = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            z_tf = tf.constant(Z[s:e], dtype=DTYPE)
            lap_tf = pf.laplace_u_tf(
                unet, gnet, lift_mode_tf, z_tf,
                float(cfg.LAPLACE_H),
                float(cfg.BUBBLE_SCALE), bubble_mode_tf, softmin_tau_tf, bubble_power_tf,
            )
            laps.append(tf.abs(lap_tf))
        eta = tf.concat(laps, axis=0).numpy().reshape(-1).astype(np.float64)

    existing_tree = pf.cKDTree(anchors) if M > 0 else None
    accepted_pts = []
    accepted_ell = []
    acc_tree = None
    rebuild_every = 128
    tries = 0
    rejects = 0

    for p_idx in range(int(parents_eff.size)):
        want = int(per_parent[p_idx]) if p_idx < len(per_parent) else 0
        if want <= 0:
            continue

        s0 = p_idx * npp
        s1 = min((p_idx + 1) * npp, N)
        if s1 <= s0:
            continue

        if grow_policy == "strategy2":
            local_pts = Z[s0:s1]
            local_ells = ell_Z[s0:s1]
            got = 0
            for z, ell_c in zip(local_pts, local_ells):
                if len(accepted_pts) >= add_target:
                    break
                tries += 1
                sep = float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * float(ell_c)
                if existing_tree is not None and float(existing_tree.query(z, k=1)[0]) < sep:
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
                accepted_pts.append(z.astype(np.float64))
                accepted_ell.append(float(ell_c))
                got += 1
                if len(accepted_pts) % rebuild_every == 0:
                    acc_tree = pf.cKDTree(np.asarray(accepted_pts, dtype=np.float64))
                if got >= want:
                    break
            if len(accepted_pts) >= add_target:
                break
            continue

        local_eta = eta[s0:s1]
        order = np.argsort(-local_eta)
        local_pts = Z[s0:s1][order]
        local_ells = ell_Z[s0:s1][order]

        got = 0
        for z, ell_c in zip(local_pts, local_ells):
            if len(accepted_pts) >= add_target:
                break
            tries += 1
            sep = float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * float(ell_c)

            if existing_tree is not None and float(existing_tree.query(z, k=1)[0]) < sep:
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

            accepted_pts.append(z.astype(np.float64))
            accepted_ell.append(float(ell_c))
            got += 1
            if len(accepted_pts) % rebuild_every == 0:
                acc_tree = pf.cKDTree(np.asarray(accepted_pts, dtype=np.float64))
            if got >= want:
                break

        if len(accepted_pts) >= add_target:
            break

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

        cand_np, _ = pf.build_ref_cloud_square(int(cfg.NCAND), seed=int(cfg.CAND_SEED + seed + 1337), cfg=cfg, dim=2)
        anchors_tmp, ell_tmp, aw_tmp, info2 = pf.add_anchors_adaptive(
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
            "base_ell": base_ell,
            "min_sep": float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * base_ell,
        }

    if len(accepted_pts) == 0:
        return anchors, ell, aw, {
            "added": 0, "tries": tries, "rejects": rejects,
            "eta_mean": float(np.mean(eta)) if eta.size else float("nan"),
            "eta_p95": float(np.quantile(eta, 0.95)) if eta.size else float("nan"),
            "base_ell": base_ell,
            "min_sep": float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * base_ell,
        }

    new_pts = np.asarray(accepted_pts, dtype=np.float64).reshape(-1, 2)
    new_ell = np.asarray(accepted_ell, dtype=np.float64).reshape(-1, 1)
    new_aw = np.full((new_pts.shape[0], 1), float(cfg.ANCHOR_WEIGHT_NEW), dtype=np.float64)

    anchors_new = np.concatenate((anchors, new_pts), axis=0)
    ell_new = np.concatenate((ell, new_ell), axis=0)
    aw_new = np.concatenate((aw, new_aw), axis=0)

    return anchors_new, ell_new, aw_new, {
        "added": int(new_pts.shape[0]),
        "tries": tries,
        "rejects": rejects,
        "eta_mean": float(np.mean(eta)) if eta.size else float("nan"),
        "eta_p95": float(np.quantile(eta, 0.95)) if eta.size else float("nan"),
        "base_ell": base_ell,
        "min_sep": float(getattr(cfg, "MIN_SEP_ALPHA", 0.7)) * base_ell,
    }

# -----------------------------------------------------------------------------
# Monkey patches: domain-aware sampler / test bubble / PDE operators
# -----------------------------------------------------------------------------
ORIG_BUILD_REF_CLOUD = pf.build_ref_cloud_square
ORIG_STRAT1_CHILDREN = pf._strategy1_children_candidates_np
ORIG_STRAT2_CHILDREN = pf._strategy2_children_candidates_np


def install_problem35_overrides(domain: Problem35Domain) -> None:
    """Patch only the problem-dependent hooks of the base patchless implementation."""

    def build_ref_cloud_problem35(n: int, seed: int, cfg: pf.Config, dim: int = 2):
        if int(dim) != 2:
            return ORIG_BUILD_REF_CLOUD(n, seed=seed, cfg=cfg, dim=dim)
        return domain.sample_domain_qmc(int(n), int(seed), cfg)

    def _strategy1_children_candidates_problem35(
        rng: np.random.Generator,
        anchors: np.ndarray,
        ell: np.ndarray,
        marked_ids: np.ndarray,
        cm: int,
        cfg: pf.Config,
        base_ell: float,
    ):
        marked_ids = np.asarray(marked_ids, dtype=np.int32).reshape(-1)
        A = int(marked_ids.shape[0])
        cm = int(max(1, cm))
        overs = int(max(1, getattr(cfg, "ANCHOR_CHILD_OVERSAMPLE", 2)))
        npp = cm * overs
        if A <= 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int32),
            )

        parents = np.asarray(anchors, dtype=np.float64)[marked_ids]
        ell_p = np.asarray(ell, dtype=np.float64).reshape(-1)[marked_ids]

        parent_rep = np.repeat(np.arange(A, dtype=np.int32), npp)
        parent_anchor_id = marked_ids[parent_rep]

        a_ratio = float(getattr(cfg, "ANCHOR_CHILD_A_RATIO", 1.25))
        scale = math.sqrt(max(1e-12, a_ratio / float(cm)))
        cutoff = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))
        rad = (0.5 * scale * cutoff) * ell_p[parent_rep]

        U = rng.random((A * npp, 2), dtype=np.float64) - 0.5
        Z = parents[parent_rep] + U * rad.reshape(-1, 1)
        Z = domain.project_points_outside_holes(Z)

        lam_lo = float(getattr(cfg, "ANCHOR_CHILD_LAM_LO", 0.9))
        lam_hi = float(getattr(cfg, "ANCHOR_CHILD_LAM_HI", 10.0 / 9.0))
        lam = rng.uniform(lam_lo, lam_hi, size=(A * npp,)).astype(np.float64)

        ell_base = np.minimum(float(base_ell), ell_p[parent_rep])
        ell_Z = np.maximum(float(getattr(cfg, "ELL_MIN", 1e-6)), ell_base * lam)
        ell_Z = np.minimum(ell_Z, float(base_ell))
        return Z.astype(np.float64), ell_Z.astype(np.float64), parent_anchor_id.astype(np.int32)

    def _strategy2_children_candidates_problem35(
        rng: np.random.Generator,
        anchors: np.ndarray,
        ell: np.ndarray,
        marked_ids: np.ndarray,
        cm: int,
        cfg: pf.Config,
        base_ell: float,
    ):
        del rng
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

        a_ratio = float(getattr(cfg, "ANCHOR_CHILD_A_RATIO", 1.25))
        scale = math.sqrt(max(1e-12, a_ratio / float(cm)))
        cutoff = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))
        rad = (0.5 * scale * cutoff) * ell_p[parent_rep]

        ref = pf._strategy2_fixed_offsets(cm) - 0.5
        ref_rep = np.tile(ref, (A, 1))
        Z = parents[parent_rep] + ref_rep * rad.reshape(-1, 1)
        Z = domain.project_points_outside_holes(Z)

        ell_base = np.minimum(float(base_ell), ell_p[parent_rep])
        ell_Z = np.maximum(float(getattr(cfg, "ELL_MIN", 1e-6)), ell_base)
        ell_Z = np.minimum(ell_Z, float(base_ell))
        return Z.astype(np.float64), ell_Z.astype(np.float64), parent_anchor_id.astype(np.int32)

    @tf.function(reduce_retracing=True)
    def grad_u_problem35_tf(unet, gnet, lift_mode, xy, bubble_scale, bubble_mode, softmin_tau, bubble_power):
        del gnet, lift_mode, bubble_scale, bubble_mode, softmin_tau, bubble_power
        xy = tf.convert_to_tensor(xy, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(xy)
            w = unet(xy)
            phi, _ = domain.phi_and_grad(xy)
            u = phi * w
        gu = tape.gradient(u, xy)
        return gu

    def _project_omega2_tf(xy_in, eps=1e-6):
        xy_in = tf.convert_to_tensor(xy_in)
        in_dtype = xy_in.dtype
        xy64 = tf.cast(xy_in, tf.float64)
        xy_out64 = tf.numpy_function(
            lambda a: np.asarray(
                domain.project_points_outside_holes(np.asarray(a, dtype=np.float64), eps=float(eps)),
                dtype=np.float64,
            ),
            [xy64],
            Tout=tf.float64,
        )
        xy_out64.set_shape(xy_in.shape)
        return tf.cast(xy_out64, in_dtype)

    def _u_problem35_fd(unet, xy):
        xy = tf.cast(tf.convert_to_tensor(xy), DTYPE)
        w = unet(xy)
        phi, _ = domain.phi_and_grad(xy)
        phi = tf.cast(phi, xy.dtype)
        return phi * w

    @tf.function(reduce_retracing=True)
    def laplace_u_problem35_tf(unet, gnet, lift_mode, xy, h, bubble_scale, bubble_mode, softmin_tau, bubble_power):
        del gnet, lift_mode, bubble_scale, bubble_mode, softmin_tau, bubble_power
        xy = tf.cast(tf.convert_to_tensor(xy), DTYPE)
        h = tf.cast(h, xy.dtype)
        zero = tf.cast(0.0, xy.dtype)

        dx = tf.reshape(tf.stack([h, zero], axis=0), (1, 2))
        dy = tf.reshape(tf.stack([zero, h], axis=0), (1, 2))

        xy0 = tf.cast(xy, DTYPE)
        xyp = tf.cast(_project_omega2_tf(xy0 + dx), DTYPE)
        xym = tf.cast(_project_omega2_tf(xy0 - dx), DTYPE)
        yxp = tf.cast(_project_omega2_tf(xy0 + dy), DTYPE)
        yxm = tf.cast(_project_omega2_tf(xy0 - dy), DTYPE)

        xy_all = tf.cast(tf.concat([xy0, xyp, xym, yxp, yxm], axis=0), DTYPE)
        u_all = _u_problem35_fd(unet, xy_all)
        u0, uxp, uxm, uyp, uym = tf.split(u_all, num_or_size_splits=5, axis=0)

        inv_h2 = tf.cast(1.0, DTYPE) / (h * h)
        two = tf.cast(2.0, DTYPE)
        lap = (uxp - two * u0 + uxm) * inv_h2 + (uyp - two * u0 + uym) * inv_h2
        return tf.cast(lap, DTYPE)

    @tf.function(reduce_retracing=True)
    def R_block_problem35(
        unet,
        gnet,
        lift_mode_tf,
        xy,
        b0,
        db0,
        xi,
        ell,
        bubble_scale,
        bubble_mode,
        softmin_tau,
        bubble_power,
        cutoff_s=None,
    ):
        gu = grad_u_problem35_tf(
            unet, gnet, lift_mode_tf, xy,
            bubble_scale, bubble_mode, softmin_tau, bubble_power,
        )
        k, gkx, gky = pf.kernel_k_gradxy_matmul(xy, xi, ell, cutoff_s=cutoff_s)
        dbx = db0[:, 0:1]
        dby = db0[:, 1:2]
        gvx = k * dbx + b0 * gkx
        gvy = k * dby + b0 * gky
        v = b0 * k
        f = p35.f_rhs_tf(xy)
        dot = gu[:, 0:1] * gvx + gu[:, 1:2] * gvy - f * v
        return tf.reduce_sum(dot, axis=0)

    pf.build_ref_cloud_square = build_ref_cloud_problem35
    pf._strategy1_children_candidates_np = _strategy1_children_candidates_problem35
    pf._strategy2_children_candidates_np = _strategy2_children_candidates_problem35
    pf.grad_u_tf = grad_u_problem35_tf
    pf.laplace_u_tf = laplace_u_problem35_tf
    pf.R_block = R_block_problem35
    pf.add_anchors_strategy1_cm_meshfree = fixed_add_anchors_strategy1_cm_meshfree


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clone_model_weights(src: tf.keras.Model, dst: tf.keras.Model) -> None:
    dst.set_weights([np.array(w, copy=True) for w in src.get_weights()])


def build_pf_config(args: argparse.Namespace) -> pf.Config:
    cfg = pf.Config()
    cfg.SEED = int(args.seed)
    cfg.PROGRESS_EVERY = int(getattr(args, "progress_every", 50))
    cfg.SNAPSHOT_EVERY = int(getattr(args, "snapshot_every", 1))

    cfg.L_LAYERS = int(args.layers)
    cfg.WIDTH = int(args.width)

    cfg.LR0 = float(args.lr0)
    cfg.LR1 = float(args.lr1)
    cfg.LAMBDA_REG = float(args.lambda_reg)
    cfg.W_PDE = float(args.w_pde)
    cfg.W_ENERGY = float(args.w_energy)

    cfg.ADAM_STEPS_FIRST = int(args.adam_steps_first)
    cfg.ADAM_STEPS_AFTER_GROW = int(args.adam_steps_after_grow)
    cfg.LBFGS_MAXITER_FIRST = int(args.lbfgs_first)
    cfg.LBFGS_MAXITER_NEXT = int(args.lbfgs_next)
    cfg.LBFGS_FTOL = float(args.lbfgs_ftol)
    cfg.LBFGS_GTOL = float(args.lbfgs_gtol)
    cfg.LBFGS_MAXCOR = int(args.lbfgs_maxcor)
    cfg.LBFGS_MAXFUN = int(args.lbfgs_maxfun)

    cfg.ADAM_POINT_BATCH = int(args.adam_point_batch)
    cfg.ADAM_ANCHOR_BATCH = int(args.adam_anchor_batch)
    cfg.ADAM_GRAD_ACCUM = int(args.adam_grad_accum)
    cfg.EVAL_POINT_BATCH = int(args.eval_point_batch)
    cfg.EVAL_ANCHOR_BATCH = int(args.eval_anchor_batch)
    cfg.LBFGS_POINT_BATCH = int(args.lbfgs_point_batch)
    cfg.LBFGS_ANCHOR_BATCH = int(args.lbfgs_anchor_batch)

    cfg.NQ_GLOBAL = int(args.nq_global)
    cfg.NCAND = int(args.ncand)
    cfg.QMC_KIND = str(args.qmc_kind)
    cfg.QMC_SCRAMBLE = bool(int(args.qmc_scramble))

    cfg.KERNEL_KIND = str(args.kernel)
    cfg.ELL0 = float(args.ell0)
    cfg.ELL_MIN = float(args.ell_min)
    cfg.ELL_SCHED_P = float(args.ell_sched_p)
    cfg.KERNEL_CUTOFF = float(args.cutoff)
    cfg.ELL_FROM_H = int(args.ell_from_h)
    cfg.ELL_C_RHO = float(args.ell_c_rho)
    cfg.DOMAIN_NORMALIZE = int(args.domain_normalize)

    cfg.ANCHOR_INIT = int(args.anchor_init)
    cfg.TARGET_NANCHOR = int(args.target_nanchor)
    cfg.ADD_FRAC_MAX = float(args.add_frac_max)
    cfg.MIN_ADD_PER_ITER = int(args.min_add)
    cfg.MAX_ADD_PER_ITER = int(args.max_add)
    cfg.MARK_FRAC = float(args.mark_frac)
    cfg.MARK_CAP = float(args.mark_cap)
    cfg.ANCHOR_CHILD_A_RATIO = float(args.child_a_ratio)
    cfg.ANCHOR_CHILD_LAM_LO = float(args.child_lam_lo)
    cfg.ANCHOR_CHILD_LAM_HI = float(args.child_lam_hi)
    cfg.ANCHOR_CHILD_OVERSAMPLE = int(args.child_oversample)
    cfg.STRATEGY1_FILL_GLOBAL = int(args.fill_global)
    cfg.MIN_SEP_ALPHA = float(args.min_sep_alpha)
    cfg.REJECT_MAX_TRIES = int(args.reject_max_tries)
    cfg.ETA_POWER = float(args.eta_power)
    cfg.ETA_EPS = float(args.eta_eps)

    # Hole-aware / obstacle-aware growth controls
    cfg.GEO_BIAS_HOLE = float(args.geo_bias_hole)
    cfg.GEO_BIAS_OUTER = float(args.geo_bias_outer)
    cfg.GEO_BIAS_RADIUS_MULT = float(args.geo_bias_radius_mult)
    cfg.HOLE_PRIORITY_FACTOR = float(args.hole_priority_factor)
    cfg.CAND_GLOBAL_FRAC = float(args.cand_global_frac)
    cfg.CAND_HOLE_FRAC = float(args.cand_hole_frac)
    cfg.CAND_OUTER_FRAC = float(args.cand_outer_frac)
    cfg.HOLE_BAND_MULT = float(args.hole_band_mult)
    cfg.OUTER_BAND_MULT = float(args.outer_band_mult)
    cfg.HOLE_AWARE_CANDIDATES = 1
    cfg.GROWTH_REF_ELL = float(max(cfg.ELL_MIN, cfg.ELL0))

    # Problem (35): use the exact Ω₂ ADF bubble, so the benchmark-specific bubble knobs are irrelevant.
    cfg.BUBBLE_SCALE = 1.0
    cfg.BUBBLE_KIND = "product"
    cfg.BUBBLE_POWER = 1.0
    cfg.SOFTMIN_TAU = 1e-3
    cfg.LAPLACE_H = 1e-3

    # Correctness first for Poisson-with-source in Ω₂: use the dense path.
    cfg.USE_KDTREE_PRUNING = 0
    cfg.USE_STENCIL_BANK = 0
    cfg.CACHE_ENERGY = 1
    cfg.NESTED_ENERGY = 0
    cfg.WHITEN_MODE = "energy"
    cfg.WHITEN_EPS = float(args.whiten_eps)

    # We evaluate H1 with the dedicated Ω₂ quadrature from the patch-based Problem (35) code.
    cfg.H1_USE_GIQMC = 0
    cfg.MONOTONE_H1_REPORT = 0
    cfg.MONOTONE_H1_RESTORE = 0
    cfg.RESTORE_BEST_AFTER_GROW = False

    # Lift is irrelevant here because g=0 on the whole boundary.
    cfg.GNET_STEPS = 0
    cfg.GNET_BATCH = 0

    return cfg


def build_aux_cfg(args: argparse.Namespace) -> Problem35AuxConfig:
    return Problem35AuxConfig(
        ADF_M=int(args.adf_m),
        ADF_EPS=float(args.adf_eps),
        H1_NCELLS=int(args.h1_ncells),
        H1_GAUSS_N=int(args.h1_gauss_n),
        H1_EVAL_CHUNK=int(args.h1_eval_chunk),
    )


def compute_energy_denom(G_full: Optional[np.ndarray], M: int, cfg: pf.Config) -> tf.Tensor:
    if str(getattr(cfg, "WHITEN_MODE", "energy")) == "energy":
        if G_full is None:
            raise ValueError("G_full is required when WHITEN_MODE='energy'.")
        denom = np.sqrt(np.maximum(np.asarray(G_full, dtype=np.float64).reshape(-1), 0.0) + float(cfg.WHITEN_EPS))
        denom = np.maximum(denom, 1e-8)
        return tf.constant(denom.reshape(M, 1), dtype=DTYPE)
    return tf.ones((M, 1), dtype=DTYPE)


def compute_G_full_dense(
    Xq_tf: tf.Tensor,
    b0_all_tf: tf.Tensor,
    db0_all_tf: tf.Tensor,
    anchors: np.ndarray,
    ell: np.ndarray,
    cfg: pf.Config,
) -> np.ndarray:
    return np.asarray(
        pf.compute_G_full_cached(
            Xq_tf,
            b0_all_tf,
            db0_all_tf,
            np.asarray(anchors, dtype=np.float64),
            np.asarray(ell, dtype=np.float64).reshape(-1),
            cfg,
            cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
        ),
        dtype=np.float64,
    ).reshape(-1)


# -----------------------------------------------------------------------------
# Dense Adam for the Poisson weak residual  ∫ (∇u·∇v - f v)
# -----------------------------------------------------------------------------
def adam_train_poisson_cached(
    unet: tf.keras.Model,
    Xq_tf: tf.Tensor,
    b0_all_tf: tf.Tensor,
    db0_all_tf: tf.Tensor,
    f_all_tf: tf.Tensor,
    anchors: np.ndarray,
    ell: np.ndarray,
    aw: np.ndarray,
    denom_tf: tf.Tensor,
    cfg: pf.Config,
    steps: int,
    seed: int,
    tag: str = "adam",
) -> None:
    steps = int(steps)
    if steps <= 0:
        return

    vars_ = unet.trainable_variables
    if not vars_:
        return

    rng = np.random.default_rng(int(seed))

    lr0 = float(cfg.LR0)
    lr1 = float(cfg.LR1)
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
    reg_coeff = tf.constant(float(cfg.LAMBDA_REG), dtype=DTYPE)

    Nq = int(Xq_tf.shape[0])
    M = int(np.asarray(anchors).shape[0])
    pb0 = int(cfg.ADAM_POINT_BATCH)
    ab0 = int(cfg.ADAM_ANCHOR_BATCH)
    ga0 = int(cfg.ADAM_GRAD_ACCUM)
    w_sum_total_tf = tf.maximum(tf.reduce_sum(tf.constant(np.asarray(aw, dtype=np.float64), dtype=DTYPE)), tf.cast(1e-30, DTYPE))

    anchors_tf = tf.constant(np.asarray(anchors, dtype=np.float64), dtype=DTYPE)
    ell_tf = tf.constant(np.asarray(ell, dtype=np.float64), dtype=DTYPE)
    aw_tf = tf.constant(np.asarray(aw, dtype=np.float64), dtype=DTYPE)

    cutoff_s = None
    cutoff_factor = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))
    if cutoff_factor > 0.0:
        cutoff_s = tf.constant(cutoff_factor ** 2, dtype=DTYPE)

    @tf.function(reduce_retracing=True)
    def train_step(xy, b0, db0, f_batch, xi, el, aw_batch, den):
        with tf.GradientTape() as tape:
            gu = pf.grad_u_tf(
                unet, None, tf.constant(0, dtype=tf.int32), xy,
                1.0, tf.constant(0, dtype=tf.int32), tf.constant(1e-3, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE),
            )
            k, gkx, gky = pf.kernel_k_gradxy_matmul(xy, xi, el, cutoff_s=cutoff_s)
            dbx = db0[:, 0:1]
            dby = db0[:, 1:2]
            gvx = k * dbx + b0 * gkx
            gvy = k * dby + b0 * gky
            v = b0 * k
            dot = gu[:, 0:1] * gvx + gu[:, 1:2] * gvy - f_batch * v
            R = tf.reduce_mean(dot, axis=0)
            r = R[:, None] / den
            num = tf.reduce_sum(aw_batch * tf.square(r))
            A = tf.cast(tf.shape(aw_batch)[0], DTYPE)
            L_pde = (tf.cast(M, DTYPE) / tf.maximum(A, tf.cast(1.0, DTYPE))) * (num / w_sum_total_tf)
            L_energy = tf.reduce_mean(tf.reduce_sum(tf.square(gu), axis=1))
            L = tf.cast(cfg.W_PDE, DTYPE) * L_pde + tf.cast(cfg.W_ENERGY, DTYPE) * L_energy
            reg = tf.add_n([tf.reduce_sum(tf.square(vv)) for vv in vars_])
            J = L + reg_coeff * reg
        grads = tape.gradient(J, vars_)
        return J, grads

    pb = int(pb0)
    ab = int(ab0)
    ga = int(ga0)

    print_every = int(max(0, getattr(cfg, "PROGRESS_EVERY", 0)))
    last_J = float("nan")

    for _it in range(int(steps)):
        while True:
            try:
                grads_accum = pf._zeros_like(vars_)
                for _acc in range(int(ga)):
                    pids = rng.integers(0, Nq, size=pb, dtype=np.int32)
                    aids = rng.integers(0, M, size=ab, dtype=np.int32)

                    xy = tf.gather(Xq_tf, pids)
                    b0 = tf.gather(b0_all_tf, pids)
                    db0 = tf.gather(db0_all_tf, pids)
                    f_batch = tf.gather(f_all_tf, pids)

                    xi = tf.gather(anchors_tf, aids)
                    el = tf.gather(ell_tf, aids)
                    aw_batch = tf.gather(aw_tf, aids)
                    den = tf.gather(denom_tf, aids)

                    J_batch, g = train_step(xy, b0, db0, f_batch, xi, el, aw_batch, den)
                    grads_accum = pf._safe_add_grads(grads_accum, g)

                inv = tf.constant(1.0 / float(max(1, ga)), dtype=DTYPE)
                grads_accum = [g * inv for g in grads_accum]
                clip = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0))
                if clip > 0.0:
                    grads_accum, _ = tf.clip_by_global_norm(grads_accum, tf.cast(clip, DTYPE))
                opt.apply_gradients(zip(grads_accum, vars_))
                last_J = float(J_batch.numpy())
                break

            except tf.errors.ResourceExhaustedError:
                if pb <= 256 and ab <= 1:
                    raise
                pb = max(256, pb // 2)
                ab = max(1, ab // 2)
                eff0 = int(pb0) * int(max(1, ga0))
                ga = int(min(64, max(1, math.ceil(eff0 / max(1, pb)))))
                print(f"[OOM backoff] Adam pb -> {pb}, ab -> {ab}, grad_accum -> {ga}", flush=True)

        if print_every and (((_it + 1) % print_every) == 0 or (_it + 1) == int(steps)):
            print(f"    [{tag}] step {_it+1:05d}/{int(steps):05d} | J={last_J:.3e} | pb={pb} ab={ab} ga={ga}", flush=True)


# -----------------------------------------------------------------------------
# H1 error on Ω₂: reuse the exact quadrature from the patch-based Problem (35)
# -----------------------------------------------------------------------------
def rel_H1_error_omega2(unet: tf.keras.Model, domain: Problem35Domain, aux_cfg: Problem35AuxConfig, stage_name: str = "stage") -> float:
    print(f"[{stage_name}] H1 evaluation start", flush=True)
    t_h = time.time()
    cfg_h1 = p35.Config()
    cfg_h1.H1_NCELLS = int(aux_cfg.H1_NCELLS)
    cfg_h1.H1_GAUSS_N = int(aux_cfg.H1_GAUSS_N)
    cfg_h1.H1_EVAL_CHUNK = int(aux_cfg.H1_EVAL_CHUNK)
    val = float(p35.rel_H1_error_omega2(unet, domain.holes, cfg_h1, domain.phi_and_grad))
    print(f"[{stage_name}] H1 evaluation done in {time.time() - t_h:.1f}s", flush=True)
    return val


# -----------------------------------------------------------------------------
# Training / evaluation stages
# -----------------------------------------------------------------------------
def train_stage(
    unet: tf.keras.Model,
    anchors: np.ndarray,
    ell: np.ndarray,
    aw: np.ndarray,
    cfg: pf.Config,
    Xq_tf: tf.Tensor,
    b0_all_tf: tf.Tensor,
    db0_all_tf: tf.Tensor,
    f_all_tf: tf.Tensor,
    stage_seed: int,
    is_first: bool,
    stage_name: str = "stage",
) -> Tuple[Dict[str, Any], np.ndarray]:
    print(f"[{stage_name}] assembling G_full for M={int(np.asarray(anchors).shape[0])}", flush=True)
    t_g = time.time()
    G_full = compute_G_full_dense(Xq_tf, b0_all_tf, db0_all_tf, anchors, ell, cfg)
    print(f"[{stage_name}] G_full done in {time.time() - t_g:.1f}s", flush=True)
    denom_tf = compute_energy_denom(G_full, int(anchors.shape[0]), cfg)

    adam_steps = int(cfg.ADAM_STEPS_FIRST if is_first else cfg.ADAM_STEPS_AFTER_GROW)
    if adam_steps > 0:
        print(f"[{stage_name}] Adam start | steps={adam_steps}", flush=True)
        t_a = time.time()
        adam_train_poisson_cached(
            unet,
            Xq_tf,
            b0_all_tf,
            db0_all_tf,
            f_all_tf,
            anchors,
            ell,
            aw,
            denom_tf,
            cfg,
            steps=adam_steps,
            seed=int(stage_seed),
            tag=f"{stage_name}:adam",
        )
        print(f"[{stage_name}] Adam done in {time.time() - t_a:.1f}s", flush=True)

    lbfgs_steps = int(cfg.LBFGS_MAXITER_FIRST if is_first else cfg.LBFGS_MAXITER_NEXT)
    if lbfgs_steps > 0:
        print(f"[{stage_name}] L-BFGS start | maxiter={lbfgs_steps}", flush=True)
        t_l = time.time()
        pf.lbfgs_optimize(
            unet,
            None,
            0,
            Xq_tf,
            anchors,
            ell,
            aw,
            cfg,
            tf.constant(0, dtype=tf.int32),
            tf.constant(1e-3, dtype=DTYPE),
            tf.constant(1.0, dtype=DTYPE),
            tf.constant(0, dtype=tf.int32),
            maxiter=int(lbfgs_steps),
            callback_every=10,
            b0_all_tf=b0_all_tf,
            db0_all_tf=db0_all_tf,
            denom_tf=denom_tf,
            cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
            pruner=None,
        )
        print(f"[{stage_name}] L-BFGS done in {time.time() - t_l:.1f}s", flush=True)

    print(f"[{stage_name}] full evaluation start", flush=True)
    t_e = time.time()
    ev = pf.eval_full_cached(
        unet,
        None,
        0,
        Xq_tf,
        b0_all_tf,
        db0_all_tf,
        anchors,
        ell,
        aw,
        G_full,
        cfg,
        tf.constant(0, dtype=tf.int32),
        tf.constant(1e-3, dtype=DTYPE),
        tf.constant(1.0, dtype=DTYPE),
        tf.constant(0, dtype=tf.int32),
        cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
    )
    print(f"[{stage_name}] full evaluation done in {time.time() - t_e:.1f}s | loss={float(ev['loss']):.3e}", flush=True)
    return ev, G_full


def build_initial_model(cfg: pf.Config) -> tf.keras.Model:
    unet = pf.MLP(width=cfg.WIDTH, layers=cfg.L_LAYERS, normalize_input=bool(cfg.DOMAIN_NORMALIZE))
    unet.call(tf.zeros((1, 2), dtype=DTYPE))
    return unet


def describe_stage(tag: str, M: int, h1: float, ev: Dict[str, Any], dt: float) -> None:
    print(
        f"[{tag}] M={int(M):4d} | loss={float(ev['loss']):.3e} | H1={float(h1):.3e} | time[s]={float(dt):.1f}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Marking helpers (same logic as the patchless Strategy 1/2 code)
# -----------------------------------------------------------------------------
def eta_from_eval(ev: Dict[str, Any], G_full: np.ndarray, cfg: pf.Config) -> np.ndarray:
    Rabs = np.asarray(ev.get("Rabs", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    if str(getattr(cfg, "WHITEN_MODE", "energy")) == "energy":
        denom = np.sqrt(np.maximum(np.asarray(G_full, dtype=np.float64).reshape(-1), 0.0) + float(cfg.WHITEN_EPS))
    else:
        denom = np.ones_like(Rabs, dtype=np.float64)
    return (Rabs / np.maximum(denom, 1e-30)) ** 2


def select_marked_parents(
    eta_a: np.ndarray,
    M: int,
    room: int,
    cm: int,
    cfg: pf.Config,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    eta_a = np.asarray(eta_a, dtype=np.float64).reshape(-1)
    idx = np.argsort(-eta_a)
    total = float(np.sum(eta_a))
    if total <= 0.0 or not np.isfinite(total):
        tau_tilde = 1
    else:
        cumsum = np.cumsum(eta_a[idx])
        tau_tilde = int(np.searchsorted(cumsum, float(getattr(cfg, "MARK_FRAC", 0.75)) * total) + 1)

    tau_cap = int(math.ceil(float(getattr(cfg, "MARK_CAP", 0.30)) * M))
    tau_m = max(1, min(tau_tilde, tau_cap))

    add_budget = int(min(cfg.MAX_ADD_PER_ITER, max(cfg.MIN_ADD_PER_ITER, math.ceil(cfg.ADD_FRAC_MAX * M))))
    add_budget = int(min(add_budget, room))
    parents_needed = int(math.ceil(float(add_budget) / float(max(1, cm))))
    parents_needed = max(1, min(M, parents_needed))

    marked_core = idx[:tau_m].astype(np.int32, copy=False)
    fill_global = bool(int(getattr(cfg, "STRATEGY1_FILL_GLOBAL", 1)) == 1)

    if marked_core.size >= parents_needed:
        marked_ids = marked_core[:parents_needed]
        tau_eff = int(marked_ids.size)
    else:
        marked_ids = idx[:parents_needed].astype(np.int32, copy=False) if fill_global else marked_core
        tau_eff = int(marked_ids.size)

    add_count = int(min(add_budget, cm * max(1, tau_eff)))
    stats = {
        "tau_tilde": int(tau_tilde),
        "tau_cap": int(tau_cap),
        "tau_m": int(tau_m),
        "tau_eff": int(tau_eff),
        "cm": int(cm),
        "add_budget": int(add_budget),
        "parents_needed": int(parents_needed),
    }
    return marked_ids, add_count, stats


# -----------------------------------------------------------------------------
# Figure 17 / Figure 18 run
# -----------------------------------------------------------------------------
def run_problem35_patchless(args: argparse.Namespace) -> Tuple[str, str]:
    ensure_dir(args.outdir)
    apply_tf_optimizer_options_from_args(args)
    set_all_seeds(args.seed)

    if str(args.lift).strip().lower() not in ("zero", "none", "exact", "coons", "gnet"):
        raise ValueError("Unsupported --lift value.")
    if str(args.lift).strip().lower() not in ("zero", "none"):
        print("[INFO] Problem (35) has homogeneous Dirichlet boundary data on ∂Ω₂; using zero lift.", flush=True)

    cfg = build_pf_config(args)
    aux_cfg = build_aux_cfg(args)
    domain = Problem35Domain(aux_cfg)
    install_problem35_overrides(domain)

    print("=== PATCHLESS Problem (35): Strategy #1/#2, C_M=4/9 ===", flush=True)
    print(f"Output dir: {args.outdir}", flush=True)
    print(f"Seed={cfg.SEED}  Anchor init={cfg.ANCHOR_INIT}  Base-add={args.base_add_count}  Branch refines={args.n_branch_refines}", flush=True)
    print(f"Kernel={cfg.KERNEL_KIND}  ell0={cfg.ELL0:g}  ell_min={cfg.ELL_MIN:g}  cutoff={cfg.KERNEL_CUTOFF:g}", flush=True)
    print(f"Training: adam_first={cfg.ADAM_STEPS_FIRST}  adam_after={cfg.ADAM_STEPS_AFTER_GROW}  lbfgs_first={cfg.LBFGS_MAXITER_FIRST}  lbfgs_next={cfg.LBFGS_MAXITER_NEXT}", flush=True)
    print(f"QMC: NQ_GLOBAL={cfg.NQ_GLOBAL}  NCAND={cfg.NCAND}  |Ω₂|≈{domain.area:.6f}", flush=True)

    # Fixed global quadrature cloud on Ω₂
    Xq_np, _ = domain.sample_domain_qmc(int(cfg.NQ_GLOBAL), seed=cfg.SEED + 11, cfg=cfg)
    Xq_tf = tf.constant(Xq_np, dtype=DTYPE)
    b0_all_tf, db0_all_tf = domain.phi_and_grad(Xq_tf)
    f_all_tf = p35.f_rhs_tf(Xq_tf)

    # ------------------------------------------------------------------
    # Common base stage A0
    # ------------------------------------------------------------------
    unet_base = build_initial_model(cfg)
    anchors0, ell0, aw0 = pf.init_anchors(cfg, seed=cfg.SEED + 99)

    t0 = time.time()
    ev0, G0 = train_stage(
        unet_base, anchors0, ell0, aw0,
        cfg, Xq_tf, b0_all_tf, db0_all_tf, f_all_tf,
        stage_seed=cfg.SEED + 1000,
        is_first=True,
        stage_name="Base A0",
    )
    h1_0 = rel_H1_error_omega2(unet_base, domain, aux_cfg, stage_name="Base A0")
    dt0 = time.time() - t0
    x0 = int(anchors0.shape[0])
    describe_stage("Base A0", x0, h1_0, ev0, dt0)

    # ------------------------------------------------------------------
    # Common base stage A1: global eta-driven enrichment, independent of strategy/CM
    # ------------------------------------------------------------------
    ref_ell_base = float(pf.ell_schedule(cfg, M=int(anchors0.shape[0]) + int(max(1, args.base_add_count)), M0=int(getattr(cfg, 'ANCHOR_INIT', max(1, int(anchors0.shape[0]))))))
    cfg.GROWTH_REF_ELL = ref_ell_base
    cand_base, _ = domain.sample_domain_qmc(int(cfg.NCAND), seed=cfg.SEED + 12345, cfg=cfg)
    anchors1, ell1, aw1, info_base = pf.add_anchors_adaptive(
        unet_base,
        None,
        0,
        anchors0,
        ell0,
        aw0,
        cand_base,
        add_count=int(max(1, args.base_add_count)),
        cfg=cfg,
        bubble_mode_tf=tf.constant(0, dtype=tf.int32),
        softmin_tau_tf=tf.constant(1e-3, dtype=DTYPE),
        bubble_power_tf=tf.constant(1.0, dtype=DTYPE),
        seed=cfg.SEED + 2000,
        use_eta=True,
    )
    print(
        f"[Base A1 grow] req={int(args.base_add_count)} added={info_base.get('added', 0)} tries={info_base.get('tries', 0)} reject_rate={100.0 * info_base.get('rejects', 0) / max(1, info_base.get('tries', 1)):.1f}%",
        flush=True,
    )

    t1 = time.time()
    ev1, G1 = train_stage(
        unet_base, anchors1, ell1, aw1,
        cfg, Xq_tf, b0_all_tf, db0_all_tf, f_all_tf,
        stage_seed=cfg.SEED + 3000,
        is_first=False,
        stage_name="Base A1",
    )
    h1_1 = rel_H1_error_omega2(unet_base, domain, aux_cfg, stage_name="Base A1")
    dt1 = time.time() - t1
    x1 = int(anchors1.shape[0])
    describe_stage("Base A1", x1, h1_1, ev1, dt1)

    base_weights = [np.array(w, copy=True) for w in unet_base.get_weights()]

    def branch_run(strategy_id: int, cm: int, tag: str):
        cfg_branch = copy.deepcopy(cfg)
        cfg_branch.ANCHOR_GROW_POLICY = "strategy1" if int(strategy_id) == 1 else "strategy2"
        cfg_branch.ANCHOR_CHILD_POLICY = "strategy1" if int(strategy_id) == 1 else "strategy2"
        cfg_branch.ANCHOR_CM = int(cm)

        unet = build_initial_model(cfg_branch)
        unet.set_weights([np.array(w, copy=True) for w in base_weights])

        anchors = np.asarray(anchors1, dtype=np.float64).copy()
        ell = np.asarray(ell1, dtype=np.float64).copy()
        aw = np.asarray(aw1, dtype=np.float64).copy()

        xs = [int(x0), int(x1)]
        ys = [float(h1_0), float(h1_1)]
        ev_cur = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in ev1.items()}
        G_cur = np.asarray(G1, dtype=np.float64).copy()

        for k in range(int(args.n_branch_refines)):
            M = int(anchors.shape[0])
            room = int(max(0, int(cfg_branch.TARGET_NANCHOR) - M)) if int(cfg_branch.TARGET_NANCHOR) > 0 else int(cfg_branch.MAX_ADD_PER_ITER)
            if room <= 0:
                print(f"[{tag}] room=0, stopping branch.", flush=True)
                break

            eta_a = eta_from_eval(ev_cur, G_cur, cfg_branch)
            ref_ell_branch = float(pf.ell_schedule(cfg_branch, M=int(M) + int(max(1, room)), M0=int(getattr(cfg_branch, 'ANCHOR_INIT', max(1, int(M))))))
            cfg_branch.GROWTH_REF_ELL = ref_ell_branch
            marked_ids, add_count, mark_stats = select_marked_parents(
                eta_a, M=M, room=room, cm=int(cm), cfg=cfg_branch
            )
            if add_count <= 0 or marked_ids.size == 0:
                print(f"[{tag}] add_count=0 or no marked parents, stopping branch.", flush=True)
                break

            anchors, ell, aw, info = pf.add_anchors_strategy1_cm_meshfree(
                unet,
                None,
                0,
                anchors,
                ell,
                aw,
                marked_ids=marked_ids,
                cm=int(cm),
                add_count=int(add_count),
                child_policy=("strategy1" if int(strategy_id) == 1 else "strategy2"),
                cfg=cfg_branch,
                bubble_mode_tf=tf.constant(0, dtype=tf.int32),
                softmin_tau_tf=tf.constant(1e-3, dtype=DTYPE),
                bubble_power_tf=tf.constant(1.0, dtype=DTYPE),
                seed=cfg_branch.SEED + 5000 + 97 * int(strategy_id) + 13 * int(cm) + k,
                use_eta=(int(strategy_id) == 1),
            )

            rej_rate = float(info.get("rejects", 0)) / max(1, int(info.get("tries", 1)))
            print(
                f"[{tag}] grow {k + 1}: req={add_count} added={info.get('added', 0)} tries={info.get('tries', 0)} reject_rate={100.0 * rej_rate:.1f}%  min_sep={info.get('min_sep', float('nan')):.3g}  base_ell={info.get('base_ell', float('nan')):.3g}  mark:(tau~={mark_stats['tau_tilde']},cap={mark_stats['tau_cap']},tau={mark_stats['tau_m']},parents={mark_stats['tau_eff']},req={mark_stats['add_budget']})",
                flush=True,
            )

            t_stage = time.time()
            ev_cur, G_cur = train_stage(
                unet, anchors, ell, aw,
                cfg_branch, Xq_tf, b0_all_tf, db0_all_tf, f_all_tf,
                stage_seed=cfg_branch.SEED + 6000 + 173 * int(strategy_id) + 19 * int(cm) + k,
                is_first=False,
                stage_name=f"{tag} step {k + 2}",
            )
            h1 = rel_H1_error_omega2(unet, domain, aux_cfg, stage_name=f"{tag} step {k + 2}")
            dt_stage = time.time() - t_stage
            M_now = int(anchors.shape[0])
            xs.append(M_now)
            ys.append(float(h1))
            describe_stage(f"{tag} step {k + 2}", M_now, h1, ev_cur, dt_stage)

        h2 = np.asarray(ell, dtype=np.float64).reshape(-1) ** 2
        r2 = np.asarray(ev_cur.get("Rabs", np.zeros((anchors.shape[0],), dtype=np.float64)), dtype=np.float64).reshape(-1) ** 2
        snap = {
            "centers": np.asarray(anchors, dtype=np.float64).copy(),
            "h2": np.asarray(h2, dtype=np.float64).copy(),
            "r2": np.asarray(r2, dtype=np.float64).copy(),
        }
        return xs, ys, snap

    curves: Dict[str, Tuple[List[int], List[float]]] = {}
    snaps18: Dict[str, Dict[str, np.ndarray]] = {}

    x, y, snap = branch_run(1, 4, "S1 CM=4")
    curves["S1_CM4"] = (x, y)
    snaps18["S1_CM4"] = snap

    x, y, snap = branch_run(2, 4, "S2 CM=4")
    curves["S2_CM4"] = (x, y)
    snaps18["S2_CM4"] = snap

    x, y, snap = branch_run(1, 9, "S1 CM=9")
    curves["S1_CM9"] = (x, y)
    snaps18["S1_CM9"] = snap

    x, y, snap = branch_run(2, 9, "S2 CM=9")
    curves["S2_CM9"] = (x, y)
    snaps18["S2_CM9"] = snap

    # ---- Figure 17 plot ----
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=200)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of test functions")
    ax.set_ylabel(r"Relative $H^1$ error")

    x, y = curves["S1_CM4"]
    ax.plot(x, y, marker="o", linewidth=1.5, markersize=4, label=r"Strategy \#1 $C_M=4$")
    x, y = curves["S2_CM4"]
    ax.plot(x, y, marker="^", linewidth=1.5, markersize=4, label=r"Strategy \#2 $C_M=4$")
    x, y = curves["S1_CM9"]
    ax.plot(x, y, marker="v", linewidth=1.5, markersize=4, label=r"Strategy \#1 $C_M=9$")
    x, y = curves["S2_CM9"]
    ax.plot(x, y, marker="s", linewidth=1.5, markersize=4, label=r"Strategy \#2 $C_M=9$")

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out_fig17 = os.path.join(args.outdir, "figure17_problem35_patchless.png")
    fig.savefig(out_fig17, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 17 curves NPZ ----
    out_curves = os.path.join(args.outdir, "figure17_problem35_patchless_curves.npz")
    np.savez(
        out_curves,
        **{k + "_x": np.asarray(v[0], np.float64) for k, v in curves.items()},
        **{k + "_y": np.asarray(v[1], np.float64) for k, v in curves.items()},
        base_x=np.asarray([x0, x1], np.float64),
        base_y=np.asarray([h1_0, h1_1], np.float64),
    )

    # ---- Figure 18 NPZ payload ----
    out_npz18 = os.path.join(args.outdir, "figure18_problem35_patchless.npz")
    npz18: Dict[str, np.ndarray] = {"holes_xyxy": np.asarray(domain.holes_xyxy, dtype=np.float64)}
    for key, snap in snaps18.items():
        npz18[f"{key}_centers"] = np.asarray(snap["centers"], dtype=np.float64)
        npz18[f"{key}_h2"] = np.asarray(snap["h2"], dtype=np.float64)
        npz18[f"{key}_r2"] = np.asarray(snap["r2"], dtype=np.float64)
    np.savez(out_npz18, **npz18)

    print(f"Saved: {out_fig17}", flush=True)
    print(f"Saved: {out_curves}", flush=True)
    print(f"Saved: {out_npz18}", flush=True)
    return out_fig17, out_npz18


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Patchless Problem (35): Strategy #1/#2 for C_M=4/9, with Figure 17/18 outputs."
    )
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--lift", type=str, default="zero", choices=["zero", "none", "exact", "coons", "gnet"])

    # Common/base stages
    ap.add_argument("--anchor-init", type=int, default=25)
    ap.add_argument("--base-add-count", type=int, default=25,
                    help="Common A0 -> A1 enrichment count, shared by all four curves.")
    ap.add_argument("--n-branch-refines", type=int, default=2,
                    help="Number of strategy-dependent refinements after the common A1 stage.")
    ap.add_argument("--target-nanchor", type=int, default=1000)

    # Network / optimizer
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=1e-2)
    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--lambda-reg", type=float, default=1e-5)
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-energy", type=float, default=0.0)

    ap.add_argument("--adam-steps-first", type=int, default=2000)
    ap.add_argument("--adam-steps-after-grow", type=int, default=1200)
    ap.add_argument("--lbfgs-first", type=int, default=0)
    ap.add_argument("--lbfgs-next", type=int, default=0)
    ap.add_argument("--lbfgs-ftol", type=float, default=1e-14)
    ap.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    ap.add_argument("--lbfgs-maxcor", type=int, default=50)
    ap.add_argument("--lbfgs-maxfun", type=int, default=0)

    # Batching / quadrature
    ap.add_argument("--adam-point-batch", type=int, default=4096)
    ap.add_argument("--adam-anchor-batch", type=int, default=64)
    ap.add_argument("--adam-grad-accum", type=int, default=16)
    ap.add_argument("--eval-point-batch", type=int, default=8192)
    ap.add_argument("--eval-anchor-batch", type=int, default=128)
    ap.add_argument("--lbfgs-point-batch", type=int, default=8192)
    ap.add_argument("--lbfgs-anchor-batch", type=int, default=128)
    ap.add_argument("--nq-global", type=int, default=16384)
    ap.add_argument("--ncand", type=int, default=16384)
    ap.add_argument("--qmc-kind", type=str, default="sobol", choices=["sobol", "halton"])
    ap.add_argument("--qmc-scramble", type=int, default=1)

    # Kernel / scales
    ap.add_argument("--kernel", type=str, default="wendland_c2", choices=["gauss", "wendland_c2"])
    ap.add_argument("--ell0", type=float, default=0.15)
    ap.add_argument("--ell-min", type=float, default=0.015)
    ap.add_argument("--ell-sched-p", type=float, default=0.5)
    ap.add_argument("--cutoff", type=float, default=3.6)
    ap.add_argument("--ell-from-h", type=int, default=1)
    ap.add_argument("--ell-c-rho", type=float, default=2.0)
    ap.add_argument("--domain-normalize", type=int, default=1)
    ap.add_argument("--whiten-eps", type=float, default=1e-12)

    # Growth / marking
    ap.add_argument("--add-frac-max", type=float, default=1.0)
    ap.add_argument("--min-add", type=int, default=16)
    ap.add_argument("--max-add", type=int, default=4096)
    ap.add_argument("--mark-frac", type=float, default=0.75)
    ap.add_argument("--mark-cap", type=float, default=0.30)
    ap.add_argument("--child-a-ratio", type=float, default=1.25)
    ap.add_argument("--child-lam-lo", type=float, default=0.9)
    ap.add_argument("--child-lam-hi", type=float, default=10.0 / 9.0)
    ap.add_argument("--child-oversample", type=int, default=2)
    ap.add_argument("--fill-global", type=int, default=1)
    ap.add_argument("--min-sep-alpha", type=float, default=0.7)
    ap.add_argument("--reject-max-tries", type=int, default=200000)
    ap.add_argument("--eta-power", type=float, default=1.5)
    ap.add_argument("--eta-eps", type=float, default=1e-12)

    # Problem (35) ADF / H1 quadrature
    ap.add_argument("--adf-m", type=int, default=2)
    ap.add_argument("--adf-eps", type=float, default=1e-12)
    ap.add_argument("--h1-ncells", type=int, default=221)
    ap.add_argument("--h1-gauss-n", type=int, default=4)
    ap.add_argument("--h1-eval-chunk", type=int, default=65536)

    # extra CLI compatibility / runtime controls
    ap.add_argument("--snapshot-every", type=int, default=1,
                    help="Accepted for CLI compatibility; currently a no-op in this runner.")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="Print Adam progress every this many steps.")
    ap.add_argument("--disable-meta-optimizer", type=int, default=1, choices=[0, 1],
                    help="If 1, disable TF Grappler meta optimizer for stability.")
    ap.add_argument("--disable-arithmetic-optimizer", type=int, default=1, choices=[0, 1],
                    help="If 1, disable TF arithmetic optimization pass.")
    ap.add_argument("--disable-dependency-optimizer", type=int, default=1, choices=[0, 1],
                    help="If 1, disable TF dependency optimization pass.")

    # hole-aware growth controls
    ap.add_argument("--geo-bias-hole", type=float, default=1.75)
    ap.add_argument("--geo-bias-outer", type=float, default=0.85)
    ap.add_argument("--geo-bias-radius-mult", type=float, default=2.5)
    ap.add_argument("--hole-priority-factor", type=float, default=1.15)
    ap.add_argument("--cand-global-frac", type=float, default=0.40)
    ap.add_argument("--cand-hole-frac", type=float, default=0.40)
    ap.add_argument("--cand-outer-frac", type=float, default=0.20)
    ap.add_argument("--hole-band-mult", type=float, default=2.5)
    ap.add_argument("--outer-band-mult", type=float, default=2.0)

    args, unknown = ap.parse_known_args(list(argv) if argv is not None else None)
    if unknown:
        print(f"[warn] ignoring unknown CLI arguments: {unknown}", flush=True)
    run_problem35_patchless(args)


if __name__ == "__main__":
    main()
