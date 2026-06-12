#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reference_vpinn_figure13.py

Reference VPINN for reproducing the standard Delaunay-mesh VPINN curve
used as the baseline in Figure 13 of:

    S. Berrone and M. Pintore,
    Meshfree Variational-Physics-Informed Neural Networks (MF-VPINN):
    an adaptive training strategy, Algorithms 17(9), 415, 2024.

This script computes the *Reference VPINN* curve only. It does not implement
MF-VPINN patch strategies. That distinction matters: CM=4 and CM=9 are
MF-VPINN patch multiplication factors, not parameters of the standard VPINN.

Problem used in the first numerical test of the MF-VPINN paper:
    -Delta u = 0 in Omega=(0,1)^2,
    u = u_exact on boundary,

with the corner-singular harmonic exact solution
    u(r,theta) = r^(2/3) sin( (2/3) (theta + pi/2) ).

Reference VPINN implementation details:
    * global Delaunay triangulations of the unit square;
    * continuous P1 nodal Lagrange test functions on interior mesh nodes;
    * Petrov-Galerkin weak residuals against all test functions;
    * Gaussian quadrature on every triangle;
    * trial function u_NN = g + phi*w_NN, where phi vanishes on the boundary;
    * first level: ADAM + L-BFGS-B; later levels: warm-started L-BFGS-B;
    * relative H1 error evaluated by an independent graded high-order quadrature;
    * professional logging and CSV/JSON/NPZ/PNG outputs.

The defaults are chosen to be faithful to the numerical setup described in the
paper while remaining executable on a normal workstation. For a publication-grade
run, increase --adam-steps, --lbfgs-first, --lbfgs-next, and use --lift gnet.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import scipy.optimize as sopt
from scipy.spatial import Delaunay

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

tf.keras.backend.set_floatx("float64")
DTYPE = tf.float64


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Neural network architecture: input dimension 2, scalar output.
    layers: int = 5
    width: int = 50
    activation: str = "tanh"

    # VPINN loss and regularization.
    lambda_reg: float = 1.0e-5
    gamma_mode: str = "support"  # support, mass, none

    # Boundary enforcement: paper-like default is gnet.
    lift: str = "gnet"           # gnet, coons, zero, debug_exact
    bubble: str = "product"      # product, softmin
    bubble_scale: float = 16.0    # product bubble max is 1 at center when scale=16
    softmin_tau: float = 3.0e-3
    bubble_power: float = 1.0

    # Mesh sequence. If mesh_ns is empty, it is built from target_tests.
    mesh_ns: Tuple[int, ...] = (3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    target_tests: Tuple[int, ...] = ()
    jitter: float = 0.03          # interior jitter amplitude measured in local h; 0 gives structured grid
    seed: int = 2000

    # Triangle quadrature for VPINN residuals.
    q_tri: int = 5                # Duffy tensor Gauss rule: q_tri^2 points per triangle

    # H1 error quadrature. Graded grid helps integrate the singular gradient near origin.
    h1_ncells: int = 96
    h1_grade_p: float = 3.0
    h1_gauss_n: int = 7
    eval_chunk: int = 32768

    # Optimizer settings.
    adam_steps: int = 3000
    adam_lr0: float = 1.0e-2
    adam_lr1: float = 1.0e-4
    lbfgs_first: int = 800
    lbfgs_next: int = 1200
    lbfgs_ftol: float = 1.0e-14
    lbfgs_gtol: float = 1.0e-14
    lbfgs_maxcor: int = 50
    lbfgs_callback_every: int = 10

    # Gnet boundary lift training.
    gnet_steps: int = 20000
    gnet_lr: float = 1.0e-3
    gnet_batch: int = 8192
    gnet_print_every: int = 2000
    gnet_corner_power: float = 10.0
    gnet_p_focus: float = 0.70
    gnet_p_norm: float = 8.0
    gnet_beta_p: float = 0.05
    gnet_beta_top: float = 0.50
    gnet_top_frac: float = 0.10
    gnet_grid_n_per_edge: int = 20000
    gnet_maxerr_warn: float = 5.0e-3

    # Output and execution.
    outdir: str = "results_reference_vpinn_figure13"
    save_meshes: bool = True
    warm_start: bool = True
    smoke: bool = False


# =============================================================================
# Utilities and exact solution
# =============================================================================

ALPHA = 2.0 / 3.0


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def u_exact_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    out = np.zeros_like(r, dtype=np.float64)
    mask = r > 0.0
    out[mask] = (r[mask] ** ALPHA) * np.sin(ALPHA * (th[mask] + np.pi / 2.0))
    return out


def grad_u_exact_np(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    gx = np.zeros_like(r, dtype=np.float64)
    gy = np.zeros_like(r, dtype=np.float64)
    mask = r > 0.0
    angle = ALPHA * (th[mask] + np.pi / 2.0) - th[mask]
    coeff = ALPHA * (r[mask] ** (ALPHA - 1.0))
    gx[mask] = coeff * np.sin(angle)
    gy[mask] = coeff * np.cos(angle)
    return gx, gy


@tf.function
def u_exact_tf(xy: tf.Tensor) -> tf.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    r = tf.sqrt(tf.maximum(x * x + y * y, tf.constant(0.0, DTYPE)))
    th = tf.atan2(y, x)
    a = tf.constant(ALPHA, dtype=DTYPE)
    out = tf.pow(r, a) * tf.sin(a * (th + tf.constant(np.pi / 2.0, dtype=DTYPE)))
    return tf.where(r > tf.constant(0.0, dtype=DTYPE), out, tf.zeros_like(out))


@tf.function
def g_coons_tf(xy: tf.Tensor) -> tf.Tensor:
    """Coons transfinite interpolation of exact boundary values on the unit square."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    z0 = tf.zeros_like(x)
    z1 = tf.ones_like(x)

    u_left = u_exact_tf(tf.concat([z0, y], axis=1))
    u_right = u_exact_tf(tf.concat([z1, y], axis=1))
    u_bottom = u_exact_tf(tf.concat([x, z0], axis=1))
    u_top = u_exact_tf(tf.concat([x, z1], axis=1))

    u00 = u_exact_tf(tf.constant([[0.0, 0.0]], dtype=DTYPE))
    u10 = u_exact_tf(tf.constant([[1.0, 0.0]], dtype=DTYPE))
    u01 = u_exact_tf(tf.constant([[0.0, 1.0]], dtype=DTYPE))
    u11 = u_exact_tf(tf.constant([[1.0, 1.0]], dtype=DTYPE))

    edge_blend = (1.0 - x) * u_left + x * u_right + (1.0 - y) * u_bottom + y * u_top
    corner_blend = (
        (1.0 - x) * (1.0 - y) * u00
        + x * (1.0 - y) * u10
        + (1.0 - x) * y * u01
        + x * y * u11
    )
    return edge_blend - corner_blend


@tf.function
def bubble_phi_tf(xy: tf.Tensor, cfg_tuple: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Boundary bubble. cfg_tuple = (scale, mode, tau, power), mode 0 product, 1 softmin."""
    scale, mode, tau, power = cfg_tuple
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    def product() -> tf.Tensor:
        base = x * (1.0 - x) * y * (1.0 - y)
        return scale * tf.pow(tf.maximum(base, tf.constant(0.0, DTYPE)), power)

    def softmin() -> tf.Tensor:
        d = tf.concat([x, 1.0 - x, y, 1.0 - y], axis=1)
        sm = -tau * (tf.reduce_logsumexp(-d / tau, axis=1, keepdims=True) - tf.math.log(tf.constant(4.0, DTYPE)))
        sm = tf.maximum(sm, tf.constant(0.0, DTYPE))
        return scale * tf.pow(sm, power)

    return tf.switch_case(tf.cast(mode, tf.int32), branch_fns=[product, softmin], default=product)


# =============================================================================
# Neural networks and boundary lift
# =============================================================================

class MLP(tf.keras.Model):
    def __init__(self, width: int, layers: int, activation: str = "tanh", name: str = "mlp"):
        super().__init__(name=name)
        if layers < 2:
            raise ValueError("layers must be at least 2")
        if activation != "tanh":
            raise ValueError("This reproduction script uses tanh, as in the paper setup.")
        init = tf.keras.initializers.GlorotNormal()
        self.hidden = [
            tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer=init, dtype=DTYPE)
            for _ in range(layers - 1)
        ]
        self.out = tf.keras.layers.Dense(1, activation=None, kernel_initializer=init, dtype=DTYPE)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z = x
        for layer in self.hidden:
            z = layer(z)
        return self.out(z)


def sample_boundary_corner_biased(rng: np.random.Generator, n: int, power: float, p_focus: float) -> np.ndarray:
    """Boundary samples, with extra density near the singular corner on x=0 and y=0."""
    n = int(n)
    xy = np.empty((n, 2), dtype=np.float64)
    focus_mask = rng.random(n) < float(p_focus)
    nf = int(np.sum(focus_mask))

    if nf:
        edge = rng.integers(0, 2, size=nf)
        t = rng.random(nf) ** float(power)
        pts = np.zeros((nf, 2), dtype=np.float64)
        pts[edge == 0, 1] = t[edge == 0]  # x=0
        pts[edge == 1, 0] = t[edge == 1]  # y=0
        xy[focus_mask] = pts

    nr = n - nf
    if nr:
        side = rng.integers(0, 4, size=nr)
        t = rng.random(nr)
        pts = np.empty((nr, 2), dtype=np.float64)
        m = side == 0
        pts[m] = np.stack([np.zeros(np.sum(m)), t[m]], axis=1)
        m = side == 1
        pts[m] = np.stack([np.ones(np.sum(m)), t[m]], axis=1)
        m = side == 2
        pts[m] = np.stack([t[m], np.zeros(np.sum(m))], axis=1)
        m = side == 3
        pts[m] = np.stack([t[m], np.ones(np.sum(m))], axis=1)
        xy[~focus_mask] = pts
    return xy


def boundary_max_error(gnet: MLP, n_per_edge: int, chunk: int) -> float:
    t = np.linspace(0.0, 1.0, int(n_per_edge), dtype=np.float64)
    xy = np.vstack([
        np.stack([np.zeros_like(t), t], axis=1),
        np.stack([np.ones_like(t), t], axis=1),
        np.stack([t, np.zeros_like(t)], axis=1),
        np.stack([t, np.ones_like(t)], axis=1),
    ])
    truth = u_exact_np(xy[:, 0], xy[:, 1]).reshape(-1, 1)
    pred = np.zeros_like(truth)
    chunk = int(chunk)
    for s in range(0, xy.shape[0], chunk):
        e = min(s + chunk, xy.shape[0])
        pred[s:e] = gnet.call(tf.constant(xy[s:e], dtype=DTYPE)).numpy()
    return float(np.max(np.abs(pred - truth)))


def train_boundary_gnet(cfg: Config, logger: logging.Logger) -> MLP:
    """Train a non-variational network extension g matching boundary data."""
    logger.info("Training boundary lift network gnet: steps=%d batch=%d", cfg.gnet_steps, cfg.gnet_batch)
    rng = np.random.default_rng(cfg.seed + 991)
    gnet = MLP(cfg.width, cfg.layers, cfg.activation, name="gnet")
    _ = gnet.call(tf.zeros((1, 2), dtype=DTYPE))

    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.gnet_lr,
        decay_steps=2000,
        decay_rate=0.9,
        staircase=False,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    p_norm = tf.constant(cfg.gnet_p_norm, dtype=DTYPE)
    beta_p = tf.constant(cfg.gnet_beta_p, dtype=DTYPE)
    beta_top = tf.constant(cfg.gnet_beta_top, dtype=DTYPE)
    top_frac = float(cfg.gnet_top_frac)

    @tf.function
    def step(xy: tf.Tensor, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            err = gnet.call(xy) - target
            l2 = tf.reduce_mean(tf.square(err))
            lp = tf.reduce_mean(tf.pow(tf.abs(err) + tf.constant(1.0e-14, DTYPE), p_norm))
            abs_err = tf.reshape(tf.abs(err), [-1])
            k = tf.maximum(1, tf.cast(tf.round(top_frac * tf.cast(tf.size(abs_err), DTYPE)), tf.int32))
            top = tf.nn.top_k(abs_err, k=k, sorted=False).values
            ltop = tf.reduce_mean(tf.square(top))
            loss = l2 + beta_p * lp + beta_top * ltop
        grads = tape.gradient(loss, gnet.trainable_variables)
        opt.apply_gradients(zip(grads, gnet.trainable_variables))
        return loss, l2, ltop

    for it in range(1, int(cfg.gnet_steps) + 1):
        xy = sample_boundary_corner_biased(
            rng,
            cfg.gnet_batch,
            power=cfg.gnet_corner_power,
            p_focus=cfg.gnet_p_focus,
        )
        y = u_exact_np(xy[:, 0], xy[:, 1]).reshape(-1, 1)
        loss, l2, ltop = step(tf.constant(xy, dtype=DTYPE), tf.constant(y, dtype=DTYPE))
        if cfg.gnet_print_every and it % int(cfg.gnet_print_every) == 0:
            logger.info("gnet step=%05d loss=%.3e l2=%.3e top=%.3e", it, loss.numpy(), l2.numpy(), ltop.numpy())

    maxerr = boundary_max_error(gnet, cfg.gnet_grid_n_per_edge, cfg.eval_chunk)
    logger.info("gnet deterministic boundary max error = %.6e", maxerr)
    if maxerr > cfg.gnet_maxerr_warn:
        logger.warning("gnet boundary error is above %.2e; Figure-13 trends may be contaminated by boundary lift error.", cfg.gnet_maxerr_warn)
    return gnet


def lifted_u_and_grad(
    unet: MLP,
    gnet: Optional[MLP],
    xy: tf.Tensor,
    cfg: Config,
    bubble_cfg: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return u_NN and grad u_NN at xy. The inner tape differentiates wrt xy."""
    with tf.GradientTape(watch_accessed_variables=False) as tape_xy:
        tape_xy.watch(xy)
        w = unet.call(xy)
        phi = bubble_phi_tf(xy, bubble_cfg)
        if cfg.lift == "gnet":
            if gnet is None:
                raise RuntimeError("lift='gnet' requires a trained gnet")
            g = gnet.call(xy)
        elif cfg.lift == "coons":
            g = g_coons_tf(xy)
        elif cfg.lift == "zero":
            g = tf.zeros_like(w)
        elif cfg.lift == "debug_exact":
            # This makes the problem trivial if used for production. It is only a sanity check.
            g = u_exact_tf(xy)
        else:
            raise ValueError(f"Unknown lift mode: {cfg.lift}")
        u = g + phi * w
    grad = tape_xy.gradient(u, xy)
    if grad is None:
        raise RuntimeError("Failed to compute spatial gradient of the lifted network.")
    return u, grad


# =============================================================================
# Quadrature and Delaunay P1 test space
# =============================================================================

def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    return (0.5 * (x + 1.0)).astype(np.float64), (0.5 * w).astype(np.float64)


def tri_ref_duffy(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor-product Duffy quadrature on reference triangle (0,0),(1,0),(0,1)."""
    r, wr = gauss_legendre_01(n)
    t, wt = gauss_legendre_01(n)
    rr, tt = np.meshgrid(r, t, indexing="ij")
    wrr, wtt = np.meshgrid(wr, wt, indexing="ij")
    x = rr.reshape(-1)
    y = ((1.0 - rr) * tt).reshape(-1)
    w = (wrr * wtt * (1.0 - rr)).reshape(-1)
    phi = np.stack([1.0 - x - y, x, y], axis=1)
    pts = np.stack([x, y], axis=1)
    return pts.astype(np.float64), w.astype(np.float64), phi.astype(np.float64)


def make_delaunay_unit_square(n: int, jitter: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a quasi-uniform Delaunay triangulation of [0,1]^2.

    n is the number of subdivisions per coordinate before triangulation. The
    exact number of interior P1 test functions is (n-1)^2.
    """
    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, n + 1)
    ys = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float64)

    boundary = (np.isclose(pts[:, 0], 0.0) | np.isclose(pts[:, 0], 1.0) |
                np.isclose(pts[:, 1], 0.0) | np.isclose(pts[:, 1], 1.0))
    if jitter > 0.0:
        h = 1.0 / float(n)
        noise = rng.uniform(-0.5, 0.5, size=pts.shape) * (float(jitter) * h)
        pts[~boundary] += noise[~boundary]
        # Keep interior nodes strictly inside the square, away from boundary by a tiny amount.
        eps = 1.0e-13
        pts[~boundary, 0] = np.clip(pts[~boundary, 0], eps, 1.0 - eps)
        pts[~boundary, 1] = np.clip(pts[~boundary, 1], eps, 1.0 - eps)

    tri = Delaunay(pts)
    simplices = tri.simplices.astype(np.int64)

    # Remove zero-area triangles and orient counter-clockwise.
    good = []
    for s in simplices:
        a, b, c = pts[s]
        det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(det) < 1.0e-15:
            continue
        if det < 0.0:
            s = np.array([s[0], s[2], s[1]], dtype=np.int64)
        good.append(s)
    triangles = np.asarray(good, dtype=np.int64)
    interior_nodes = np.where(~boundary)[0].astype(np.int64)
    return pts, triangles, interior_nodes, boundary.astype(bool)


@dataclass
class VPINNBankNP:
    points: np.ndarray
    triangles: np.ndarray
    interior_nodes: np.ndarray
    boundary_mask: np.ndarray
    xyq: np.ndarray
    q_ids: np.ndarray
    test_ids: np.ndarray
    contrib_w: np.ndarray
    contrib_phi: np.ndarray
    contrib_gradv: np.ndarray
    gamma: np.ndarray
    support_area: np.ndarray
    n_test: int
    n_q: int
    n_contrib: int
    n_tri: int


@dataclass
class VPINNBankTF:
    xyq: tf.Tensor
    q_ids: tf.Tensor
    test_ids: tf.Tensor
    contrib_w: tf.Tensor
    contrib_phi: tf.Tensor
    contrib_gradv: tf.Tensor
    gamma: tf.Tensor
    n_test: int


def build_vpinn_bank(points: np.ndarray, triangles: np.ndarray, interior_nodes: np.ndarray, boundary_mask: np.ndarray, cfg: Config) -> VPINNBankNP:
    ref_pts, ref_w, ref_phi = tri_ref_duffy(cfg.q_tri)
    ref_grad = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    n_refq = ref_pts.shape[0]

    node_to_test = -np.ones(points.shape[0], dtype=np.int64)
    node_to_test[interior_nodes] = np.arange(interior_nodes.size, dtype=np.int64)
    n_test = int(interior_nodes.size)

    xyq_list: List[np.ndarray] = []
    q_ids: List[np.ndarray] = []
    test_ids: List[np.ndarray] = []
    cw: List[np.ndarray] = []
    cphi: List[np.ndarray] = []
    cgv: List[np.ndarray] = []
    support_area = np.zeros(n_test, dtype=np.float64)

    q_base = 0
    for tri_nodes in triangles:
        A = points[tri_nodes[0]]
        B = points[tri_nodes[1]]
        C = points[tri_nodes[2]]
        M = np.column_stack([B - A, C - A])  # x = A + M*[r,s]
        det = float(np.linalg.det(M))
        if det <= 0.0:
            raise RuntimeError("Triangles must be positively oriented.")
        area = 0.5 * det
        xy = A[None, :] + ref_pts @ M.T
        weights = ref_w * det
        grad_phys = ref_grad @ np.linalg.inv(M)  # grad_x phi = grad_ref phi * M^{-1}
        xyq_list.append(xy)

        for loc in range(3):
            tid = node_to_test[tri_nodes[loc]]
            if tid < 0:
                continue
            support_area[tid] += area
            qloc = q_base + np.arange(n_refq, dtype=np.int64)
            q_ids.append(qloc)
            test_ids.append(np.full(n_refq, int(tid), dtype=np.int64))
            cw.append(weights.reshape(-1, 1))
            cphi.append(ref_phi[:, loc:loc + 1])
            cgv.append(np.tile(grad_phys[loc:loc + 1, :], (n_refq, 1)))
        q_base += n_refq

    if n_test <= 0:
        raise ValueError("Mesh has no interior test functions. Use n>=3.")
    if not np.all(support_area > 0.0):
        bad = np.where(support_area <= 0.0)[0]
        raise RuntimeError(f"Interior nodes with zero support area: {bad[:10]}")

    xyq = np.vstack(xyq_list).astype(np.float64)
    q_ids_a = np.concatenate(q_ids).astype(np.int32)
    test_ids_a = np.concatenate(test_ids).astype(np.int32)
    cw_a = np.vstack(cw).astype(np.float64)
    cphi_a = np.vstack(cphi).astype(np.float64)
    cgv_a = np.vstack(cgv).astype(np.float64)

    if cfg.gamma_mode == "support":
        gamma = 1.0 / np.maximum(support_area, 1.0e-300)
    elif cfg.gamma_mode == "mass":
        # P1 basis has integral phi_i^2 over each triangle = area/6.
        gamma = 6.0 / np.maximum(support_area, 1.0e-300)
    elif cfg.gamma_mode == "none":
        gamma = np.ones_like(support_area)
    else:
        raise ValueError("gamma_mode must be support, mass, or none")

    return VPINNBankNP(
        points=points,
        triangles=triangles,
        interior_nodes=interior_nodes,
        boundary_mask=boundary_mask,
        xyq=xyq,
        q_ids=q_ids_a,
        test_ids=test_ids_a,
        contrib_w=cw_a,
        contrib_phi=cphi_a,
        contrib_gradv=cgv_a,
        gamma=gamma.reshape(-1, 1).astype(np.float64),
        support_area=support_area,
        n_test=n_test,
        n_q=int(xyq.shape[0]),
        n_contrib=int(q_ids_a.size),
        n_tri=int(triangles.shape[0]),
    )


def bank_to_tf(bank: VPINNBankNP) -> VPINNBankTF:
    # Store static quadrature/test data as tensors. They are not trainable.
    with tf.device("/CPU:0"):
        return VPINNBankTF(
            xyq=tf.constant(bank.xyq, dtype=DTYPE),
            q_ids=tf.constant(bank.q_ids, dtype=tf.int32),
            test_ids=tf.constant(bank.test_ids, dtype=tf.int32),
            contrib_w=tf.constant(bank.contrib_w, dtype=DTYPE),
            contrib_phi=tf.constant(bank.contrib_phi, dtype=DTYPE),
            contrib_gradv=tf.constant(bank.contrib_gradv, dtype=DTYPE),
            gamma=tf.constant(bank.gamma, dtype=DTYPE),
            n_test=int(bank.n_test),
        )


# =============================================================================
# VPINN loss and optimizers
# =============================================================================

def make_bubble_cfg(cfg: Config) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    mode = 0 if cfg.bubble == "product" else 1
    return (
        tf.constant(cfg.bubble_scale, dtype=DTYPE),
        tf.constant(mode, dtype=tf.int32),
        tf.constant(cfg.softmin_tau, dtype=DTYPE),
        tf.constant(cfg.bubble_power, dtype=DTYPE),
    )


def residual_vector(unet: MLP, gnet: Optional[MLP], bank: VPINNBankTF, cfg: Config, bubble_cfg) -> tf.Tensor:
    # For this benchmark: mu=1, beta=0, sigma=0, f=0.
    # General lower-order terms are left visible in the tensor structure through contrib_phi.
    _, grad_u = lifted_u_and_grad(unet, gnet, bank.xyq, cfg, bubble_cfg)
    grad_gather = tf.gather(grad_u, bank.q_ids)
    integrand = tf.reduce_sum(grad_gather * bank.contrib_gradv, axis=1, keepdims=True)
    contrib = bank.contrib_w * integrand
    return tf.math.unsorted_segment_sum(contrib, bank.test_ids, bank.n_test)


def vpinn_loss(unet: MLP, gnet: Optional[MLP], bank: VPINNBankTF, cfg: Config, bubble_cfg) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    R = residual_vector(unet, gnet, bank, cfg, bubble_cfg)
    residual_loss = tf.reduce_mean(bank.gamma * tf.square(R))
    if unet.trainable_variables:
        reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in unet.trainable_variables])
    else:
        reg = tf.constant(0.0, dtype=DTYPE)
    total = residual_loss + tf.constant(cfg.lambda_reg, dtype=DTYPE) * reg
    metrics = {
        "loss_total": total,
        "loss_residual": residual_loss,
        "loss_reg_raw": reg,
        "residual_l2_raw": tf.sqrt(tf.reduce_mean(tf.square(R))),
        "residual_linf_raw": tf.reduce_max(tf.abs(R)),
    }
    return total, metrics


def train_adam(unet: MLP, gnet: Optional[MLP], bank: VPINNBankTF, cfg: Config, logger: logging.Logger) -> Dict[str, float]:
    if cfg.adam_steps <= 0:
        return {}
    decay_rate = (cfg.adam_lr1 / cfg.adam_lr0) ** (1.0 / max(1, cfg.adam_steps - 1))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        cfg.adam_lr0, decay_steps=1, decay_rate=decay_rate, staircase=False
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    bubble_cfg = make_bubble_cfg(cfg)

    @tf.function
    def step() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            loss, metrics = vpinn_loss(unet, gnet, bank, cfg, bubble_cfg)
        grads = tape.gradient(loss, unet.trainable_variables)
        opt.apply_gradients(zip(grads, unet.trainable_variables))
        ginf = tf.reduce_max(tf.concat([tf.reshape(tf.abs(g), [-1]) for g in grads if g is not None], axis=0))
        return loss, metrics["loss_residual"], ginf

    last = {}
    report_every = max(1, cfg.adam_steps // 10)
    for it in range(1, int(cfg.adam_steps) + 1):
        loss, residual, ginf = step()
        if it == 1 or it % report_every == 0 or it == cfg.adam_steps:
            logger.info("ADAM step=%05d total=%.6e residual=%.6e grad_inf=%.3e", it, loss.numpy(), residual.numpy(), ginf.numpy())
        last = {"adam_loss": float(loss.numpy()), "adam_residual_loss": float(residual.numpy()), "adam_grad_inf": float(ginf.numpy())}
    return last


def pack_weights(model: MLP) -> np.ndarray:
    return np.concatenate([v.numpy().reshape(-1) for v in model.trainable_variables]).astype(np.float64)


def unpack_weights(model: MLP, x: np.ndarray) -> None:
    pos = 0
    for v in model.trainable_variables:
        size = int(np.prod(v.shape))
        v.assign(x[pos:pos + size].reshape(v.shape))
        pos += size


def train_lbfgs(unet: MLP, gnet: Optional[MLP], bank: VPINNBankTF, cfg: Config, maxiter: int, logger: logging.Logger) -> Dict[str, float]:
    if maxiter <= 0:
        return {}
    bubble_cfg = make_bubble_cfg(cfg)
    state: Dict[str, float] = {"iter": 0.0, "loss": math.nan, "grad_inf": math.nan}

    def fun_and_jac(x: np.ndarray) -> Tuple[float, np.ndarray]:
        unpack_weights(unet, x)
        with tf.GradientTape() as tape:
            loss, metrics = vpinn_loss(unet, gnet, bank, cfg, bubble_cfg)
        grads = tape.gradient(loss, unet.trainable_variables)
        flat_parts = []
        for v, g in zip(unet.trainable_variables, grads):
            if g is None:
                flat_parts.append(np.zeros(int(np.prod(v.shape)), dtype=np.float64))
            else:
                flat_parts.append(g.numpy().reshape(-1).astype(np.float64))
        grad = np.concatenate(flat_parts)
        state["loss"] = float(loss.numpy())
        state["residual_loss"] = float(metrics["loss_residual"].numpy())
        state["grad_inf"] = float(np.max(np.abs(grad)))
        state["residual_linf_raw"] = float(metrics["residual_linf_raw"].numpy())
        return state["loss"], grad

    def callback(_xk: np.ndarray) -> None:
        state["iter"] += 1.0
        if cfg.lbfgs_callback_every and int(state["iter"]) % int(cfg.lbfgs_callback_every) == 0:
            logger.info(
                "L-BFGS iter=%04d total=%.6e residual=%.6e grad_inf=%.3e R_linf=%.3e",
                int(state["iter"]), state["loss"], state.get("residual_loss", math.nan),
                state["grad_inf"], state.get("residual_linf_raw", math.nan),
            )

    x0 = pack_weights(unet)
    result = sopt.minimize(
        fun_and_jac,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": int(maxiter),
            "ftol": float(cfg.lbfgs_ftol),
            "gtol": float(cfg.lbfgs_gtol),
            "maxcor": int(cfg.lbfgs_maxcor),
            "maxls": 50,
        },
    )
    unpack_weights(unet, result.x)
    logger.info("L-BFGS done: success=%s nit=%d nfev=%d fun=%.6e message=%s", result.success, result.nit, result.nfev, result.fun, result.message)
    return {
        "lbfgs_success": bool(result.success),
        "lbfgs_nit": int(result.nit),
        "lbfgs_nfev": int(result.nfev),
        "lbfgs_fun": float(result.fun),
        "lbfgs_final_grad_inf": float(state.get("grad_inf", math.nan)),
        "lbfgs_final_residual_loss": float(state.get("residual_loss", math.nan)),
        "lbfgs_final_residual_linf_raw": float(state.get("residual_linf_raw", math.nan)),
    }


# =============================================================================
# Error evaluation and plotting
# =============================================================================

def eval_lifted_np(unet: MLP, gnet: Optional[MLP], xy: np.ndarray, cfg: Config, chunk: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bubble_cfg = make_bubble_cfg(cfg)
    n = xy.shape[0]
    u = np.zeros(n, dtype=np.float64)
    gx = np.zeros(n, dtype=np.float64)
    gy = np.zeros(n, dtype=np.float64)
    for s in range(0, n, int(chunk)):
        e = min(s + int(chunk), n)
        xyt = tf.constant(xy[s:e], dtype=DTYPE)
        ut, gt = lifted_u_and_grad(unet, gnet, xyt, cfg, bubble_cfg)
        u[s:e] = ut.numpy().reshape(-1)
        g = gt.numpy()
        gx[s:e] = g[:, 0]
        gy[s:e] = g[:, 1]
    return u, gx, gy


def relative_h1_error(unet: MLP, gnet: Optional[MLP], cfg: Config) -> Tuple[float, float, float]:
    n_cells = int(cfg.h1_ncells)
    p = float(cfg.h1_grade_p)
    nq = int(cfg.h1_gauss_n)
    grid = (np.arange(n_cells + 1, dtype=np.float64) / float(n_cells)) ** p
    x0, x1 = grid[:-1], grid[1:]
    y0, y1 = grid[:-1], grid[1:]
    dx, dy = x1 - x0, y1 - y0
    t, w = gauss_legendre_01(nq)

    X0, Y0 = np.meshgrid(x0, y0, indexing="ij")
    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    X0f = X0.reshape(-1)
    Y0f = Y0.reshape(-1)
    DXf = DX.reshape(-1)
    DYf = DY.reshape(-1)

    Xn = X0f[:, None] + DXf[:, None] * t[None, :]
    Yn = Y0f[:, None] + DYf[:, None] * t[None, :]
    Wx = DXf[:, None] * w[None, :]
    Wy = DYf[:, None] * w[None, :]
    XX = Xn[:, :, None] * np.ones((1, 1, nq))
    YY = Yn[:, None, :] * np.ones((1, nq, 1))
    WW = Wx[:, :, None] * Wy[:, None, :]
    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    weights = WW.reshape(-1)

    u_pred, gx_pred, gy_pred = eval_lifted_np(unet, gnet, pts, cfg, cfg.eval_chunk)
    u_ex = u_exact_np(pts[:, 0], pts[:, 1])
    gx_ex, gy_ex = grad_u_exact_np(pts[:, 0], pts[:, 1])

    err_l2_sq = np.sum(weights * (u_pred - u_ex) ** 2)
    err_h1semi_sq = np.sum(weights * ((gx_pred - gx_ex) ** 2 + (gy_pred - gy_ex) ** 2))
    den_l2_sq = np.sum(weights * u_ex ** 2)
    den_h1semi_sq = np.sum(weights * (gx_ex ** 2 + gy_ex ** 2))
    rel_l2 = math.sqrt(max(err_l2_sq, 0.0) / max(den_l2_sq, 1.0e-300))
    rel_h1semi = math.sqrt(max(err_h1semi_sq, 0.0) / max(den_h1semi_sq, 1.0e-300))
    rel_h1 = math.sqrt(max(err_l2_sq + err_h1semi_sq, 0.0) / max(den_l2_sq + den_h1semi_sq, 1.0e-300))
    return float(rel_h1), float(rel_h1semi), float(rel_l2)


def estimate_slope(x: Sequence[float], y: Sequence[float], last_k: Optional[int] = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if last_k is not None and x.size > last_k:
        x = x[-last_k:]
        y = y[-last_k:]
    if x.size < 2:
        return math.nan
    coeff = np.polyfit(np.log(x), np.log(y), 1)
    return float(coeff[0])


def plot_reference_curve(history: List[Dict[str, float]], outdir: Path) -> None:
    tests = [row["n_test"] for row in history]
    h1 = [row["rel_h1"] for row in history]
    slope = estimate_slope(tests, h1)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=220)
    ax.plot(tests, h1, color="black", marker="*", markersize=6, linewidth=1.5, label=fr"Reference VPINN, slope={slope:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of test functions")
    ax.set_ylabel(r"Relative $H^1$ error")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "figure13_reference_vpinn.png", bbox_inches="tight")
    plt.close(fig)

    # Two-panel helper, because Figure 13 has panels for C_M=4 and C_M=9.
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=220, sharey=True)
    for ax, title in zip(axes, [r"Panel target: $C_M=4$", r"Panel target: $C_M=9$"]):
        ax.plot(tests, h1, color="black", marker="*", markersize=6, linewidth=1.5, label="Reference VPINN")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of test functions")
        ax.set_title(title)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.4)
        ax.legend(frameon=True)
    axes[0].set_ylabel(r"Relative $H^1$ error")
    fig.tight_layout()
    fig.savefig(outdir / "figure13_reference_vpinn_two_panels.png", bbox_inches="tight")
    plt.close(fig)


def write_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


# =============================================================================
# Driver
# =============================================================================

def setup_logger(outdir: Path) -> logging.Logger:
    logger = logging.getLogger("reference_vpinn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "reference_vpinn_figure13.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    text = (text or "").strip()
    if not text:
        return tuple()
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def mesh_ns_from_targets(targets: Sequence[int]) -> Tuple[int, ...]:
    ns = []
    for target in targets:
        # Number of P1 interior test functions on an n-subdivision square grid is (n-1)^2.
        n = int(round(math.sqrt(max(1, int(target))) + 1))
        n = max(3, n)
        if n not in ns:
            ns.append(n)
    return tuple(ns)


def run(cfg: Config) -> List[Dict[str, float]]:
    outdir = Path(cfg.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)
    set_seed(cfg.seed)

    if cfg.smoke:
        logger.warning("SMOKE MODE ENABLED: fast wiring test, not a scientific run. Humanity survives another misleading graph only if you do not publish this.")
        cfg.mesh_ns = (3, 4)
        cfg.q_tri = min(cfg.q_tri, 3)
        cfg.h1_ncells = min(cfg.h1_ncells, 16)
        cfg.h1_gauss_n = min(cfg.h1_gauss_n, 4)
        cfg.adam_steps = min(cfg.adam_steps, 25)
        cfg.lbfgs_first = min(cfg.lbfgs_first, 10)
        cfg.lbfgs_next = min(cfg.lbfgs_next, 10)
        cfg.gnet_steps = min(cfg.gnet_steps, 50)
        cfg.gnet_batch = min(cfg.gnet_batch, 512)
        cfg.gnet_grid_n_per_edge = min(cfg.gnet_grid_n_per_edge, 512)

    if cfg.target_tests:
        cfg.mesh_ns = mesh_ns_from_targets(cfg.target_tests)

    if cfg.lift == "debug_exact":
        logger.warning("debug_exact lift uses the exact solution in the interior. Do not use this for Figure 13 reproduction.")

    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.info("Reference VPINN Figure 13 run started")
    logger.info("Config: %s", json.dumps(asdict(cfg), sort_keys=True))

    unet = MLP(cfg.width, cfg.layers, cfg.activation, name="unet")
    _ = unet.call(tf.zeros((1, 2), dtype=DTYPE))

    gnet: Optional[MLP] = None
    if cfg.lift == "gnet":
        gnet = train_boundary_gnet(cfg, logger)
    elif cfg.lift == "coons":
        logger.info("Using exact Coons boundary extension g. This is cleaner than gnet, but less literal than the paper setup.")
    elif cfg.lift == "zero":
        logger.warning("Using zero boundary lift. This is only valid for homogeneous boundary data, not for the paper test unless you know what you are doing.")

    history: List[Dict[str, float]] = []
    previous_weights: Optional[np.ndarray] = None
    total_t0 = time.time()

    for level, n in enumerate(cfg.mesh_ns):
        level_t0 = time.time()
        if level > 0 and (not cfg.warm_start) and previous_weights is not None:
            unet = MLP(cfg.width, cfg.layers, cfg.activation, name=f"unet_level_{level}")
            _ = unet.call(tf.zeros((1, 2), dtype=DTYPE))

        logger.info("--- Mesh level %d/%d: n=%d subdivisions ---", level + 1, len(cfg.mesh_ns), n)
        pts, tris, interior, bmask = make_delaunay_unit_square(n, cfg.jitter, cfg.seed + 17 * level)
        bank_np = build_vpinn_bank(pts, tris, interior, bmask, cfg)
        bank_tf = bank_to_tf(bank_np)

        logger.info(
            "mesh stats: points=%d triangles=%d interior_tests=%d q_points=%d contribs=%d support_area[min,median,max]=[%.3e, %.3e, %.3e]",
            pts.shape[0], tris.shape[0], bank_np.n_test, bank_np.n_q, bank_np.n_contrib,
            float(np.min(bank_np.support_area)), float(np.median(bank_np.support_area)), float(np.max(bank_np.support_area)),
        )

        if cfg.save_meshes:
            np.savez_compressed(
                outdir / f"mesh_level_{level:02d}_Ntest_{bank_np.n_test}.npz",
                points=bank_np.points,
                triangles=bank_np.triangles,
                interior_nodes=bank_np.interior_nodes,
                boundary_mask=bank_np.boundary_mask,
                support_area=bank_np.support_area,
            )

        optim_info: Dict[str, float] = {}
        if level == 0:
            optim_info.update(train_adam(unet, gnet, bank_tf, cfg, logger))
            optim_info.update(train_lbfgs(unet, gnet, bank_tf, cfg, cfg.lbfgs_first, logger))
        else:
            optim_info.update(train_lbfgs(unet, gnet, bank_tf, cfg, cfg.lbfgs_next, logger))

        bubble_cfg = make_bubble_cfg(cfg)
        loss_value, metrics = vpinn_loss(unet, gnet, bank_tf, cfg, bubble_cfg)
        rel_h1, rel_h1semi, rel_l2 = relative_h1_error(unet, gnet, cfg)
        elapsed = time.time() - level_t0
        previous_weights = pack_weights(unet)

        row: Dict[str, float] = {
            "level": int(level),
            "mesh_n": int(n),
            "n_points": int(pts.shape[0]),
            "n_triangles": int(tris.shape[0]),
            "n_test": int(bank_np.n_test),
            "n_q_points": int(bank_np.n_q),
            "n_contrib": int(bank_np.n_contrib),
            "loss_total": float(loss_value.numpy()),
            "loss_residual": float(metrics["loss_residual"].numpy()),
            "loss_reg_raw": float(metrics["loss_reg_raw"].numpy()),
            "residual_l2_raw": float(metrics["residual_l2_raw"].numpy()),
            "residual_linf_raw": float(metrics["residual_linf_raw"].numpy()),
            "rel_h1": float(rel_h1),
            "rel_h1semi": float(rel_h1semi),
            "rel_l2": float(rel_l2),
            "elapsed_sec": float(elapsed),
        }
        for k, v in optim_info.items():
            if isinstance(v, (bool, np.bool_)):
                row[k] = bool(v)
            else:
                row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        history.append(row)
        running_slope = estimate_slope([r["n_test"] for r in history], [r["rel_h1"] for r in history])
        logger.info(
            "level done: Ntest=%d rel_H1=%.6e rel_H1semi=%.6e rel_L2=%.6e loss=%.6e elapsed=%.2fs slope_so_far=%.3f",
            row["n_test"], row["rel_h1"], row["rel_h1semi"], row["rel_l2"], row["loss_total"], elapsed, running_slope,
        )

        write_history_csv(history, outdir / "history_reference_vpinn.csv")
        with (outdir / "history_reference_vpinn.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        plot_reference_curve(history, outdir)

    total_elapsed = time.time() - total_t0
    final_slope = estimate_slope([r["n_test"] for r in history], [r["rel_h1"] for r in history])
    final_slope_tail = estimate_slope([r["n_test"] for r in history], [r["rel_h1"] for r in history], last_k=min(5, len(history)))
    summary = {
        "final_slope_all": final_slope,
        "final_slope_tail": final_slope_tail,
        "total_elapsed_sec": total_elapsed,
        "n_levels": len(history),
        "history_csv": "history_reference_vpinn.csv",
        "plot_single": "figure13_reference_vpinn.png",
        "plot_two_panels": "figure13_reference_vpinn_two_panels.png",
        "log": "reference_vpinn_figure13.log",
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Run complete: final_slope_all=%.6f final_slope_tail=%.6f total_elapsed=%.2fs", final_slope, final_slope_tail, total_elapsed)
    logger.info("Outputs written to %s", outdir)
    return history


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reference VPINN baseline for Figure 13 of the MF-VPINN paper.")
    p.add_argument("--outdir", type=str, default=Config.outdir)
    p.add_argument("--mesh-ns", type=str, default=",".join(map(str, Config.mesh_ns)), help="Comma-separated square-grid subdivisions n. Ntest=(n-1)^2.")
    p.add_argument("--target-tests", type=str, default="", help="Optional comma-separated target test counts; nearest square-grid Delaunay levels are used.")
    p.add_argument("--jitter", type=float, default=Config.jitter)
    p.add_argument("--seed", type=int, default=Config.seed)

    p.add_argument("--layers", type=int, default=Config.layers)
    p.add_argument("--width", type=int, default=Config.width)
    p.add_argument("--lambda-reg", type=float, default=Config.lambda_reg)
    p.add_argument("--gamma-mode", type=str, default=Config.gamma_mode, choices=["support", "mass", "none"])

    p.add_argument("--lift", type=str, default=Config.lift, choices=["gnet", "coons", "zero", "debug_exact"])
    p.add_argument("--bubble", type=str, default=Config.bubble, choices=["product", "softmin"])
    p.add_argument("--bubble-scale", type=float, default=Config.bubble_scale)
    p.add_argument("--softmin-tau", type=float, default=Config.softmin_tau)
    p.add_argument("--bubble-power", type=float, default=Config.bubble_power)

    p.add_argument("--q-tri", type=int, default=Config.q_tri)
    p.add_argument("--h1-ncells", type=int, default=Config.h1_ncells)
    p.add_argument("--h1-grade-p", type=float, default=Config.h1_grade_p)
    p.add_argument("--h1-gauss-n", type=int, default=Config.h1_gauss_n)
    p.add_argument("--eval-chunk", type=int, default=Config.eval_chunk)

    p.add_argument("--adam-steps", type=int, default=Config.adam_steps)
    p.add_argument("--adam-lr0", type=float, default=Config.adam_lr0)
    p.add_argument("--adam-lr1", type=float, default=Config.adam_lr1)
    p.add_argument("--lbfgs-first", type=int, default=Config.lbfgs_first)
    p.add_argument("--lbfgs-next", type=int, default=Config.lbfgs_next)
    p.add_argument("--lbfgs-ftol", type=float, default=Config.lbfgs_ftol)
    p.add_argument("--lbfgs-gtol", type=float, default=Config.lbfgs_gtol)
    p.add_argument("--lbfgs-maxcor", type=int, default=Config.lbfgs_maxcor)
    p.add_argument("--lbfgs-callback-every", type=int, default=Config.lbfgs_callback_every)

    p.add_argument("--gnet-steps", type=int, default=Config.gnet_steps)
    p.add_argument("--gnet-lr", type=float, default=Config.gnet_lr)
    p.add_argument("--gnet-batch", type=int, default=Config.gnet_batch)
    p.add_argument("--gnet-print-every", type=int, default=Config.gnet_print_every)
    p.add_argument("--gnet-corner-power", type=float, default=Config.gnet_corner_power)
    p.add_argument("--gnet-p-focus", type=float, default=Config.gnet_p_focus)
    p.add_argument("--gnet-grid-n-per-edge", type=int, default=Config.gnet_grid_n_per_edge)

    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--no-save-meshes", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Fast sanity check; not a scientific run.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = Config(
        outdir=args.outdir,
        mesh_ns=parse_int_tuple(args.mesh_ns),
        target_tests=parse_int_tuple(args.target_tests),
        jitter=args.jitter,
        seed=args.seed,
        layers=args.layers,
        width=args.width,
        lambda_reg=args.lambda_reg,
        gamma_mode=args.gamma_mode,
        lift=args.lift,
        bubble=args.bubble,
        bubble_scale=args.bubble_scale,
        softmin_tau=args.softmin_tau,
        bubble_power=args.bubble_power,
        q_tri=args.q_tri,
        h1_ncells=args.h1_ncells,
        h1_grade_p=args.h1_grade_p,
        h1_gauss_n=args.h1_gauss_n,
        eval_chunk=args.eval_chunk,
        adam_steps=args.adam_steps,
        adam_lr0=args.adam_lr0,
        adam_lr1=args.adam_lr1,
        lbfgs_first=args.lbfgs_first,
        lbfgs_next=args.lbfgs_next,
        lbfgs_ftol=args.lbfgs_ftol,
        lbfgs_gtol=args.lbfgs_gtol,
        lbfgs_maxcor=args.lbfgs_maxcor,
        lbfgs_callback_every=args.lbfgs_callback_every,
        gnet_steps=args.gnet_steps,
        gnet_lr=args.gnet_lr,
        gnet_batch=args.gnet_batch,
        gnet_print_every=args.gnet_print_every,
        gnet_corner_power=args.gnet_corner_power,
        gnet_p_focus=args.gnet_p_focus,
        gnet_grid_n_per_edge=args.gnet_grid_n_per_edge,
        save_meshes=not args.no_save_meshes,
        warm_start=not args.no_warm_start,
        smoke=args.smoke,
    )
    if not cfg.mesh_ns and not cfg.target_tests:
        raise SystemExit("Provide --mesh-ns or --target-tests.")
    run(cfg)


if __name__ == "__main__":
    main()
