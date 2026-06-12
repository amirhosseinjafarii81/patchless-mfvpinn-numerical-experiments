# -*- coding: utf-8 -*-
"""
fig17_problem35_strat12.py  (FIXED + Fig.18 NPZ export)

MF-VPINN Figure 17 reproduction attempt (Problem (35), Sec. 3.4) +
saves the final patch sets needed to reproduce Figure 18.

Main fixes vs your crashing version:
1) FIX TensorFlow poly_eval_tf: prebuild TF constants once (no .astype inside tf.function).
2) Save Figure 18 NPZ:
   - centers c_{P_i}
   - size ~ h_i^2 (for rectangles we use area = hx*hy)
   - color quantity r_{h,i}^2 = |R_i|^2 from weak residuals.
   - holes rectangles [x0,y0,x1,y1]
"""

import os
import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import tensorflow as tf
import scipy.optimize as sopt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# TF runtime config
# -------------------------
try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

tf.keras.backend.set_floatx("float32")
DTYPE = tf.float32


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
    H_MIN: float = 0.02

    ADAM_STEPS_FIRST: int = 2500
    LBFGS_MAXITER_FIRST: int = 500
    LBFGS_MAXITER_NEXT: int = 1500
    LBFGS_FTOL: float = 1e-14
    LBFGS_MAXCOR: int = 50

    ADAM_STEPS_AFTER: int = 500  # warm-adam steps after each refine (short)
    GAMMA_ALPHA: float = 0.5  # use (1/area)^alpha instead of 1/area (alpha in (0,1])
    GAMMA_MAX: float = 1e6  # cap for gamma

    # GPU-safety batching
    ADAM_PATCH_BATCH: int = 256
    ADAM_GRAD_ACCUM: int = 16
    EVAL_PATCH_BATCH: int = 2048
    LBFGS_PATCH_BATCH: int = 1024

    # Strategy parameters
    A_RATIO: float = 1.25
    LAM_LO: float = 9.0 / 10.0
    LAM_HI: float = 10.0 / 9.0
    MARK_FRAC: float = 0.70
    MARK_CAP: float = 0.30

    # Quadrature / projection
    Q_TRI: int = 3
    Q_EDGE: int = 3
    PROJ_DEG: int = 3
    EPS_PROJ: float = 1e-12

    # Estimator constant
    C_H: float = 1.0

    # Figure 17: 4 training points total: P0cut, P1cut, + 2 refinements
    N_REFINES_AFTER_P1: int = 2

    TARGET_NPATCH: int = 1000  # هدف تعداد پچ نهایی (برای هر شاخه/استراتژی)
    MAX_REFINES: int = 60  # حداکثر تعداد دور refine (گارد علیه لوپ بی‌نهایت)

    # Cutting controls (Sec. 3.4)
    CUT_MAX_ASPECT: float = 100.0
    CUT_MIN_AREA_RATIO: float = 1.0 / 100.0
    CUT_SPLIT_OVERLAP: float = 1.0  # 1.0 => partition; >1 => overlapping children

    # ADF / distance-function controls (Heliyon 2023)
    ADF_M: int = 2
    ADF_EPS: float = 1e-12

    # Relative H1 error quadrature on Ω2
    # Choose n=221 so hole boundaries (k/13 and k/17) align exactly with grid lines.
    H1_NCELLS: int = 221
    H1_GAUSS_N: int = 4
    H1_EVAL_CHUNK: int = 65536  # lower if you get OOM

    # Seed
    SEED: int = 2000


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------------
# Domain Ω2 (Sec. 3.4)
#   Ω2 = (0,1)^2 \ ⋃_{i=1}^4 H_i
#   hole centers: (9/26,9/34), (17/26,9/34), (9/26,25/34), (17/26,25/34)
#   "basis" and "height": 1/26 and 1/34 => interpret as HALF-sizes
# -------------------------
def holes_problem35():
    cx = [9.0/26.0, 17.0/26.0]
    cy = [9.0/34.0, 25.0/34.0]
    hx = 1.0/26.0  # half-width
    hy = 1.0/34.0  # half-height
    holes = []
    for x0 in cx:
        for y0 in cy:
            holes.append({"cx": float(x0), "cy": float(y0), "hx": float(hx), "hy": float(hy)})
    return holes

def rect_bounds_from_center(cx, cy, hx, hy):
    # NOTE: here hx,hy are FULL widths/heights
    xmin = cx - 0.5*hx
    xmax = cx + 0.5*hx
    ymin = cy - 0.5*hy
    ymax = cy + 0.5*hy
    return xmin, xmax, ymin, ymax

def hole_bounds(hole):
    # hole stores half-sizes hx,hy => bounds are cx±hx, cy±hy
    xmin = hole["cx"] - hole["hx"]
    xmax = hole["cx"] + hole["hx"]
    ymin = hole["cy"] - hole["hy"]
    ymax = hole["cy"] + hole["hy"]
    return xmin, xmax, ymin, ymax

def point_in_holes(x, y, holes):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    inside = np.zeros_like(x, dtype=bool)
    for H in holes:
        xmin, xmax, ymin, ymax = hole_bounds(H)
        inside |= (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    return inside

def rects_intersect(a, b):
    # strict intersection (positive area)
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return (max(ax0, bx0) < min(ax1, bx1)) and (max(ay0, by0) < min(ay1, by1))

def patch_bounds(p):
    cx, cy = p["c"]
    hx, hy = p["hx"], p["hy"]  # FULL widths/heights
    return rect_bounds_from_center(cx, cy, hx, hy)

def patch_intersecting_holes(p, holes):
    pb = patch_bounds(p)
    idxs = []
    for j, H in enumerate(holes):
        hb = hole_bounds(H)
        if rects_intersect(pb, hb):
            idxs.append(j)
    return idxs


# -------------------------
# Cutting procedure (Sec. 3.4)
# -------------------------
def split_patch_four(p, overlap=1.0):
    cx, cy = p["c"]
    hx, hy = float(p["hx"]), float(p["hy"])  # FULL widths/heights (as in rest of code)

    # compute child FULL sizes: child_full = 0.5 * parent_full * overlap
    child_hx = max(1e-15, 0.5 * hx * overlap)
    child_hy = max(1e-15, 0.5 * hy * overlap)

    # child half-sizes
    child_half_x = 0.5 * child_hx
    child_half_y = 0.5 * child_hy

    parent_half_x = 0.5 * hx
    parent_half_y = 0.5 * hy

    # place centers so that children cover the parent; clamp dx/dy >= 0
    dx = max(0.0, parent_half_x - child_half_x)
    dy = max(0.0, parent_half_y - child_half_y)

    children = [
        {"c": (cx - dx, cy - dy), "hx": child_hx, "hy": child_hy},
        {"c": (cx + dx, cy - dy), "hx": child_hx, "hy": child_hy},
        {"c": (cx - dx, cy + dy), "hx": child_hx, "hy": child_hy},
        {"c": (cx + dx, cy + dy), "hx": child_hx, "hy": child_hy},
    ]
    return children

def cut_patch_by_hole(p, H, cfg: Config):
    # returns list of rectangle patches that cover P \ H (axis-aligned)
    hx, hy = float(p["hx"]), float(p["hy"])
    px0, px1, py0, py1 = patch_bounds(p)

    hx0, hx1, hy0, hy1 = hole_bounds(H)

    # intersection
    x0i = max(px0, hx0); x1i = min(px1, hx1)
    y0i = max(py0, hy0); y1i = min(py1, hy1)

    rects = []
    # left
    if px0 < x0i:
        rects.append((px0, x0i, py0, py1))
    # right
    if x1i < px1:
        rects.append((x1i, px1, py0, py1))
    # bottom
    if py0 < y0i:
        rects.append((x0i, x1i, py0, y0i))
    # top
    if y1i < py1:
        rects.append((x0i, x1i, y1i, py1))

    orig_area = hx*hy
    out = []
    for (x0, x1, y0, y1) in rects:
        w = x1 - x0
        h = y1 - y0
        if w <= 0.0 or h <= 0.0:
            continue
        aspect = max(w, h) / max(min(w, h), 1e-300)
        area = w*h
        if aspect > float(cfg.CUT_MAX_ASPECT):
            continue
        if area < float(cfg.CUT_MIN_AREA_RATIO) * orig_area:
            continue
        out.append({"c": (0.5*(x0+x1), 0.5*(y0+y1)), "hx": w, "hy": h})
    return out

def cut_patches_to_omega2(patches, holes, cfg: Config, max_depth=20):
    out = []
    stack = [(p, 0) for p in patches]
    while stack:
        p, depth = stack.pop()

        x0, x1, y0, y1 = patch_bounds(p)
        if x1 <= 0.0 or x0 >= 1.0 or y1 <= 0.0 or y0 >= 1.0:
            continue

        idxs = patch_intersecting_holes(p, holes)
        if len(idxs) == 0:
            out.append(p)
            continue

        if len(idxs) == 1:
            new_ps = cut_patch_by_hole(p, holes[idxs[0]], cfg)
            out.extend(new_ps)
            continue

        if depth >= max_depth:
            continue

        children = split_patch_four(p, overlap=float(cfg.CUT_SPLIT_OVERLAP))
        for ch in children:
            stack.append((ch, depth+1))
    return out


# -------------------------
# Exact solution (36) and RHS f for problem (35)
# u(x,y) = 1/Cu * Π_{rx} (x-rx) * Π_{ry} (y-ry)
# normalized so u(2/13,2/17)=1.
# -------------------------
ROOTS_X = np.array([0.0, 1.0, 4.0/13.0, 5.0/13.0, 8.0/13.0, 9.0/13.0], dtype=np.float64)
ROOTS_Y = np.array([0.0, 1.0, 4.0/17.0, 5.0/17.0, 12.0/17.0, 13.0/17.0], dtype=np.float64)

P_COEFF = np.poly(ROOTS_X)  # degree 6
Q_COEFF = np.poly(ROOTS_Y)
P2_COEFF = np.polyder(P_COEFF, 2)
Q2_COEFF = np.polyder(Q_COEFF, 2)

_CU = float(np.polyval(P_COEFF, 2.0/13.0) * np.polyval(Q_COEFF, 2.0/17.0))

def poly_eval_np(coeff, x):
    return np.polyval(coeff, x)

def u_exact_np(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    return (poly_eval_np(P_COEFF, x) * poly_eval_np(Q_COEFF, y)) / _CU

def grad_u_exact_np(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    P1 = np.polyder(P_COEFF, 1)
    Q1 = np.polyder(Q_COEFF, 1)
    ux = (poly_eval_np(P1, x) * poly_eval_np(Q_COEFF, y)) / _CU
    uy = (poly_eval_np(P_COEFF, x) * poly_eval_np(Q1, y)) / _CU
    return ux, uy

def f_rhs_np(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    p  = poly_eval_np(P_COEFF, x)
    q  = poly_eval_np(Q_COEFF, y)
    p2 = poly_eval_np(P2_COEFF, x)
    q2 = poly_eval_np(Q2_COEFF, y)
    lap = (p2*q + p*q2) / _CU
    return -lap

# ---- FIX: build TF polynomial constants ONCE (no .astype inside tf.function) ----
P_COEFF_TF  = tf.constant(P_COEFF,  dtype=DTYPE)
Q_COEFF_TF  = tf.constant(Q_COEFF,  dtype=DTYPE)
P2_COEFF_TF = tf.constant(P2_COEFF, dtype=DTYPE)
Q2_COEFF_TF = tf.constant(Q2_COEFF, dtype=DTYPE)
CU_TF       = tf.constant(_CU, dtype=DTYPE)

# -------------------------------------------------
# Stable polynomial evaluation (no retracing issues)
# -------------------------------------------------
# جایگزین poly_eval_tf فعلی — استفاده از tf.while_loop برای Horner
@tf.function(experimental_relax_shapes=True)
def poly_eval_tf(coeff_tf, x):
    # coeff_tf: 1-D tensor (shape possibly unknown at trace time)
    # x: shape (N,1) or (M,1)
    # Horner: y = (((c0 * x + c1) * x + c2) * x + ...) but implemented iteratively
    coeff_tf = tf.convert_to_tensor(coeff_tf, dtype=DTYPE)
    n = tf.shape(coeff_tf)[0]

    y = tf.zeros_like(x, dtype=DTYPE)
    i = tf.constant(0, dtype=tf.int32)

    # shape invariant for y must allow None batch dimension
    shape_inv = tf.TensorShape([None, 1])

    def cond(i, y):
        return tf.less(i, n)

    def body(i, y):
        c = coeff_tf[i]              # scalar
        # broadcast c to shape of y automatically in arithmetic
        y = y * x + c
        return tf.add(i, 1), y

    _, y_out = tf.while_loop(cond, body, [i, y],
                             shape_invariants=[i.get_shape(), shape_inv])
    return y_out


# -------------------------
# Exact solution u(x,y)
# -------------------------
@tf.function(
    input_signature=[tf.TensorSpec([None, 2], dtype=DTYPE)]
)
def u_exact_tf(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    p = poly_eval_tf(P_COEFF_TF, x)
    q = poly_eval_tf(Q_COEFF_TF, y)
    return (p * q) / CU_TF


# -------------------------
# RHS f(x,y) = -Δu
# -------------------------
@tf.function(
    input_signature=[tf.TensorSpec([None, 2], dtype=DTYPE)]
)
def f_rhs_tf(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    p  = poly_eval_tf(P_COEFF_TF,  x)
    q  = poly_eval_tf(Q_COEFF_TF,  y)
    p2 = poly_eval_tf(P2_COEFF_TF, x)
    q2 = poly_eval_tf(Q2_COEFF_TF, y)

    lap = (p2 * q + p * q2) / CU_TF
    return -lap


# -------------------------
# MLP
# -------------------------
class MLP(tf.keras.Model):
    def __init__(self, width=50, layers=5):
        super().__init__()
        init = tf.keras.initializers.GlorotNormal()
        self.hidden = []
        for _ in range(max(1, layers-1)):
            self.hidden.append(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer=init))
        self.out = tf.keras.layers.Dense(1, activation=None, kernel_initializer=init)

    def call(self, x):
        z = x
        for lyr in self.hidden:
            z = lyr(z)
        return self.out(z)


# -------------------------
# Quadrature helpers
# -------------------------
def gauss_legendre_01(n):
    x, w = np.polynomial.legendre.leggauss(int(n))
    t = 0.5 * (x + 1.0)
    wt = 0.5 * w
    return t.astype(np.float64), wt.astype(np.float64)

def tri_ref_duffy(n):
    u, wu = gauss_legendre_01(n)
    v, wv = gauss_legendre_01(n)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    wuu, wvv = np.meshgrid(wu, wv, indexing="ij")
    x = uu.reshape(-1)
    y = ((1.0 - uu) * vv).reshape(-1)
    w = (wuu * wvv * (1.0 - uu)).reshape(-1)
    pts = np.stack([x, y], axis=1)
    return pts, w

def triangle_quadrature(A, B, C, TRI_REF_PTS, TRI_REF_W):
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    C = np.asarray(C, np.float64)
    BA = (B - A)[None, :]
    CA = (C - A)[None, :]
    pts = A[None, :] + TRI_REF_PTS[:, 0:1] * BA + TRI_REF_PTS[:, 1:2] * CA
    J = abs(np.linalg.det(np.stack([(B - A), (C - A)], axis=1)))
    w = TRI_REF_W * J
    return pts, w


# -------------------------
# Reference patch quadrature for hat test function (unit square)
# returns: PTS_HAT, W_HAT, GRAD_HAT, VAL_HAT
# -------------------------
def build_ref_patch_quadrature(Q_TRI):
    TRI_REF_PTS, TRI_REF_W = tri_ref_duffy(Q_TRI)

    center = np.array([0.5, 0.5], np.float64)
    corners = [
        np.array([0.0, 0.0], np.float64),
        np.array([1.0, 0.0], np.float64),
        np.array([1.0, 1.0], np.float64),
        np.array([0.0, 1.0], np.float64),
    ]
    tri_verts = [
        (corners[0], corners[1], center),
        (corners[1], corners[2], center),
        (corners[2], corners[3], center),
        (corners[3], corners[0], center),
    ]

    all_pts, all_w, all_grad, all_val = [], [], [], []
    for (A, B, C) in tri_verts:
        pts, w = triangle_quadrature(A, B, C, TRI_REF_PTS, TRI_REF_W)
        M = np.stack([B - A, C - A], axis=1)  # 2x2
        Minv = np.linalg.inv(M)

        # barycentric coordinate for vertex C is s, where [r,s]^T = Minv @ (p-A)
        grad_s = Minv[1, :]  # row
        s_val = (pts - A[None, :]) @ grad_s.reshape(2, 1)

        all_pts.append(pts)
        all_w.append(w)
        all_grad.append(np.tile(grad_s.reshape(1, 2), (pts.shape[0], 1)))
        all_val.append(s_val)

    pts_hat = np.vstack(all_pts)
    w_hat = np.concatenate(all_w)
    grad_hat = np.vstack(all_grad)
    val_hat = np.vstack(all_val)  # (Nq,1)

    return pts_hat, w_hat, grad_hat, val_hat

def patch_quadrature_hat_rect(c, hx, hy, PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL):
    c = np.asarray(c, np.float64)
    hx = float(hx); hy = float(hy)
    xy = c[None, :] + np.stack([hx*(PTS_HAT[:,0]-0.5), hy*(PTS_HAT[:,1]-0.5)], axis=1)
    w = (hx * hy) * W_HAT
    gradv = np.stack([HAT_GRAD[:,0]/hx, HAT_GRAD[:,1]/hy], axis=1)
    v = HAT_VAL.copy()
    return xy, w, gradv, v


# -------------------------
# Patch sets P0, P1
# -------------------------
def make_P0_rect():
    return [{"c": (0.5, 0.5), "hx": 1.0, "hy": 1.0}]

def make_P1_rect():
    P = make_P0_rect()
    for cx, cy in [(0.3,0.3), (0.3,0.7), (0.7,0.3), (0.7,0.7)]:
        P.append({"c": (float(cx), float(cy)), "hx": 0.6, "hy": 0.6})
    return P


# -------------------------
# Strategy #1 and #2 centers in reference patch
# -------------------------
def strategy2_hat_points(CM: int) -> np.ndarray:
    CM = int(CM)
    if CM == 4:
        return np.array([
            [1.0/4.0, 1.0/4.0],
            [3.0/4.0, 1.0/4.0],
            [1.0/4.0, 3.0/4.0],
            [3.0/4.0, 3.0/4.0],
        ], dtype=np.float64)
    if CM == 9:
        return np.array([
            [0.2, 0.2], [0.2, 0.5], [0.2, 0.8],
            [0.5, 0.2], [0.5, 0.5], [0.5, 0.8],
            [0.8, 0.2], [0.8, 0.5], [0.8, 0.8],
        ], dtype=np.float64)
    m = int(round(math.sqrt(CM)))
    if m*m != CM:
        raise ValueError("CM must be 4 or 9 (or perfect squares).")
    xs = np.array([(2*i + 1) / (2*m) for i in range(m)], dtype=np.float64)
    pts = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    return pts

def clip_center_unit_square(ex, ey, hx_new, hy_new):
    lo_x = 0.5 * hx_new
    hi_x = 1.0 - lo_x
    lo_y = 0.5 * hy_new
    hi_y = 1.0 - lo_y
    ex = min(max(ex, lo_x), hi_x)
    ey = min(max(ey, lo_y), hi_y)
    return ex, ey

def dorfler_mark_indices(values, cfg: Config):
    v = np.asarray(values, np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-v)
    total = float(np.sum(v))
    if total <= 0.0:
        return order[:1]
    cum = 0.0
    tau_tilde = 0
    for idx in order:
        cum += float(v[idx])
        tau_tilde += 1
        if cum >= float(cfg.MARK_FRAC) * total:
            break
    tau_cap = int(math.ceil(float(cfg.MARK_CAP) * n))
    tau_m = max(1, min(tau_cap, tau_tilde))
    return order[:tau_m]

def refine_strategy1_rect(Pm, eta_gamma, CM, cfg: Config, seed=0):
    rng = np.random.default_rng(seed)
    marked = dorfler_mark_indices(eta_gamma, cfg)
    new_patches = []
    scale = math.sqrt(float(cfg.A_RATIO) / float(CM))
    for i in marked:
        c = np.asarray(Pm[i]["c"], np.float64)
        hx = float(Pm[i]["hx"]); hy = float(Pm[i]["hy"])
        for _k in range(int(CM)):
            lam = rng.uniform(float(cfg.LAM_LO), float(cfg.LAM_HI))
            hx_new = max(1e-15, hx * scale * lam)
            hy_new = max(1e-15, hy * scale * lam)
            ex = rng.uniform(c[0] - 0.5*hx, c[0] + 0.5*hx)
            ey = rng.uniform(c[1] - 0.5*hy, c[1] + 0.5*hy)
            ex, ey = clip_center_unit_square(float(ex), float(ey), hx_new, hy_new)
            new_patches.append({"c": (float(ex), float(ey)), "hx": hx_new, "hy": hy_new})
    return Pm + new_patches, {"marked": int(len(marked)), "added": int(len(new_patches))}

def refine_strategy2_rect(Pm, eta_gamma, CM, cfg: Config, seed=0):
    rng = np.random.default_rng(seed)
    marked = dorfler_mark_indices(eta_gamma, cfg)
    new_patches = []
    scale = math.sqrt(float(cfg.A_RATIO) / float(CM))
    chat = strategy2_hat_points(int(CM))  # (CM,2)
    for i in marked:
        c = np.asarray(Pm[i]["c"], np.float64)
        hx = float(Pm[i]["hx"]); hy = float(Pm[i]["hy"])
        for k in range(int(CM)):
            lam = rng.uniform(float(cfg.LAM_LO), float(cfg.LAM_HI))
            hx_new = max(1e-15, hx * scale * lam)
            hy_new = max(1e-15, hy * scale * lam)
            ex = c[0] + hx * (chat[k,0] - 0.5)
            ey = c[1] + hy * (chat[k,1] - 0.5)
            ex, ey = clip_center_unit_square(float(ex), float(ey), hx_new, hy_new)
            new_patches.append({"c": (float(ex), float(ey)), "hx": hx_new, "hy": hy_new})
    return Pm + new_patches, {"marked": int(len(marked)), "added": int(len(new_patches))}


# -------------------------
# ADF distance function φ(x) to the boundary Γ2
# -------------------------
def boundary_segments_omega2(holes):
    # outer boundary 4 segments
    segs = [((0.0, 0.0), (1.0, 0.0)),
            ((1.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 1.0)),
            ((0.0, 1.0), (0.0, 0.0))]

    # hole boundaries: 4 segments per hole
    for H in holes:
        xmin, xmax, ymin, ymax = hole_bounds(H)
        segs.append(((xmin,ymin),(xmax,ymin)))
        segs.append(((xmax,ymin),(xmax,ymax)))
        segs.append(((xmax,ymax),(xmin,ymax)))
        segs.append(((xmin,ymax),(xmin,ymin)))
    return segs

# -----------------------------------------
# ADF distance to a single line segment
# -----------------------------------------
@tf.function(experimental_relax_shapes=True)
def adf_segment_tf(xy, A_tf, B_tf):
    # A_tf, B_tf: tf.Tensor shape (2,)
    Ax = A_tf[0]
    Ay = A_tf[1]
    Bx = B_tf[0]
    By = B_tf[1]

    x = xy[:, 0:1]
    y = xy[:, 1:2]

    dx = Bx - Ax
    dy = By - Ay
    L  = tf.sqrt(dx*dx + dy*dy)

    L_safe = tf.maximum(L, tf.constant(1e-300, dtype=DTYPE))

    # signed distance to infinite line
    d = ((x - Ax) * dy - (y - Ay) * dx) / L_safe

    # projection term for segment endpoints
    xc = 0.5 * (Ax + Bx)
    yc = 0.5 * (Ay + By)
    t  = ((0.5*L)**2 - ((x-xc)**2 + (y-yc)**2)) / L_safe

    tmp = tf.sqrt(t*t + d*d*d*d) - t
    phi = tf.sqrt(d*d + tmp*tmp)

    return phi


# -------------------------------------------------------
# Build full ADF φ(x) and ∇φ(x) from all segments
# -------------------------------------------------------
def build_phi_adf_tf(segments, cfg):

    m   = tf.constant(float(cfg.ADF_M),   dtype=DTYPE)
    eps = tf.constant(float(cfg.ADF_EPS), dtype=DTYPE)

    # IMPORTANT:
    # Convert all segment endpoints ONCE to tf.constant
    segments_tf = [
        (tf.constant(A, dtype=DTYPE),
         tf.constant(B, dtype=DTYPE))
        for (A, B) in segments
    ]

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], dtype=DTYPE)]
    )
    def phi_and_grad(xy):
        with tf.GradientTape() as tape:
            tape.watch(xy)

            inv_terms = []
            for (A_tf, B_tf) in segments_tf:
                phi_i = adf_segment_tf(xy, A_tf, B_tf)
                inv_terms.append(tf.pow(tf.maximum(phi_i, eps), -m))

            s   = tf.add_n(inv_terms)
            phi = tf.pow(s, -1.0/m)

        gphi = tape.gradient(phi, xy)
        return phi, gphi

    return phi_and_grad


# -------------------------
# Patch bank: build quadrature tensors + precompute φ, ∇φ, f
# -------------------------
def build_patch_bank(patches, cfg: Config, PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL, phi_and_grad):
    if len(patches) == 0:
        raise ValueError("patch list is empty")

    # جمع‌آوری نقاط کوآدراچر، وزن‌ها، گرادیان‌های hat و مقادیر hat برای هر patch
    xy_list, w_list, gv_list, v_list, areas = [], [], [], [], []
    for p in patches:
        xy, w, gradv, v = patch_quadrature_hat_rect(p["c"], p["hx"], p["hy"], PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL)
        xy_list.append(xy)                     # (Nq,2)
        w_list.append(w.reshape(-1,1))         # (Nq,1)
        gv_list.append(gradv)                  # (Nq,2)
        v_list.append(v.reshape(-1,1))         # (Nq,1)
        areas.append(float(p["hx"]) * float(p["hy"]))

    # تانسورهای همه‌پچ (Np x Nq x ...)
    xy_all = np.stack(xy_list, axis=0).astype(np.float64)      # (Np,Nq,2)
    w_all  = np.stack(w_list, axis=0).astype(np.float64)       # (Np,Nq,1)
    gv_all = np.stack(gv_list, axis=0).astype(np.float64)      # (Np,Nq,2)
    v_all  = np.stack(v_list, axis=0).astype(np.float64)       # (Np,Nq,1)

    # area هر پچ و gamma (مقیاس نرمال‌سازی) — اینجا مطابق نسخه‌ی تثبیت‌شده:
    area = np.asarray(areas, np.float64).reshape(-1,1)         # (Np,1)
    # === improved gamma (reduce dominance of tiny patches) ===
    alpha = float(cfg.GAMMA_ALPHA)
    gamma_raw = 1.0 / np.maximum(area, 1e-300)  # (Np,1)
    # soften with exponent alpha (alpha in (0,1] reduces dominance)
    gamma = np.power(gamma_raw, alpha).reshape(-1)
    # normalize so mean(gamma)=1 (keeps scale stable when Npatch changes)
    mean_g = float(np.maximum(np.mean(gamma), 1e-300))
    gamma = gamma / mean_g
    # clip to avoid numerics
    gamma = np.minimum(gamma, float(cfg.GAMMA_MAX)).reshape(-1, 1)

    # Precompute φ, ∇φ and f at all quadrature points (یکبار برای همه نقاط)
    xy_flat = xy_all.reshape(-1,2)
    with tf.device("/CPU:0"):
        xy_tf = tf.constant(xy_flat, dtype=DTYPE)
        phi_tf, gphi_tf = phi_and_grad(xy_tf)   # phi_tf: (Ntot,1), gphi_tf: (Ntot,2)
        f_tf = f_rhs_tf(xy_tf)

        # بازشکل‌دهی به ابعاد (Np,Nq,...)
        Np = xy_all.shape[0]
        Nq = xy_all.shape[1]
        phi = phi_tf.numpy().reshape(Np, Nq, 1)
        gphi = gphi_tf.numpy().reshape(Np, Nq, 2)
        fval = f_tf.numpy().reshape(Np, Nq, 1)

        # ساخت بانک (همه چیز به صورت tf.constant برای استفاده در tf.functionها)
        bank = {
            "xy_all": tf.constant(xy_all, dtype=DTYPE),
            "w_all": tf.constant(w_all, dtype=DTYPE),
            "gradv_all": tf.constant(gv_all, dtype=DTYPE),
            "v_all": tf.constant(v_all, dtype=DTYPE),
            "phi_all": tf.constant(phi, dtype=DTYPE),
            "gphi_all": tf.constant(gphi, dtype=DTYPE),
            "f_all": tf.constant(fval, dtype=DTYPE),
            "gamma": tf.constant(gamma, dtype=DTYPE),
            "area": tf.constant(area, dtype=DTYPE),
            "Npatch": int(Np),
            "Nq": int(Nq),
        }
    return bank

def minibatch_from_bank(bank, patch_ids):
    patch_ids = tf.convert_to_tensor(patch_ids, dtype=tf.int32)
    B = tf.shape(patch_ids)[0]
    Nq = tf.shape(bank["w_all"])[1]

    xy = tf.reshape(tf.gather(bank["xy_all"], patch_ids), [-1, 2])
    w  = tf.reshape(tf.gather(bank["w_all"],  patch_ids), [-1, 1])
    gv = tf.reshape(tf.gather(bank["gradv_all"], patch_ids), [-1, 2])
    v  = tf.reshape(tf.gather(bank["v_all"], patch_ids), [-1, 1])
    phi = tf.reshape(tf.gather(bank["phi_all"], patch_ids), [-1, 1])
    gphi = tf.reshape(tf.gather(bank["gphi_all"], patch_ids), [-1, 2])
    fval = tf.reshape(tf.gather(bank["f_all"], patch_ids), [-1, 1])

    pid = tf.repeat(tf.range(B, dtype=tf.int32), repeats=Nq)
    gamma = tf.gather(bank["gamma"], patch_ids)  # (B,1)
    area  = tf.gather(bank["area"], patch_ids)   # (B,1)

    return {"xy": xy, "w": w, "gradv": gv, "v": v, "phi": phi, "gphi": gphi, "f": fval,
            "pid": pid, "gamma": gamma, "area": area}


# -------------------------
# u = φ w, grad u = (∇φ) w + φ ∇w
# -------------------------
@tf.function
def w_and_grad_w(unet, xy):
    with tf.GradientTape() as tape:
        tape.watch(xy)
        w = unet(xy)
    gw = tape.gradient(w, xy)
    return w, gw

@tf.function
def loss_fn_hat(unet, XY, W, GRADV, V, PHI, GPHI, F, pid, gamma,
                pmask=None, reduction="mean"):
    w, gw = w_and_grad_w(unet, XY)
    gu = GPHI * w + PHI * gw

    integ = W * (tf.reduce_sum(gu * GRADV, axis=1, keepdims=True) - F * V)
    seg = tf.math.unsorted_segment_sum(integ, pid, tf.shape(gamma)[0])  # R_i

    r_abs = tf.abs(seg)
    term = gamma * tf.square(seg)

    if pmask is not None:
        pmask = tf.cast(pmask, DTYPE)
        term = term * pmask
        denom = tf.reduce_sum(pmask)
    else:
        denom = tf.cast(tf.shape(gamma)[0], DTYPE)

    if reduction == "sum":
        Rh2 = tf.reduce_sum(term)
    else:
        Rh2 = tf.reduce_sum(term) / tf.maximum(denom, tf.constant(1.0, dtype=DTYPE))

    return Rh2, r_abs


def _zeros_like(vars_):
    return [tf.zeros_like(v) for v in vars_]

def _safe_add_grads(accum, grads):
    out = []
    for a, g in zip(accum, grads):
        out.append(a if g is None else (a + g))
    return out


# -------------------------
# Adam first stage
# -------------------------
def adam_first(unet, bank, cfg: Config, seed=0):
    if cfg.ADAM_STEPS_FIRST <= 0:
        return

    vars_ = unet.trainable_variables
    reg_coeff = tf.constant(cfg.LAMBDA_REG, dtype=DTYPE)
    rng = np.random.default_rng(seed)

    decay_rate = (cfg.LR1 / cfg.LR0) ** (1.0 / float(max(1, cfg.ADAM_STEPS_FIRST - 1)))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.LR0,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=False,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

    patch_batch = int(cfg.ADAM_PATCH_BATCH)
    grad_accum_steps = int(cfg.ADAM_GRAD_ACCUM)
    nP = int(bank["Npatch"])

    for _it in range(int(cfg.ADAM_STEPS_FIRST)):
        while True:
            try:
                grads_accum = _zeros_like(vars_) if vars_ else []
                for _ in range(grad_accum_steps):
                    ids = rng.integers(0, nP, size=patch_batch, dtype=np.int32)
                    mb = minibatch_from_bank(bank, ids)

                    with tf.GradientTape() as tape:
                        Rh2, _ = loss_fn_hat(
                            unet,
                            mb["xy"], mb["w"], mb["gradv"], mb["v"],
                            mb["phi"], mb["gphi"], mb["f"],
                            mb["pid"], mb["gamma"],
                            reduction="mean",
                        )
                        reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) if vars_ else tf.constant(0.0, dtype=DTYPE)
                        L = Rh2 + reg_coeff * reg

                    g = tape.gradient(L, vars_)
                    grads_accum = _safe_add_grads(grads_accum, g)

                if vars_:
                    inv = tf.constant(1.0 / float(grad_accum_steps), dtype=DTYPE)
                    grads_accum = [g * inv for g in grads_accum]
                    opt.apply_gradients(zip(grads_accum, vars_))
                break
            except tf.errors.ResourceExhaustedError:
                if patch_batch <= 8:
                    raise
                patch_batch = max(8, patch_batch // 2)
                grad_accum_steps = max(1, min(grad_accum_steps, 8))
                print(f"[OOM backoff] Adam patch_batch={patch_batch}, grad_accum={grad_accum_steps}")


def eval_full_Rabs_and_Rh2(unet, bank, cfg: Config, patch_batch=1024):
    nP = int(bank["Npatch"])
    B = int(patch_batch)
    Rabs = np.zeros((nP, 1), dtype=np.float64)
    sum_term = 0.0

    ones_full = np.ones((B, 1), dtype=np.float64)

    for start in range(0, nP, B):
        end = min(start + B, nP)
        ids = np.arange(start, end, dtype=np.int32)
        n = int(end - start)

        if n < B:
            pad_ids = np.full((B - n,), ids[-1], dtype=np.int32)
            ids_pad = np.concatenate([ids, pad_ids], axis=0)
            mask = np.vstack([np.ones((n, 1), np.float64), np.zeros((B - n, 1), np.float64)])
        else:
            ids_pad = ids
            mask = ones_full

        mb = minibatch_from_bank(bank, ids_pad)
        pmask_tf = tf.constant(mask, dtype=DTYPE)

        Rh2_sum_mb, r_mb = loss_fn_hat(
            unet,
            mb["xy"], mb["w"], mb["gradv"], mb["v"],
            mb["phi"], mb["gphi"], mb["f"],
            mb["pid"], mb["gamma"],
            pmask=pmask_tf,
            reduction="sum",
        )
        r_np = r_mb.numpy()
        Rabs[start:end, :] = r_np[:n, :]
        sum_term += float(Rh2_sum_mb.numpy())

    Rh2_mean = sum_term / float(nP)
    return Rh2_mean, Rabs.reshape(-1)


# -------------------------
# L-BFGS optimize (deterministic full-batch via patch batching)
# -------------------------
def lbfgs_optimize(unet, bank, cfg: Config, maxiter, patch_batch=1024, callback_every=10):
    maxiter = int(maxiter)
    if maxiter <= 0:
        return None

    vars_ = unet.trainable_variables
    shapes = [v.shape for v in vars_]
    sizes = [int(np.prod(s)) for s in shapes]

    nP = int(bank["Npatch"])
    invN = tf.constant(1.0 / float(nP), dtype=DTYPE)
    reg_coeff = tf.constant(cfg.LAMBDA_REG, dtype=DTYPE)

    def pack():
        return np.concatenate([v.numpy().reshape(-1) for v in vars_]).astype(np.float64)

    def unpack(x):
        i = 0
        for v, s, shp in zip(vars_, sizes, shapes):
            v.assign(x[i:i+s].reshape(shp))
            i += s

    state = {"it": 0, "last_f": None, "last_ginf": None}

    def fun_and_jac(x):
        unpack(x)

        Rh2_sum_total = tf.constant(0.0, dtype=DTYPE)
        grads_sum = _zeros_like(vars_)

        B = int(patch_batch)
        ones_full = np.ones((B, 1), dtype=np.float64)

        for start in range(0, nP, B):
            end = min(start + B, nP)
            ids = np.arange(start, end, dtype=np.int32)
            n = int(end - start)

            if n < B:
                pad_ids = np.full((B - n,), ids[-1], dtype=np.int32)
                ids_pad = np.concatenate([ids, pad_ids], axis=0)
                mask = np.vstack([np.ones((n, 1), np.float64), np.zeros((B - n, 1), np.float64)])
            else:
                ids_pad = ids
                mask = ones_full

            mb = minibatch_from_bank(bank, ids_pad)
            pmask_tf = tf.constant(mask, dtype=DTYPE)

            with tf.GradientTape() as tape:
                Rh2_sum_mb, _ = loss_fn_hat(
                    unet,
                    mb["xy"], mb["w"], mb["gradv"], mb["v"],
                    mb["phi"], mb["gphi"], mb["f"],
                    mb["pid"], mb["gamma"],
                    pmask=pmask_tf,
                    reduction="sum",
                )
                reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) if vars_ else tf.constant(0.0, dtype=DTYPE)
                loss_mb = Rh2_sum_mb * invN + reg_coeff * reg

            g_mb = tape.gradient(loss_mb, vars_)
            grads_sum = _safe_add_grads(grads_sum, g_mb)
            Rh2_sum_total += Rh2_sum_mb

        Rh2_mean = Rh2_sum_total * invN
        reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) if vars_ else tf.constant(0.0, dtype=DTYPE)
        loss = Rh2_mean + reg_coeff * reg

        flat = []
        for v, g in zip(vars_, grads_sum):
            if g is None:
                flat.append(np.zeros(int(np.prod(v.shape)), dtype=np.float64))
            else:
                flat.append(g.numpy().reshape(-1).astype(np.float64))
        grad_vec = np.concatenate(flat, axis=0)

        loss_val = float(loss.numpy())
        ginf = float(np.max(np.abs(grad_vec))) if grad_vec.size else 0.0
        state["last_f"] = loss_val
        state["last_ginf"] = ginf
        return loss_val, grad_vec

    def callback(xk):
        state["it"] += 1
        if callback_every and (state["it"] % int(callback_every) == 0):
            print(f"    [L-BFGS iter {state['it']:04d}] f={state['last_f']:.3e}  ||g||_inf={state['last_ginf']:.3e}")

    x0 = pack()
    res = sopt.minimize(
        fun_and_jac,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": int(maxiter), "ftol": float(cfg.LBFGS_FTOL), "gtol": 1e-14, "maxcor": int(cfg.LBFGS_MAXCOR)},
    )
    unpack(res.x)
    print(f"[L-BFGS] {res.message}  nit={res.nit} nfev={res.nfev} fun={res.fun:.3e}")
    return res


# -------------------------
# Indicator (Eq. 25-26) (rect version)
# -------------------------
def monomials_deg3(x, y):
    ones = np.ones_like(x)
    return np.stack([ones, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y], axis=-1)

def dmonomials_dx_deg3(x, y):
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    return np.stack([zeros, ones, zeros, 2*x, y, zeros, 3*x*x, 2*x*y, y*y, zeros], axis=-1)

def dmonomials_dy_deg3(x, y):
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    return np.stack([zeros, zeros, ones, zeros, x, 2*y, zeros, x*x, 2*x*y, 3*y*y], axis=-1)

def solve_10x10_batch(A, b):
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    try:
        return np.linalg.solve(A, b)
    except Exception:
        B = A.shape[0]
        x = np.zeros((B, 10, 1), dtype=np.float64)
        for i in range(B):
            Ai, bi = A[i], b[i]
            try:
                x[i] = np.linalg.solve(Ai, bi)
            except np.linalg.LinAlgError:
                x[i] = np.linalg.lstsq(Ai, bi, rcond=None)[0]
        return x

def patch_vertices_rect(centers, hx, hy):
    cx = centers[:,0]; cy = centers[:,1]
    hx2 = 0.5*hx; hy2 = 0.5*hy
    v0 = np.stack([cx - hx2, cy - hy2], axis=1)
    v1 = np.stack([cx + hx2, cy - hy2], axis=1)
    v2 = np.stack([cx + hx2, cy + hy2], axis=1)
    v3 = np.stack([cx - hx2, cy + hy2], axis=1)
    return np.stack([v0,v1,v2,v3], axis=1).astype(np.float64)

def tri_points_weights_batch(A, B, C, TRI_REF_PTS, TRI_REF_W):
    A = A[:, None, :]
    BA = (B - A[:, 0, :])[:, None, :]
    CA = (C - A[:, 0, :])[:, None, :]
    rx = TRI_REF_PTS[None, :, 0:1]
    ry = TRI_REF_PTS[None, :, 1:2]
    pts = A + rx * BA + ry * CA

    BA2 = (B - A[:, 0, :])
    CA2 = (C - A[:, 0, :])
    det = BA2[:, 0] * CA2[:, 1] - BA2[:, 1] * CA2[:, 0]
    J = np.abs(det)
    w = TRI_REF_W[None, :] * J[:, None]
    return pts.astype(np.float64), w.astype(np.float64)

def eval_grad_u_np_batch(unet, pts_flat, phi_and_grad, cfg: Config, chunk=32768):
    pts_flat = np.asarray(pts_flat, np.float64)
    N = int(pts_flat.shape[0])
    out = np.zeros((N,2), np.float64)
    B = int(chunk)
    buf = np.empty((B,2), dtype=np.float64)

    for s in range(0, N, B):
        e = min(s+B, N)
        n = int(e-s)
        buf[:n,:] = pts_flat[s:e,:]
        if n < B:
            buf[n:,:] = buf[n-1:n,:]

        xy_tf = tf.constant(buf, dtype=DTYPE)
        phi_tf, gphi_tf = phi_and_grad(xy_tf)
        w_tf, gw_tf = w_and_grad_w(unet, xy_tf)
        gu_tf = gphi_tf*w_tf + phi_tf*gw_tf
        out[s:e,:] = gu_tf.numpy()[:n,:]
    return out

def indicator_eta_gamma_eq25_rect(unet, patches, Rabs_vec, cfg: Config,
                                 TRI_REF_PTS, TRI_REF_W, EDGE_T, EDGE_W,
                                 phi_and_grad):
    nP = int(len(patches))
    if nP == 0:
        return np.zeros((0,), dtype=np.float64)

    centers = np.array([p["c"] for p in patches], dtype=np.float64)
    hx = np.array([p["hx"] for p in patches], dtype=np.float64).reshape(-1)
    hy = np.array([p["hy"] for p in patches], dtype=np.float64).reshape(-1)
    r_hi = np.asarray(Rabs_vec, dtype=np.float64).reshape(-1)

    V = patch_vertices_rect(centers, hx, hy)
    c = centers

    pts0, w0 = tri_points_weights_batch(c, V[:, 0, :], V[:, 1, :], TRI_REF_PTS, TRI_REF_W)
    pts1, w1 = tri_points_weights_batch(c, V[:, 1, :], V[:, 2, :], TRI_REF_PTS, TRI_REF_W)
    pts2, w2 = tri_points_weights_batch(c, V[:, 2, :], V[:, 3, :], TRI_REF_PTS, TRI_REF_W)
    pts3, w3 = tri_points_weights_batch(c, V[:, 3, :], V[:, 0, :], TRI_REF_PTS, TRI_REF_W)

    PTS = np.stack([pts0,pts1,pts2,pts3], axis=1)  # (nP,4,NT,2)
    WTS = np.stack([w0,w1,w2,w3], axis=1)          # (nP,4,NT)
    NT = int(PTS.shape[2])

    grad_flat = eval_grad_u_np_batch(unet, PTS.reshape(-1,2), phi_and_grad, cfg, chunk=32768)
    GR = grad_flat.reshape(nP,4,NT,2)

    ux = GR[...,0]; uy = GR[...,1]
    x = PTS[...,0]; y = PTS[...,1]
    w = WTS

    M   = monomials_deg3(x, y)
    dMx = dmonomials_dx_deg3(x, y)
    dMy = dmonomials_dy_deg3(x, y)

    f_vals = f_rhs_np(x, y)

    G   = np.einsum("ntki,ntk,ntkj->ntij", M, w, M)
    bx  = np.einsum("ntki,ntk,ntk->nti",  M, w, ux)
    by  = np.einsum("ntki,ntk,ntk->nti",  M, w, uy)
    bf  = np.einsum("ntki,ntk,ntk->nti",  M, w, f_vals)

    G_reg = G + float(cfg.EPS_PROJ) * np.eye(10, dtype=np.float64)[None,None,:,:]
    A_solve = G_reg.reshape(-1,10,10)

    cx = solve_10x10_batch(A_solve, bx.reshape(-1,10,1)).reshape(nP,4,10)
    cy = solve_10x10_batch(A_solve, by.reshape(-1,10,1)).reshape(nP,4,10)
    cf = solve_10x10_batch(A_solve, bf.reshape(-1,10,1)).reshape(nP,4,10)

    div_proj = np.einsum("ntki,nti->ntk", dMx, cx) + np.einsum("ntki,nti->ntk", dMy, cy)
    f_proj   = np.einsum("ntki,nti->ntk", M, cf)
    bulk = div_proj + f_proj
    bulk_sq_int = np.sum(w * (bulk*bulk), axis=2)
    bulk_norm = np.sqrt(np.maximum(bulk_sq_int, 0.0))

    osc = f_vals - f_proj
    osc_sq_int = np.sum(w * (osc*osc), axis=2)
    osc_norm = np.sqrt(np.maximum(osc_sq_int, 0.0))

    def tri_diam_batch(C0, A0, B0):
        dAB = np.linalg.norm(A0 - B0, axis=1)
        dAC = np.linalg.norm(A0 - C0, axis=1)
        dBC = np.linalg.norm(B0 - C0, axis=1)
        return np.maximum.reduce([dAB,dAC,dBC])

    hT = np.stack([
        tri_diam_batch(c, V[:,0,:], V[:,1,:]),
        tri_diam_batch(c, V[:,1,:], V[:,2,:]),
        tri_diam_batch(c, V[:,2,:], V[:,3,:]),
        tri_diam_batch(c, V[:,3,:], V[:,0,:]),
    ], axis=1)

    dvec = V - c[:,None,:]
    L = np.linalg.norm(dvec, axis=2)
    L_safe = np.maximum(L, 1e-15)
    tvec = dvec / L_safe[:,:,None]
    nvec = np.stack([tvec[:,:,1], -tvec[:,:,0]], axis=2)

    pts_e = c[:,None,None,:] + EDGE_T[None,None,:,None] * dvec[:,:,None,:]
    xe = pts_e[...,0]; ye = pts_e[...,1]
    M_edge = monomials_deg3(xe, ye)

    def flux_for(tri_order):
        cx_sel = cx[:, tri_order, :]
        cy_sel = cy[:, tri_order, :]
        fx = np.einsum("neqk,nek->neq", M_edge, cx_sel)
        fy = np.einsum("neqk,nek->neq", M_edge, cy_sel)
        return fx, fy

    fxa, fya = flux_for([3,0,1,2])
    fxb, fyb = flux_for([0,1,2,3])

    flux_a = fxa*nvec[:,:,0,None] + fya*nvec[:,:,1,None]
    flux_b = fxb*nvec[:,:,0,None] + fyb*nvec[:,:,1,None]
    jmp = flux_a - flux_b

    j_sq_int = np.sum((jmp*jmp)*EDGE_W[None,None,:], axis=2) * L_safe
    j_norm = np.sqrt(np.maximum(j_sq_int, 0.0))

    hP = np.sqrt(hx*hx + hy*hy)

    eta_res = np.sum(hT * bulk_norm, axis=1) + np.sqrt(hP) * np.sum(j_norm, axis=1)
    eta_osc = np.sum(hT * osc_norm, axis=1)

    eta_sq = eta_res**2 + eta_osc**2 + (float(cfg.C_H)*r_hi)**2
    eta = np.sqrt(np.maximum(eta_sq, 0.0))
    return eta


# -------------------------
# Relative H1 error on Ω2
# -------------------------
def rel_H1_error_omega2(unet, holes, cfg: Config, phi_and_grad):
    n = int(cfg.H1_NCELLS)
    nq = int(cfg.H1_GAUSS_N)

    x_edges = np.linspace(0.0, 1.0, n+1, dtype=np.float64)
    y_edges = np.linspace(0.0, 1.0, n+1, dtype=np.float64)

    t, w = gauss_legendre_01(nq)

    pts_list = []
    wts_list = []

    for i in range(n):
        x0, x1 = x_edges[i], x_edges[i+1]
        dx = x1 - x0
        xq = x0 + dx * t
        wx = dx * w
        for j in range(n):
            y0, y1 = y_edges[j], y_edges[j+1]
            dy = y1 - y0
            xc = 0.5*(x0+x1); yc = 0.5*(y0+y1)
            if point_in_holes(np.array([xc]), np.array([yc]), holes)[0]:
                continue
            yq = y0 + dy * t
            wy = dy * w
            XX, YY = np.meshgrid(xq, yq, indexing="ij")
            WW = np.outer(wx, wy)
            pts_list.append(np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1))
            wts_list.append(WW.reshape(-1))

    pts = np.vstack(pts_list).astype(np.float64)
    wts = np.concatenate(wts_list).astype(np.float64)

    N = int(pts.shape[0])
    u_pred = np.zeros((N,), np.float64)
    gx_pred = np.zeros((N,), np.float64)
    gy_pred = np.zeros((N,), np.float64)

    B = int(cfg.H1_EVAL_CHUNK)
    buf = np.empty((B,2), dtype=np.float64)

    for s in range(0, N, B):
        e = min(s+B, N)
        nn = e-s
        buf[:nn,:] = pts[s:e,:]
        if nn < B:
            buf[nn:,:] = buf[nn-1:nn,:]

        xy_tf = tf.constant(buf, dtype=DTYPE)
        phi_tf, gphi_tf = phi_and_grad(xy_tf)
        w_tf, gw_tf = w_and_grad_w(unet, xy_tf)
        u_tf = (phi_tf * w_tf)
        gu_tf = gphi_tf*w_tf + phi_tf*gw_tf

        u_np = u_tf.numpy().reshape(-1)
        gu_np = gu_tf.numpy()

        u_pred[s:e] = u_np[:nn]
        gx_pred[s:e] = gu_np[:nn,0]
        gy_pred[s:e] = gu_np[:nn,1]

    u_ex = u_exact_np(pts[:,0], pts[:,1])
    gx_ex, gy_ex = grad_u_exact_np(pts[:,0], pts[:,1])

    e0 = u_pred - u_ex
    ex = gx_pred - gx_ex
    ey = gy_pred - gy_ex

    num = np.sum(wts * (e0*e0 + ex*ex + ey*ey))
    den = np.sum(wts * (u_ex*u_ex + gx_ex*gx_ex + gy_ex*gy_ex))
    return float(np.sqrt(max(num, 0.0) / max(den, 1e-300)))


# -------------------------
# One training iteration
# -------------------------
def train_unet_iteration(unet, patches, cfg: Config,
                         PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL,
                         phi_and_grad,
                         is_first=False, seed=0, lbfgs_max_patches=20000):

    bank = build_patch_bank(
        patches, cfg,
        PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL,
        phi_and_grad
    )

    # =========================
    # First iteration
    # =========================
    if is_first:
        adam_first(unet, bank, cfg, seed=seed)

        if bank["Npatch"] <= int(lbfgs_max_patches):
            lbfgs_optimize(
                unet, bank, cfg,
                maxiter=cfg.LBFGS_MAXITER_FIRST,
                patch_batch=cfg.LBFGS_PATCH_BATCH
            )

    # =========================
    # Refinement iterations
    # =========================
    else:
        # ---- warm Adam AFTER refine ----
        if getattr(cfg, "ADAM_STEPS_AFTER", 0) > 0:
            steps_after = int(cfg.ADAM_STEPS_AFTER)
            print(f"[warm-adam] {steps_after} steps before L-BFGS")

            vars_ = unet.trainable_variables
            reg_coeff = tf.constant(cfg.LAMBDA_REG, dtype=DTYPE)
            rng = np.random.default_rng(seed + 12345)

            patch_batch = int(cfg.ADAM_PATCH_BATCH)
            grad_accum_steps = int(cfg.ADAM_GRAD_ACCUM)
            opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.LR0))

            nP = int(bank["Npatch"])

            for _ in range(steps_after):
                try:
                    grads_accum = _zeros_like(vars_)
                    for _ in range(grad_accum_steps):
                        ids = rng.integers(0, nP, size=patch_batch, dtype=np.int32)
                        mb = minibatch_from_bank(bank, ids)

                        with tf.GradientTape() as tape:
                            Rh2, _ = loss_fn_hat(
                                unet,
                                mb["xy"], mb["w"], mb["gradv"], mb["v"],
                                mb["phi"], mb["gphi"], mb["f"],
                                mb["pid"], mb["gamma"],
                                reduction="mean",
                            )
                            reg = tf.add_n(
                                [tf.reduce_sum(tf.square(v)) for v in vars_]
                            )
                            L = Rh2 + reg_coeff * reg

                        g = tape.gradient(L, vars_)
                        grads_accum = _safe_add_grads(grads_accum, g)

                    grads_accum = [g / float(grad_accum_steps) for g in grads_accum]
                    opt.apply_gradients(zip(grads_accum, vars_))

                except tf.errors.ResourceExhaustedError:
                    if patch_batch <= 8:
                        raise
                    patch_batch //= 2
                    grad_accum_steps = min(8, grad_accum_steps * 2)
                    print(f"[warm-adam OOM] patch_batch={patch_batch}, accum={grad_accum_steps}")

        # ---- L-BFGS ----
        if bank["Npatch"] <= int(lbfgs_max_patches):
            lbfgs_optimize(
                unet, bank, cfg,
                maxiter=cfg.LBFGS_MAXITER_NEXT,
                patch_batch=cfg.LBFGS_PATCH_BATCH
            )

    # =========================
    # Evaluation
    # =========================
    Rh2_mean, Rabs = eval_full_Rabs_and_Rh2(
        unet, bank, cfg,
        patch_batch=cfg.EVAL_PATCH_BATCH
    )

    return {
        "Rabs": Rabs,
        "Rh2": float(Rh2_mean),
        "bank": bank
    }


# -------------------------
# Figure 17 run + Figure 18 NPZ saving
# -------------------------
def run_fig17(cfg: Config, outdir: str, lbfgs_max_patches=20000):
    holes = holes_problem35()
    segments = boundary_segments_omega2(holes)
    phi_and_grad = build_phi_adf_tf(segments, cfg)

    EDGE_T, EDGE_W = gauss_legendre_01(cfg.Q_EDGE)
    TRI_REF_PTS, TRI_REF_W = tri_ref_duffy(cfg.Q_TRI)
    PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL = build_ref_patch_quadrature(cfg.Q_TRI)

    # --- base run through P0 and P1 (shared) ---
    set_seed(cfg.SEED)
    unet_base = MLP(width=cfg.WIDTH, layers=cfg.L_LAYERS)
    unet_base.call(tf.zeros((1,2), dtype=DTYPE))

    P0 = cut_patches_to_omega2(make_P0_rect(), holes, cfg)
    print(f"[Base] P0 after cut: N={len(P0)}")
    _ = train_unet_iteration(unet_base, P0, cfg, PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL,
                             phi_and_grad, is_first=True, seed=cfg.SEED,
                             lbfgs_max_patches=lbfgs_max_patches)
    h1_0 = rel_H1_error_omega2(unet_base, holes, cfg, phi_and_grad)
    x0 = len(P0)

    P1 = cut_patches_to_omega2(make_P1_rect(), holes, cfg)
    print(f"[Base] P1 after cut: N={len(P1)}")
    res1 = train_unet_iteration(unet_base, P1, cfg, PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL,
                                phi_and_grad, is_first=False, seed=cfg.SEED+1,
                                lbfgs_max_patches=lbfgs_max_patches)
    h1_1 = rel_H1_error_omega2(unet_base, holes, cfg, phi_and_grad)
    x1 = len(P1)

    base_weights = unet_base.get_weights()
    Rabs_prev_base = res1["Rabs"]

    def branch_run(strategy: int, CM: int, tag: str):
        unet = MLP(width=cfg.WIDTH, layers=cfg.L_LAYERS)
        unet.call(tf.zeros((1,2), dtype=DTYPE))
        unet.set_weights(base_weights)

        patches = list(P1)
        Rabs_prev = Rabs_prev_base

        xs = [x0, x1]
        ys = [h1_0, h1_1]

        # --- REPLACE fixed-for with while loop until TARGET_NPATCH or MAX_REFINES ---
        # --- FIX: do exactly N_REFINES_AFTER_P1 refinements (as in paper) ---
        for k in range(int(cfg.N_REFINES_AFTER_P1)):
            eta = indicator_eta_gamma_eq25_rect(
                unet, patches, Rabs_prev, cfg,
                TRI_REF_PTS, TRI_REF_W, EDGE_T, EDGE_W,
                phi_and_grad
            )
            areas = np.array([p["hx"] * p["hy"] for p in patches], dtype=np.float64)
            gamma = 1.0 / np.maximum(areas, 1e-300)
            eta_gamma = gamma * eta

            # diagnostic: show top few patches by eta_gamma (index, center, area, eta_gamma)
            idx_order = np.argsort(-eta_gamma)[:6]
            top_info = [(int(i), patches[int(i)]["c"], patches[int(i)]["hx"]*patches[int(i)]["hy"], float(eta_gamma[int(i)])) for i in idx_order]
            print(f"[{tag}] top marked candidates (idx, center, area, eta_gamma): {top_info}")


            if strategy == 1:
                patches_new, info = refine_strategy1_rect(patches, eta_gamma, CM, cfg, seed=cfg.SEED + 100 + k)
            else:
                patches_new, info = refine_strategy2_rect(patches, eta_gamma, CM, cfg, seed=cfg.SEED + 200 + k)

            patches_new = cut_patches_to_omega2(patches_new, holes, cfg)
            patches = patches_new

            print(f"[{tag}] refine {k + 1}: marked={info['marked']} added={info['added']}  after cut N={len(patches)}")

            res = train_unet_iteration(unet, patches, cfg, PTS_HAT, W_HAT, HAT_GRAD, HAT_VAL,
                                       phi_and_grad, is_first=False, seed=cfg.SEED + 10 + k,
                                       lbfgs_max_patches=lbfgs_max_patches)
            Rabs_prev = res["Rabs"]

            h1 = rel_H1_error_omega2(unet, holes, cfg, phi_and_grad)
            xs.append(len(patches))
            ys.append(h1)
            print(f"[{tag}] step {k + 2}: N={len(patches)}  H1={h1:.3e}  Rh2={res['Rh2']:.3e}")

        # ---- Figure 18 snapshot (FINAL set) ----
        centers = np.array([p["c"] for p in patches], dtype=np.float64)                 # (N,2)
        h2 = np.array([p["hx"]*p["hy"] for p in patches], dtype=np.float64)             # size ~ h_i^2
        r2 = (np.asarray(Rabs_prev, dtype=np.float64).reshape(-1) ** 2)                # color ~ r_{h,i}^2
        snap = {"centers": centers, "h2": h2, "r2": r2}

        return xs, ys, snap

    curves = {}
    snaps18 = {}

    x, y, snap = branch_run(1, 4, "S1 CM=4")
    curves["S1_CM4"] = (x, y); snaps18["S1_CM4"] = snap

    x, y, snap = branch_run(2, 4, "S2 CM=4")
    curves["S2_CM4"] = (x, y); snaps18["S2_CM4"] = snap

    x, y, snap = branch_run(1, 9, "S1 CM=9")
    curves["S1_CM9"] = (x, y); snaps18["S1_CM9"] = snap

    x, y, snap = branch_run(2, 9, "S2 CM=9")
    curves["S2_CM9"] = (x, y); snaps18["S2_CM9"] = snap

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

    out_png = os.path.join(outdir, "figure17_problem35.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 17 curves NPZ ----
    np.savez(os.path.join(outdir, "figure17_problem35_curves.npz"),
             **{k+"_x": np.asarray(v[0], np.float64) for k,v in curves.items()},
             **{k+"_y": np.asarray(v[1], np.float64) for k,v in curves.items()},
             base_x=np.asarray([x0,x1], np.float64),
             base_y=np.asarray([h1_0,h1_1], np.float64))

    # ---- Figure 18 NPZ payload ----
    holes_arr = []
    for H in holes:
        xmin, xmax, ymin, ymax = hole_bounds(H)
        holes_arr.append([xmin, ymin, xmax, ymax])
    holes_arr = np.asarray(holes_arr, dtype=np.float64)  # (4,4) [x0,y0,x1,y1]

    npz18 = {"holes_xyxy": holes_arr}
    for key, snap in snaps18.items():
        npz18[f"{key}_centers"] = snap["centers"]
        npz18[f"{key}_h2"]      = snap["h2"]
        npz18[f"{key}_r2"]      = snap["r2"]

    out_npz18 = os.path.join(outdir, "figure18_problem35_patches.npz")
    np.savez(out_npz18, **npz18)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_npz18}")
    return out_png


# -------------------------
# Main
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="MF-VPINN Figure 17 (Problem 35) – Strategy 1&2, CM=4/9 + Fig18 NPZ.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--target-npatch", type=int, default=1000)
    parser.add_argument("--pmax", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--snapshot-every", type=int, default=1)

    # training knobs (expose important ones)
    parser.add_argument("--adam-steps", type=int, default=Config.ADAM_STEPS_FIRST)
    parser.add_argument("--lbfgs-first", type=int, default=Config.LBFGS_MAXITER_FIRST)
    parser.add_argument("--lbfgs-next", type=int, default=Config.LBFGS_MAXITER_NEXT)
    parser.add_argument("--lbfgs-patch-batch", type=int, default=Config.LBFGS_PATCH_BATCH)
    parser.add_argument("--lbfgs-max-patches", type=int, default=10000)

    # safety / batching
    parser.add_argument("--adam-patch-batch", type=int, default=Config.ADAM_PATCH_BATCH)
    parser.add_argument("--adam-grad-accum", type=int, default=Config.ADAM_GRAD_ACCUM)
    parser.add_argument("--eval-patch-batch", type=int, default=Config.EVAL_PATCH_BATCH)

    # ADF knobs
    parser.add_argument("--adf-m", type=int, default=Config.ADF_M)

    # cutting
    parser.add_argument("--cut-overlap", type=float, default=Config.CUT_SPLIT_OVERLAP)

    # ----- new CLI knobs for experiments -----
    parser.add_argument("--adam-steps-after", type=int, default=Config.ADAM_STEPS_AFTER,
                        help="Warm-Adam steps after each refine (overrides Config)")
    parser.add_argument("--gamma-alpha", type=float, default=None,
                        help="Override GAMMA_ALPHA in Config (use 1.0 to disable softening)")
    parser.add_argument("--gamma-max", type=float, default=None,
                        help="Override GAMMA_MAX in Config")
    parser.add_argument("--lbfgs-ftol", type=float, default=None,
                        help="Override LBFGS_FTOL in Config")


    args = parser.parse_args(argv)

    cfg = Config()
    cfg.SEED = int(args.seed)

    cfg.MARK_FRAC = 0.70
    cfg.GAMMA_ALPHA = 0.5
    cfg.ADAM_STEPS_AFTER = 4000

    # apply CLI overrides (safe assignments)
    cfg.ADAM_STEPS_FIRST = int(args.adam_steps)
    cfg.LBFGS_MAXITER_FIRST = int(args.lbfgs_first)
    cfg.LBFGS_MAXITER_NEXT = int(args.lbfgs_next)
    cfg.LBFGS_PATCH_BATCH = int(args.lbfgs_patch_batch)

    cfg.ADAM_PATCH_BATCH = int(args.adam_patch_batch)
    cfg.ADAM_GRAD_ACCUM = int(args.adam_grad_accum)
    cfg.EVAL_PATCH_BATCH = int(args.eval_patch_batch)

    cfg.ADF_M = int(args.adf_m)
    cfg.CUT_SPLIT_OVERLAP = float(args.cut_overlap)

    # NEW: apply optional overrides
    if args.adam_steps_after is not None:
        cfg.ADAM_STEPS_AFTER = int(args.adam_steps_after)
    if args.gamma_alpha is not None:
        cfg.GAMMA_ALPHA = float(args.gamma_alpha)
    if args.gamma_max is not None:
        cfg.GAMMA_MAX = float(args.gamma_max)
    if args.lbfgs_ftol is not None:
        cfg.LBFGS_FTOL = float(args.lbfgs_ftol)


    outdir = args.outdir or os.getcwd()
    os.makedirs(outdir, exist_ok=True)

    print(f"Output dir: {outdir}")
    print(f"Seed: {cfg.SEED}")
    print(f"ADF_M: {cfg.ADF_M}")
    print(f"Cut overlap factor: {cfg.CUT_SPLIT_OVERLAP}")
    print(f"MARK_FRAC={cfg.MARK_FRAC}  MARK_CAP={cfg.MARK_CAP}")
    print(f"ADAM_STEPS_FIRST={cfg.ADAM_STEPS_FIRST}  LBFGS_FIRST={cfg.LBFGS_MAXITER_FIRST}  LBFGS_NEXT={cfg.LBFGS_MAXITER_NEXT}")
    print(f"ADAM_PATCH_BATCH={cfg.ADAM_PATCH_BATCH}  ADAM_GRAD_ACCUM={cfg.ADAM_GRAD_ACCUM}  LBFGS_PATCH_BATCH={cfg.LBFGS_PATCH_BATCH}")

    t0 = time.time()
    run_fig17(cfg, outdir, lbfgs_max_patches=int(args.lbfgs_max_patches))
    print(f"Elapsed [s]: {time.time()-t0:.2f}")


if __name__ == "__main__":
    main()
