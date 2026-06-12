# -*- coding: utf-8 -*-
"""
fig12_indicator_importance_patchless.py

Patchless analogue of fig12_indicator_importance.py.

Purpose
-------
Reproduce the *logic* of Section 3.3 ("The Importance of the Error Indicator")
for the patchless / anchor-based MF-VPINN code base:

  - fixed good anchor set on the left panel,
  - fixed bad anchor set on the right panel,
  - train a fresh MF-VPINN on each fixed anchor set,
  - track
        E_{S,m},
        ||u-u_NN||_{H1},
        loss,
    at regular checkpoints,
  - stop early when E_{S,m} stops improving.

Design choices
--------------
1) We intentionally *freeze* the anchor set during each run.  This matches the role of
   Figure 12 in the original paper: compare training behaviour on a good discretization
   versus a bad discretization.

2) We reuse the patchless machinery from `patchless_mfvpinn_strategy3.py`:
     - neural ansatz / lift / bubble,
     - weak residual evaluation,
     - energy whitening,
     - H1 GI-QMC evaluator,
     - exact solution / boundary machinery.

3) There is a real implementation-vs-draft ambiguity in your current patchless project:
     - the Python code stores snapshot eta-values from normalized *weak residuals*,
     - while the prose in draft4 Section 2.3 emphasizes the *strong residual*
       eta(z)=|Delta u_theta(z)|.
   Because of that, this script supports two indicator modes:

       --indicator-mode weak-residual   (default, code-consistent)
       --indicator-mode strong-anchor   (draft-consistent, evaluated at anchor centers)

   Defaulting to weak-residual is the safer choice if the goal is to stay faithful to the
   *current implementation* and snapshot semantics.

4) For auditability and lower bug-risk, training uses the dense cached path on a fixed
   global QMC cloud.  This is slower than the adaptive KDTree training path, but it is
   much easier to reason about for a fixed-anchor Section-3.3 diagnostic.

Typical usage
-------------
Good anchors = Strategy 1 final CM=9 snapshot (e.g. P613)
Bad anchors  = Strategy 3 initial CM=9 snapshot (P25)

    python fig12_indicator_importance_patchless.py \
        --left-npz  snapshots_strategy1_cm9.npz --left-key  P613 \
        --right-npz snapshots_strategy3_cm9.npz --right-key P25  \
        --lift coons \
        --indicator-mode weak-residual \
        --outdir out_fig12_patchless
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

import patchless_mfvpinn_strategy3 as mf


P_RE = re.compile(r"^P(\d+)_centers$")


# -----------------------------------------------------------------------------
# Snapshot helpers
# -----------------------------------------------------------------------------
def list_snapshot_keys(npz: np.lib.npyio.NpzFile) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for name in npz.files:
        m = P_RE.match(name)
        if m:
            base = f"P{int(m.group(1))}"
            items.append((base, int(npz[name].shape[0])))
    items.sort(key=lambda t: t[1])
    return items


def choose_snapshot_key(npz: np.lib.npyio.NpzFile, mode: str = "auto-max") -> tuple[str, int]:
    items = list_snapshot_keys(npz)
    if not items:
        raise RuntimeError("No P{k}_centers arrays found in snapshot npz.")
    if mode == "auto-min":
        return items[0]
    if mode == "auto-max":
        return items[-1]
    raise ValueError(f"Unknown auto mode: {mode}")


@dataclass
class AnchorSnapshot:
    key: str
    nanchor: int
    anchors: np.ndarray   # (M,2)
    ell: np.ndarray       # (M,1)
    eta0: Optional[np.ndarray] = None


@dataclass
class RuntimeState:
    unet: tf.keras.Model
    gnet: Optional[tf.keras.Model]
    lift_mode_int: int
    bubble_mode_tf: tf.Tensor
    softmin_tau_tf: tf.Tensor
    bubble_power_tf: tf.Tensor
    whiten_mode_tf: tf.Tensor
    Xq_tf: tf.Tensor
    b0_all_tf: tf.Tensor
    db0_all_tf: tf.Tensor
    anchors_tf: tf.Tensor
    ell_tf: tf.Tensor
    aw_tf: tf.Tensor
    G_full: np.ndarray
    denom_np: np.ndarray
    denom_tf: tf.Tensor
    h1_eval: mf.H1GIQMCEvaluator
    h1_ref_energy: float


@dataclass
class History:
    evals: list[float]
    ES: list[float]
    H1: list[float]
    Loss: list[float]
    LossTrain: list[float]
    LossFull: list[float]

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "evals": np.asarray(self.evals, dtype=np.float64),
            "ES": np.asarray(self.ES, dtype=np.float64),
            "H1": np.asarray(self.H1, dtype=np.float64),
            "Loss": np.asarray(self.Loss, dtype=np.float64),
            "LossTrain": np.asarray(self.LossTrain, dtype=np.float64),
            "LossFull": np.asarray(self.LossFull, dtype=np.float64),
        }


def load_anchor_snapshot(npz_path: str, *, key: Optional[str] = None, auto_mode: Optional[str] = None) -> AnchorSnapshot:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Snapshot file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    available = list_snapshot_keys(data)
    available_keys = [k for k, _ in available]

    if key is None:
        if auto_mode is None:
            raise ValueError("Either key or auto_mode must be provided.")
        key_base, nanchor = choose_snapshot_key(data, mode=auto_mode)
    else:
        key_base = key[:-len("_centers")] if key.endswith("_centers") else key
        if key_base not in available_keys:
            raise KeyError(
                f"Snapshot key '{key_base}' not found in {npz_path}. "
                f"Available keys: {available_keys}"
            )
        nanchor = int(data[f"{key_base}_centers"].shape[0])

    anchors = np.asarray(data[f"{key_base}_centers"], dtype=np.float64)
    ell = np.asarray(data[f"{key_base}_h"], dtype=np.float64).reshape(-1, 1)
    eta0 = None
    if f"{key_base}_eta" in data.files:
        eta0 = np.asarray(data[f"{key_base}_eta"], dtype=np.float64).reshape(-1)

    return AnchorSnapshot(
        key=key_base,
        nanchor=nanchor,
        anchors=anchors,
        ell=ell,
        eta0=eta0,
    )


# -----------------------------------------------------------------------------
# Shared config / runtime setup
# -----------------------------------------------------------------------------
def configure_module_kernel(cfg: mf.Config) -> None:
    mf.KERNEL_KIND = str(getattr(cfg, "KERNEL_KIND", "gauss"))
    try:
        mf.KERNEL_CUTOFF_FACTOR = float(getattr(cfg, "KERNEL_CUTOFF", 4.0))
    except Exception:
        pass


def build_runtime(
    snapshot: AnchorSnapshot,
    cfg: mf.Config,
    *,
    lift: str,
    seed: int,
) -> RuntimeState:
    tf.keras.backend.set_floatx("float64")
    mf.set_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    configure_module_kernel(cfg)

    unet = mf.MLP(width=cfg.WIDTH, layers=cfg.L_LAYERS, normalize_input=bool(cfg.DOMAIN_NORMALIZE))
    _ = unet.call(tf.zeros((1, 2), dtype=mf.DTYPE))

    gnet = None
    lift_mode_int = 0
    if lift == "gnet":
        gnet = mf.MLP(width=cfg.WIDTH, layers=cfg.L_LAYERS, normalize_input=bool(cfg.DOMAIN_NORMALIZE))
        _ = gnet.call(tf.zeros((1, 2), dtype=mf.DTYPE))
        mf.train_gnet_boundary(gnet, cfg, seed=seed)
        lift_mode_int = 1
    elif lift == "coons":
        lift_mode_int = 2
    else:
        lift_mode_int = 0

    bubble_mode_tf = tf.constant(0 if cfg.BUBBLE_KIND == "product" else 1, dtype=tf.int32)
    softmin_tau_tf = tf.constant(float(cfg.SOFTMIN_TAU), dtype=mf.DTYPE)
    bubble_power_tf = tf.constant(float(cfg.BUBBLE_POWER), dtype=mf.DTYPE)
    whiten_mode_tf = tf.constant(0 if str(cfg.WHITEN_MODE) == "energy" else 1, dtype=tf.int32)

    Xq_np, _ = mf.build_ref_cloud_square(int(cfg.NQ_GLOBAL), seed=seed + 11, cfg=cfg, dim=2)
    Xq_tf = tf.constant(Xq_np, dtype=mf.DTYPE)
    b0_all_tf, db0_all_tf = mf.bubble0_and_grad(Xq_tf)

    anchors = np.asarray(snapshot.anchors, dtype=np.float64)
    ell = np.asarray(snapshot.ell, dtype=np.float64).reshape(-1, 1)
    aw = np.ones((anchors.shape[0], 1), dtype=np.float64)

    G_full = mf.compute_G_full_cached(
        Xq_tf,
        b0_all_tf,
        db0_all_tf,
        anchors,
        ell,
        cfg,
        cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
    )
    denom_np = np.sqrt(np.maximum(np.asarray(G_full, dtype=np.float64).reshape(-1), 0.0) + float(cfg.WHITEN_EPS)).reshape(-1, 1)
    denom_np = np.maximum(denom_np, 1.0e-8)

    anchors_tf = tf.constant(anchors, dtype=mf.DTYPE)
    ell_tf = tf.constant(ell, dtype=mf.DTYPE)
    aw_tf = tf.constant(aw, dtype=mf.DTYPE)
    denom_tf = tf.constant(denom_np, dtype=mf.DTYPE)

    h1_eval = mf.H1GIQMCEvaluator.build(cfg, int(cfg.H1_HOLDOUT_SAMPLES), seed=int(cfg.H1_HOLDOUT_SEED) + seed)
    h1_ref_energy = float(h1_eval.reference_energy())

    return RuntimeState(
        unet=unet,
        gnet=gnet,
        lift_mode_int=lift_mode_int,
        bubble_mode_tf=bubble_mode_tf,
        softmin_tau_tf=softmin_tau_tf,
        bubble_power_tf=bubble_power_tf,
        whiten_mode_tf=whiten_mode_tf,
        Xq_tf=Xq_tf,
        b0_all_tf=b0_all_tf,
        db0_all_tf=db0_all_tf,
        anchors_tf=anchors_tf,
        ell_tf=ell_tf,
        aw_tf=aw_tf,
        G_full=np.asarray(G_full, dtype=np.float64).reshape(-1),
        denom_np=denom_np,
        denom_tf=denom_tf,
        h1_eval=h1_eval,
        h1_ref_energy=h1_ref_energy,
    )


# -----------------------------------------------------------------------------
# Indicator evaluation
# -----------------------------------------------------------------------------
def weak_residual_indicator_from_eval(ev: Dict[str, Any], runtime: RuntimeState, cfg: mf.Config) -> Tuple[np.ndarray, float]:
    Rabs = np.asarray(ev["Rabs"], dtype=np.float64).reshape(-1)
    denom = np.asarray(runtime.denom_np, dtype=np.float64).reshape(-1)
    eta = (Rabs / np.maximum(denom, 1.0e-30)) ** 2
    ES = float(np.sum(eta))
    return eta.astype(np.float64), ES


def strong_anchor_indicator(
    runtime: RuntimeState,
    cfg: mf.Config,
) -> Tuple[np.ndarray, float]:
    anchors = runtime.anchors_tf.numpy()
    N = int(anchors.shape[0])
    chunk = int(max(1, min(N, getattr(cfg, "EVAL_ANCHOR_BATCH", 128))))
    lift_mode_tf = tf.constant(int(runtime.lift_mode_int), dtype=tf.int32)

    outs = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xy_tf = tf.constant(anchors[s:e], dtype=mf.DTYPE)
        lap_tf = mf.laplace_u_tf(
            runtime.unet,
            runtime.gnet,
            lift_mode_tf,
            xy_tf,
            float(cfg.LAPLACE_H),
            float(cfg.BUBBLE_SCALE),
            runtime.bubble_mode_tf,
            runtime.softmin_tau_tf,
            runtime.bubble_power_tf,
        )
        outs.append(tf.abs(lap_tf).numpy().reshape(-1))

    eta = np.concatenate(outs, axis=0).astype(np.float64)
    ES = float(np.sum(eta))
    return eta, ES


def evaluate_checkpoint(
    runtime: RuntimeState,
    cfg: mf.Config,
    *,
    indicator_mode: str,
) -> Dict[str, Any]:
    ev = mf.eval_full_cached(
        runtime.unet,
        runtime.gnet,
        runtime.lift_mode_int,
        runtime.Xq_tf,
        runtime.b0_all_tf,
        runtime.db0_all_tf,
        runtime.anchors_tf.numpy(),
        runtime.ell_tf.numpy(),
        runtime.aw_tf.numpy(),
        runtime.G_full,
        cfg,
        runtime.bubble_mode_tf,
        runtime.softmin_tau_tf,
        runtime.bubble_power_tf,
        runtime.whiten_mode_tf,
        cutoff_factor=float(getattr(cfg, "KERNEL_CUTOFF", 4.0)),
    )

    indicator_mode = indicator_mode.strip().lower()
    if indicator_mode == "strong-anchor":
        eta, ES = strong_anchor_indicator(runtime, cfg)
    elif indicator_mode == "weak-residual":
        eta, ES = weak_residual_indicator_from_eval(ev, runtime, cfg)
    else:
        raise ValueError(f"Unknown indicator mode: {indicator_mode}")

    H1_rel = float(runtime.h1_eval.rel_H1(
        runtime.unet,
        runtime.gnet,
        runtime.lift_mode_int,
        runtime.bubble_mode_tf,
        runtime.softmin_tau_tf,
        runtime.bubble_power_tf,
        ref_energy_const=runtime.h1_ref_energy,
    ))

    return {
        "eta": eta,
        "ES": ES,
        "H1": H1_rel,
        "loss_full": float(ev["loss"]),
        "Rabs": np.asarray(ev["Rabs"], dtype=np.float64).reshape(-1),
    }


# -----------------------------------------------------------------------------
# Dense cached training on a fixed anchor set
# -----------------------------------------------------------------------------
def build_dense_train_step(
    runtime: RuntimeState,
    cfg: mf.Config,
) -> tuple[tf.keras.optimizers.Optimizer, Any]:
    vars_ = runtime.unet.trainable_variables
    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.LR0))
    reg_coeff = tf.constant(float(cfg.LAMBDA_REG), dtype=mf.DTYPE)
    lift_mode_tf = tf.constant(int(runtime.lift_mode_int), dtype=tf.int32)
    sum_aw_total_tf = tf.maximum(tf.reduce_sum(runtime.aw_tf), tf.cast(1.0e-30, mf.DTYPE))
    M_total_tf = tf.constant(int(runtime.anchors_tf.shape[0]), dtype=mf.DTYPE)

    cutoff_s = None
    if float(getattr(cfg, "KERNEL_CUTOFF", 0.0)) > 0.0:
        cutoff_s = tf.constant(float(getattr(cfg, "KERNEL_CUTOFF", 4.0)) ** 2, dtype=mf.DTYPE)

    use_energy_reg = float(getattr(cfg, "W_ENERGY", 0.0)) != 0.0

    @tf.function(reduce_retracing=True)
    def train_step(xy, b0, db0, xi, el, aw, den):
        with tf.GradientTape() as tape:
            gu = mf.grad_u_tf(
                runtime.unet,
                runtime.gnet,
                lift_mode_tf,
                xy,
                float(cfg.BUBBLE_SCALE),
                runtime.bubble_mode_tf,
                runtime.softmin_tau_tf,
                runtime.bubble_power_tf,
            )
            gvx, gvy = mf.gradv_from_precomputed(xy, b0, db0, xi, el, cutoff_s=cutoff_s)
            dot = gu[:, 0:1] * gvx + gu[:, 1:2] * gvy
            R = tf.reduce_mean(dot, axis=0)
            r = R[:, None] / den

            num = tf.reduce_sum(aw * tf.square(r))
            A = tf.cast(tf.shape(aw)[0], mf.DTYPE)
            L_pde = (M_total_tf / tf.maximum(A, tf.cast(1.0, mf.DTYPE))) * (num / sum_aw_total_tf)

            if use_energy_reg:
                L_energy = tf.reduce_mean(tf.reduce_sum(tf.square(gu), axis=1))
            else:
                L_energy = tf.cast(0.0, mf.DTYPE)

            L = tf.cast(cfg.W_PDE, mf.DTYPE) * L_pde + tf.cast(cfg.W_ENERGY, mf.DTYPE) * L_energy
            reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) if vars_ else tf.cast(0.0, mf.DTYPE)
            J = L + reg_coeff * reg

        grads = tape.gradient(J, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return J

    return opt, train_step


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------
def train_with_indicator_tracking_patchless(
    snapshot: AnchorSnapshot,
    cfg: mf.Config,
    *,
    lift: str = "coons",
    indicator_mode: str = "weak-residual",
    m_index: int = 0,
    Ncheck: int = 10,
    patience: int = 10,
    max_epochs: int = 2000,
    seed: int = 1234,
    loss_source: str = "full",
) -> tuple[RuntimeState, Dict[str, np.ndarray]]:
    runtime = build_runtime(snapshot, cfg, lift=lift, seed=seed)
    _, train_step = build_dense_train_step(runtime, cfg)

    Nq = int(runtime.Xq_tf.shape[0])
    M = int(runtime.anchors_tf.shape[0])
    point_batch = int(min(max(1, cfg.ADAM_POINT_BATCH), Nq))
    anchor_batch = int(min(max(1, cfg.ADAM_ANCHOR_BATCH), M))

    rng = np.random.default_rng(int(seed) + 12345)

    Ncheck = int(Ncheck)
    patience = int(patience)
    Nnegl = 100 * (int(m_index) + 1)

    history = History(evals=[], ES=[], H1=[], Loss=[], LossTrain=[], LossFull=[])

    best_ES = np.inf
    best_eval = 0
    best_vec: Optional[np.ndarray] = None
    no_improve_epochs = 0
    eval_index = 0

    anchor_ids = np.arange(M, dtype=np.int32)

    print(
        f"  [train] snapshot={snapshot.key}  M={M}  pb={point_batch}  ab={anchor_batch}  "
        f"NQ={Nq}  max_epochs={max_epochs}  indicator={indicator_mode}"
    )
    t0 = time.time()

    for epoch in range(int(max_epochs)):
        rng.shuffle(anchor_ids)
        epoch_loss_acc = 0.0
        nbatches = 0

        for s in range(0, M, anchor_batch):
            e = min(s + anchor_batch, M)
            ids = anchor_ids[s:e]
            pids = rng.integers(0, Nq, size=point_batch, dtype=np.int32)

            ids_tf = tf.constant(ids, dtype=tf.int32)
            pids_tf = tf.constant(pids, dtype=tf.int32)

            xy = tf.gather(runtime.Xq_tf, pids_tf)
            b0 = tf.gather(runtime.b0_all_tf, pids_tf)
            db0 = tf.gather(runtime.db0_all_tf, pids_tf)
            xi = tf.gather(runtime.anchors_tf, ids_tf)
            el = tf.gather(runtime.ell_tf, ids_tf)
            aw = tf.gather(runtime.aw_tf, ids_tf)
            den = tf.gather(runtime.denom_tf, ids_tf)

            J = train_step(xy, b0, db0, xi, el, aw, den)
            epoch_loss_acc += float(J.numpy())
            nbatches += 1

        epoch_train_loss = epoch_loss_acc / max(nbatches, 1)

        if epoch >= Nnegl and ((epoch - Nnegl) % Ncheck == 0):
            eval_index += 1
            chk = evaluate_checkpoint(runtime, cfg, indicator_mode=indicator_mode)

            if str(loss_source).strip().lower() == "train":
                loss_plot = float(epoch_train_loss)
            else:
                loss_plot = float(chk["loss_full"])

            history.evals.append(float(eval_index))
            history.ES.append(float(chk["ES"]))
            history.H1.append(float(chk["H1"]))
            history.Loss.append(float(loss_plot))
            history.LossTrain.append(float(epoch_train_loss))
            history.LossFull.append(float(chk["loss_full"]))

            print(
                f"    [epoch {epoch:4d}] eval={eval_index:4d}  ES={chk['ES']:.3e}  "
                f"H1={chk['H1']:.3e}  loss(train/full)={epoch_train_loss:.3e}/{chk['loss_full']:.3e}"
            )

            if chk["ES"] < best_ES:
                best_ES = float(chk["ES"])
                best_eval = int(eval_index)
                best_vec = mf.pack_weights(runtime.unet).copy()
                no_improve_epochs = 0
            else:
                no_improve_epochs += Ncheck
                if no_improve_epochs >= patience * Ncheck:
                    print(
                        f"  [early-stop] epoch={epoch}, best_eval={best_eval}, best_ES={best_ES:.3e}",
                        flush=True,
                    )
                    break

    if best_vec is not None:
        mf.unpack_weights(runtime.unet, best_vec)

    print(f"  [train] elapsed = {time.time() - t0:.2f}s")
    return runtime, history.as_dict()


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_figure12_patchless(
        hist_left: Dict[str, np.ndarray],
        hist_right: Dict[str, np.ndarray],
        *,
        out_png: str,
        left_label: str,
        right_label: str,
        loss_log_scale: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=200)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def _maybe_rescale_evals_to_paper_units(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        if x.size == 0:
            return x
        x0 = x.min()
        x = x - x0
        if x.size >= 2:
            dx = np.diff(x)
            is_unit_step = np.allclose(dx, 1.0)
            is_index_ending = np.isclose(x.max(), x.size - 1)
            if is_unit_step and is_index_ending:
                if x.max() > 0:
                    x = x * (1000.0 / x.max())
        return x

    def _panel(ax_top, ax_bottom, hist, title_top, title_bottom, panel_top, panel_bottom, es_label: str):
        x_raw = np.asarray(hist["evals"], dtype=float)
        x = _maybe_rescale_evals_to_paper_units(x_raw)

        ES = np.asarray(hist["ES"], dtype=float)
        H1 = np.asarray(hist["H1"], dtype=float)
        Loss = np.asarray(hist["Loss"], dtype=float)

        if x.size == 0:
            raise RuntimeError("Empty history encountered while plotting Figure 12.")

        c = ES[0] / H1[0] if H1[0] > 0.0 else 1.0

        # فقط برای شکل‌های (a) و (b) که ax_top هستند:
        ax_top.plot(x, ES, "o-", label=es_label)
        ax_top.plot(x, c * H1, "s-", label=r"$c\,\|u-u_{NN}\|_{H^1}$")
        ax_top.set_xlabel("Entire dataset VPINN evaluations")
        ax_top.set_ylabel("Indicator / scaled error")
        ax_top.set_title(title_top)
        ax_top.legend(loc="best")
        ax_top.text(0.02, 0.95, panel_top, transform=ax_top.transAxes, ha="left", va="top", fontsize=12,
                    fontweight="bold")

        # برای شکل‌های (c) و (d) که ax_bottom هستند (بدون تغییر و بدون es_label):
        ax_bottom.plot(x, Loss, "o-")
        ax_bottom.set_xlabel("Entire dataset VPINN evaluations")
        ax_bottom.set_ylabel("Loss")
        ax_bottom.set_title(title_bottom)
        if loss_log_scale:
            ax_bottom.set_yscale("log")
        ax_bottom.text(0.02, 0.95, panel_bottom, transform=ax_bottom.transAxes, ha="left", va="top", fontsize=12,
                       fontweight="bold")

        ticks = [0, 200, 400, 600, 800, 1000]
        ax_top.set_xticks(ticks)
        ax_bottom.set_xticks(ticks)
        if x.max() <= 1020.0:
            ax_top.set_xlim(0.0, 1000.0)
            ax_bottom.set_xlim(0.0, 1000.0)

    # ترسیم شکل (a) در بالا-چپ و شکل (c) در پایین-چپ
    # در اینجا لیبل $E_{S,9}$ فقط به (a) داده می‌شود
    _panel(ax_a, ax_c, hist_left, left_label, f"Loss ({left_label})", "(a)", "(c)", es_label=r"$ES_{9}$")

    # ترسیم شکل (b) در بالا-راست و شکل (d) در پایین-راست
    # در اینجا لیبل $E_{S,0}$ فقط به (b) داده می‌شود
    _panel(ax_b, ax_d, hist_right, right_label, f"Loss ({right_label})", "(b)", "(d)", es_label=r"$ES_{0}$")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure12] Saved: {out_png}")
# -----------------------------------------------------------------------------
# History / metadata I/O
# -----------------------------------------------------------------------------
def save_history_npz(path: str, hist: Dict[str, np.ndarray]) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in hist.items()})


def write_run_summary(path: str, *, args: argparse.Namespace, cfg: mf.Config,
                      left: AnchorSnapshot, right: AnchorSnapshot) -> None:
    lines = ["=== Patchless Figure 12 run summary ===", f"left_npz   = {args.left_npz}",
             f"left_key   = {left.key}  (M={left.nanchor})", f"right_npz  = {args.right_npz}",
             f"right_key  = {right.key}  (M={right.nanchor})", f"lift       = {args.lift}",
             f"indicator  = {args.indicator_mode}", f"loss_src   = {args.loss_source}", f"m_left     = {args.m_left}",
             f"m_right    = {args.m_right}", f"Ncheck     = {args.Ncheck}", f"patience   = {args.patience}",
             f"max_epochs = {args.max_epochs}", f"kernel     = {cfg.KERNEL_KIND}", f"cutoff     = {cfg.KERNEL_CUTOFF}",
             f"NQ_GLOBAL  = {cfg.NQ_GLOBAL}", f"adam_pb    = {cfg.ADAM_POINT_BATCH}",
             f"adam_ab    = {cfg.ADAM_ANCHOR_BATCH}", f"eval_pb    = {cfg.EVAL_POINT_BATCH}",
             f"eval_ab    = {cfg.EVAL_ANCHOR_BATCH}", f"gnet_steps = {cfg.GNET_STEPS}", f"seed       = {cfg.SEED}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def make_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Patchless MF-VPINN: Figure 12 indicator-importance experiment.")

    # Snapshot selection
    ap.add_argument("--left-npz", type=str, required=True,
                    help="NPZ file containing the good/refined anchor set, typically Strategy 1 CM=9.")
    ap.add_argument("--left-key", type=str, default="P613",
                    help="Snapshot key for the left panel. Use e.g. P613 or auto-max.")
    ap.add_argument("--right-npz", type=str, required=True,
                    help="NPZ file containing the bad/coarse anchor set, typically Strategy 3 CM=9.")
    ap.add_argument("--right-key", type=str, default="P25",
                    help="Snapshot key for the right panel. Use e.g. P25 or auto-min.")

    # Section 3.3 control
    ap.add_argument("--m-left", type=int, default=0,
                    help="Adaptive-iteration index m for the left anchor set; default matches P613 from Step 09.")
    ap.add_argument("--m-right", type=int, default=3,
                    help="Adaptive-iteration index m for the right anchor set; default matches P25 from Step 00.")
    ap.add_argument("--Ncheck", type=int, default=10, help="Checkpoint spacing in epochs.")
    ap.add_argument("--patience", type=int, default=20, help="Early-stopping patience multiplier p.")
    ap.add_argument("--max-epochs", type=int, default=860, help="Maximum number of training epochs.")

    # Model / indicator / plotting
    ap.add_argument("--lift", type=str, default="coons", choices=["exact", "gnet", "coons"],
                    help="Boundary lift used by the patchless solver.")
    ap.add_argument("--indicator-mode", type=str, default="weak-residual",
                    choices=["weak-residual", "strong-anchor"],
                    help="weak-residual matches current Python snapshots; strong-anchor matches draft Section 2.3 prose.")
    ap.add_argument("--loss-source", type=str, default="full", choices=["full", "train"],
                    help="full = plot checkpoint full-anchor loss; train = plot epoch-average stochastic training loss.")
    ap.add_argument("--loss-log-scale", type=int, default=1, choices=[0, 1],
                    help="Use log-scale on the loss panels, like the paper figure.")

    # Shared solver config for both left/right runs
    ap.add_argument("--kernel", type=str, default="wendland_c2", choices=["gauss", "wendland_c2"])
    ap.add_argument("--kernel-cutoff", type=float, default=4.0)
    ap.add_argument("--nq-global", type=int, default=65536,
                    help="Shared global QMC cloud size used for both fixed-anchor runs.")
    ap.add_argument("--adam-point-batch", type=int, default=4096)
    ap.add_argument("--adam-anchor-batch", type=int, default=64)
    ap.add_argument("--eval-point-batch", type=int, default=8192)
    ap.add_argument("--eval-anchor-batch", type=int, default=128)
    ap.add_argument("--gnet-steps", type=int, default=20000)
    ap.add_argument("--h1-holdout-samples", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=2000)

    # Output
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory. Default: current working directory.")
    return ap


def build_cfg_from_args(args: argparse.Namespace) -> mf.Config:
    cfg = mf.Config()

    # Fixed-anchor Figure-12 defaults: accuracy/stability first.
    cfg.KERNEL_KIND = str(args.kernel)
    cfg.KERNEL_CUTOFF = float(args.kernel_cutoff)
    cfg.NQ_GLOBAL = int(args.nq_global)
    cfg.ADAM_POINT_BATCH = int(args.adam_point_batch)
    cfg.ADAM_ANCHOR_BATCH = int(args.adam_anchor_batch)
    cfg.EVAL_POINT_BATCH = int(args.eval_point_batch)
    cfg.EVAL_ANCHOR_BATCH = int(args.eval_anchor_batch)
    cfg.GNET_STEPS = int(args.gnet_steps)
    cfg.H1_HOLDOUT_SAMPLES = int(args.h1_holdout_samples)
    cfg.SEED = int(args.seed)

    cfg.BUBBLE_KIND = "product"
    cfg.BUBBLE_POWER = 1.0
    cfg.WHITEN_MODE = "energy"

    # Fixed-anchor Section 3.3 experiment: avoid adaptive/KDTree complications.
    cfg.USE_KDTREE_PRUNING = 0
    cfg.W_ENERGY = 0.0

    return cfg


def main(argv: Optional[list[str]] = None) -> None:
    ap = make_argparser()
    args = ap.parse_args(argv)

    outdir = args.outdir or os.getcwd()
    os.makedirs(outdir, exist_ok=True)

    cfg = build_cfg_from_args(args)
    configure_module_kernel(cfg)

    print("=== Patchless Figure 12: indicator importance ===")
    print(f"Output dir : {outdir}")
    print(f"Lift       : {args.lift}")
    print(f"Indicator  : {args.indicator_mode}")
    print(f"Loss source: {args.loss_source}")
    print(f"Kernel     : {cfg.KERNEL_KIND}  cutoff={cfg.KERNEL_CUTOFF:g}")
    print(f"NQ_GLOBAL  : {cfg.NQ_GLOBAL}")
    print(f"Batches    : adam(pb={cfg.ADAM_POINT_BATCH}, ab={cfg.ADAM_ANCHOR_BATCH})  "
          f"eval(pb={cfg.EVAL_POINT_BATCH}, ab={cfg.EVAL_ANCHOR_BATCH})")
    print(f"Ncheck={args.Ncheck}, patience={args.patience}, max_epochs={args.max_epochs}")
    print("")

    if args.left_key in ("auto-max", "auto-min"):
        left = load_anchor_snapshot(args.left_npz, key=None, auto_mode=args.left_key)
    else:
        left = load_anchor_snapshot(args.left_npz, key=args.left_key, auto_mode=None)

    if args.right_key in ("auto-max", "auto-min"):
        right = load_anchor_snapshot(args.right_npz, key=None, auto_mode=args.right_key)
    else:
        right = load_anchor_snapshot(args.right_npz, key=args.right_key, auto_mode=None)

    print(f"[left ] loaded {left.key}  with M={left.nanchor}  from {args.left_npz}")
    print(f"[right] loaded {right.key} with M={right.nanchor} from {args.right_npz}")

    write_run_summary(
        os.path.join(outdir, "run_summary.txt"),
        args=args,
        cfg=cfg,
        left=left,
        right=right,
    )

    runtime_L, hist_L = train_with_indicator_tracking_patchless(
        left,
        copy.deepcopy(cfg),
        lift=args.lift,
        indicator_mode=args.indicator_mode,
        m_index=int(args.m_left),
        Ncheck=int(args.Ncheck),
        patience=int(args.patience),
        max_epochs=int(args.max_epochs),
        seed=int(cfg.SEED),
        loss_source=args.loss_source,
    )
    save_history_npz(os.path.join(outdir, "history_left.npz"), hist_L)

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    # Rebuild cfg/module state after clear_session.
    cfg_R = build_cfg_from_args(args)
    configure_module_kernel(cfg_R)

    runtime_R, hist_R = train_with_indicator_tracking_patchless(
        right,
        copy.deepcopy(cfg_R),
        lift=args.lift,
        indicator_mode=args.indicator_mode,
        m_index=int(args.m_right),
        Ncheck=int(args.Ncheck),
        patience=int(args.patience),
        max_epochs=int(args.max_epochs),
        seed=int(cfg_R.SEED) + 1000,
        loss_source=args.loss_source,
    )
    save_history_npz(os.path.join(outdir, "history_right.npz"), hist_R)

    plot_figure12_patchless(
        hist_L,
        hist_R,
        out_png=os.path.join(outdir, "figure12_patchless.png"),
        left_label=f"Strategy 1 good anchors ({left.key}, M={left.nanchor})",
        right_label=f"Strategy 3 weak anchors ({right.key}, M={right.nanchor})",
        loss_log_scale=bool(int(args.loss_log_scale)),
    )


if __name__ == "__main__":
    main()
