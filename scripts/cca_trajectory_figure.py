"""Action-subspace trajectory figure via linear probe SVD (Option C).

For each of 6 backbones (3 families × {frozen, +ID}):
  1. Encode 200 LIBERO test-OOD episodes (clip_len=32) -> features X (N, D) +
     actions Y (N, 7).
  2. Fit Ridge regression Y ≈ X @ W on (N, D) -> (D, 7) weight matrix.
  3. SVD on W -> top-2 right singular directions U[:, :2] ∈ R^(D, 2).
  4. Project features onto these 2 directions.
  5. Plot 10 episodes' trajectories in this 2D space, colored by gripper / time.

This gives a paper-aligned 2D subspace: it's literally the 2D subspace where
the linear probe's predictions live. Per-panel R² is the held-out R² of the
Ridge regressor, which should rank-match the paper's MLP-probe R² ordering.

Usage:
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/cca_trajectory_figure.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from data.libero_video import LiberoVideoClips, worker_init_fn
from comparison.eval_pipeline import build_extractor


BACKBONES = [
    ("vjepa2_frozen",    "V-JEPA 2 ViT-L (frozen)",
     None, "vjepa2", None, 256),
    ("webdino_frozen",   "Web-DINO ViT-L (frozen)",
     None, "webdino", "facebook/webssl-dino300m-full2b-224", 224),
    ("sdvae_frozen",     "SDXL VAE (frozen)",
     None, "sd3vae", "stabilityai/sdxl-vae", 128),
    ("vjepa2_aux",       "V-JEPA 2 ViT-L (+ ID)",
     "runs/comparison/vjepa2_aux_finetune/ckpt_last.pt", "vjepa2", None, 256),
    ("webdino_aux",      "Web-DINO ViT-L (+ ID)",
     "runs/comparison/webdino_aux_finetune/ckpt_last.pt", "webdino", "facebook/webssl-dino300m-full2b-224", 224),
    ("sdvae_aux",        "SDXL VAE (+ ID)",
     "runs/comparison/sd3vae_aux_finetune/ckpt_last.pt", "sd3vae", "stabilityai/sdxl-vae", 128),
]

N_FIT_EPISODES = 200    # for Ridge fit
N_PLOT_EPISODES = 5     # v2: 10 -> 5 for visual traceability
N_TEST_EPISODES = 100   # held-out for R² scoring
SMOOTH_WINDOW = 3       # v2: 3-frame moving average per episode


@torch.no_grad()
def get_features_per_episode(spec, ds, episode_idx, device):
    """Returns list[(feat (T,D), action (T,7), gripper (T,))]."""
    extractor = build_extractor(spec, device)
    outs = []
    for ep in episode_idx:
        clip, acts = ds[ep]                                            # (T, C, H, W), (T, 7)
        clip = clip.unsqueeze(0).to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = extractor.extract(clip)                                  # (1, T, D)
        f = f.float().cpu().squeeze(0)
        outs.append((f, acts[..., :7].float(), acts[..., 6].float()))
    del extractor
    torch.cuda.empty_cache()
    return outs


def _tica_subspace(eps_fit, tau: int = 1, reg: float = 1e-4) -> np.ndarray:
    """Time-lagged Independent Component Analysis: returns (D, 2) projection.

    Solves the generalized eigenvalue problem C_τ v = λ C_0 v where C_τ is the
    symmetric time-lag covariance and C_0 is the instantaneous covariance.
    Top-2 eigenvectors = 2 slowest-varying linear directions across episodes.
    """
    from scipy.linalg import eigh
    Xs_t  = np.concatenate([e[0].numpy()[:-tau] for e in eps_fit], axis=0)   # (N, D)
    Xs_lt = np.concatenate([e[0].numpy()[tau:]  for e in eps_fit], axis=0)   # (N, D)
    mu = (Xs_t.mean(axis=0) + Xs_lt.mean(axis=0)) / 2.0
    Xs_t  = Xs_t  - mu
    Xs_lt = Xs_lt - mu
    N = Xs_t.shape[0]; D = Xs_t.shape[1]
    C0   = (Xs_t.T @ Xs_t + Xs_lt.T @ Xs_lt) / (2 * N)
    Ctau = (Xs_t.T @ Xs_lt + Xs_lt.T @ Xs_t) / (2 * N)
    C0 = C0 + reg * np.eye(D, dtype=C0.dtype)
    eigvals, eigvecs = eigh(Ctau, C0)
    # ascending; want LARGEST = slowest
    top2 = eigvecs[:, -2:]                                                   # (D, 2)
    return top2, float(eigvals[-1]), float(eigvals[-2])


def fit_and_project(eps_fit, eps_plot, eps_test, alpha: float = 1.0,
                      method: str = "ridge_svd"):
    """Returns (projs, r2_held_out, r2_per_dim, score1, score2)."""
    X_fit = torch.cat([e[0] for e in eps_fit], dim=0).numpy()
    Y_fit = torch.cat([e[1] for e in eps_fit], dim=0).numpy()
    X_test = torch.cat([e[0] for e in eps_test], dim=0).numpy()
    Y_test = torch.cat([e[1] for e in eps_test], dim=0).numpy()

    # Always fit a Ridge probe for held-out R² (paper-comparable),
    # regardless of projection method.
    reg_probe = Ridge(alpha=alpha)
    reg_probe.fit(X_fit, Y_fit)
    pred_test = reg_probe.predict(X_test)
    r2 = r2_score(Y_test, pred_test, multioutput="raw_values")
    r2_agg = float(r2.mean())

    if method == "ridge_svd":
        W = reg_probe.coef_.T
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        top2 = U[:, :2]
        s1, s2 = float(S[0]), float(S[1])
    elif method == "tica":
        top2, s1, s2 = _tica_subspace(eps_fit, tau=8, reg=1e-4)
    else:
        raise ValueError(f"unknown method: {method}")

    projs = [f.numpy() @ top2 for f, _, _ in eps_plot]
    return projs, r2_agg, r2, s1, s2


def smooth_trajectory(traj: np.ndarray, window: int = 3) -> np.ndarray:
    """3-frame moving average; preserves endpoints by shrinking window at edges."""
    if traj.shape[0] < window:
        return traj
    out = np.zeros_like(traj)
    half = window // 2
    for i in range(traj.shape[0]):
        lo, hi = max(0, i - half), min(traj.shape[0], i + half + 1)
        out[i] = traj[lo:hi].mean(axis=0)
    return out


def plot_panel(ax, projs, eps_plot, label, r2_agg, episode_colors,
                  smooth_window: int = SMOOTH_WINDOW, axis_label: str = "Probe dir"):
    """Each trajectory uses a per-episode colormap that goes start→end."""
    from matplotlib.collections import LineCollection
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_title(label, fontsize=10)
    ax.grid(True, alpha=0.15)
    # Per-episode colormaps: pick distinct hues for episode separation +
    # within an episode, colormap goes start→end (light→dark)
    episode_cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges"]
    for i, (proj, (_, _, gripper)) in enumerate(zip(projs, eps_plot)):
        proj = smooth_trajectory(proj, smooth_window)
        T = proj.shape[0]
        cmap = plt.get_cmap(episode_cmaps[i % len(episode_cmaps)])
        # LineCollection for time-varying color along the trajectory
        segs = np.stack([proj[:-1], proj[1:]], axis=1)
        ts = np.linspace(0.35, 1.0, T - 1)
        lc = LineCollection(segs, cmap=cmap, linewidths=2.2,
                              norm=mcolors.Normalize(vmin=0, vmax=1))
        lc.set_array(ts)
        ax.add_collection(lc)
        # Start/end markers using cmap endpoints
        start_col = cmap(0.45); end_col = cmap(1.0)
        ax.scatter(proj[0, 0], proj[0, 1], s=95, marker="o",
                      facecolor=start_col, edgecolor="black", linewidth=1.0, zorder=5)
        ax.scatter(proj[-1, 0], proj[-1, 1], s=95, marker="X",
                      facecolor=end_col, edgecolor="black", linewidth=1.0, zorder=5)
        # Gripper transitions
        g = gripper.numpy()
        gop = g > 0.5
        if gop.sum() > 0:
            trans = np.where(gop[:-1] != gop[1:])[0]
            for t in trans:
                ax.scatter(proj[t, 0], proj[t, 1], s=45, marker="s",
                              facecolor="white", edgecolor=cmap(0.75), linewidth=1.2, zorder=4)
    ax.autoscale_view()
    ax.text(0.97, 0.97, f"linear-probe R² = {r2_agg:.2f}", transform=ax.transAxes,
              ha="right", va="top", fontsize=9,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("ridge_svd", "tica"), default="ridge_svd")
    ap.add_argument("--out-tag", default="v3",
                    help="version tag in output filename (e.g., v3 or tica)")
    args = ap.parse_args()

    device = torch.device("cuda")
    out_png = REPO_ROOT / f"runs/full_results/probe_subspace_trajectory_6panel_{args.out_tag}.png"
    out_pdf = REPO_ROOT / f"runs/full_results/probe_subspace_trajectory_6panel_{args.out_tag}.pdf"
    out_json = REPO_ROOT / f"runs/full_results/probe_subspace_trajectory_summary_{args.out_tag}.jsonl"
    method = args.method
    axis_label = "Probe dir" if method == "ridge_svd" else "TICA dir"

    ds = LiberoVideoClips(split_path="data/splits_taskheld/test_ood.json",
                          datasets_dir="datasets",
                          clip_len=32, cameras=["agentview_rgb"], resolution=128,
                          seed=0, return_actions=True)
    n_eps = len(ds)
    print(f"test_ood: {n_eps} episodes total")

    rng = np.random.RandomState(0)
    all_perm = rng.permutation(n_eps)
    fit_idx  = sorted(all_perm[:N_FIT_EPISODES].tolist())
    test_idx = sorted(all_perm[N_FIT_EPISODES:N_FIT_EPISODES + N_TEST_EPISODES].tolist())
    plot_idx = fit_idx[:N_PLOT_EPISODES]  # plot from fit set, deterministic
    print(f"fit episodes: {len(fit_idx)} (N_fit ≈ {len(fit_idx)*32} frames)")
    print(f"test episodes: {len(test_idx)} (held-out R² scoring)")
    print(f"plot episodes: {len(plot_idx)} (drawn)")

    colormap = plt.cm.tab10
    episode_colors = [colormap(i / 10.0) for i in range(10)]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    panel_order = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    # Marker legend (above panels)
    legend_handles = [
        plt.Line2D([0],[0], marker="o", markersize=10, markerfacecolor="grey",
                     markeredgecolor="black", linewidth=0, label="episode start"),
        plt.Line2D([0],[0], marker="X", markersize=10, markerfacecolor="grey",
                     markeredgecolor="black", linewidth=0, label="episode end"),
        plt.Line2D([0],[0], marker="s", markersize=8, markerfacecolor="white",
                     markeredgecolor="grey", linewidth=0, label="gripper open/close transition"),
        plt.Line2D([0],[0], color="grey", linewidth=2.2,
                     label="trajectory (light → dark = early → late frame)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
                  ncol=4, fontsize=9, frameon=False,
                  bbox_to_anchor=(0.5, -0.005))

    out_rows = []
    for idx, (variant, label, ckpt, family, hf_repo, img_size) in enumerate(BACKBONES):
        row, col = panel_order[idx]
        ax = axes[row, col]
        print(f"\n=== {variant} ===")
        spec = {"family": family, "name": variant, "img_size": img_size}
        if hf_repo: spec["hf_repo"] = hf_repo
        if ckpt:    spec["ckpt"]    = ckpt
        try:
            t0 = time.time()
            # batch: extract fit + test + plot together
            all_idx = sorted(set(fit_idx) | set(test_idx) | set(plot_idx))
            outs = get_features_per_episode(spec, ds, all_idx, device)
            ep_lookup = dict(zip(all_idx, outs))
            eps_fit  = [ep_lookup[i] for i in fit_idx]
            eps_test = [ep_lookup[i] for i in test_idx]
            eps_plot = [ep_lookup[i] for i in plot_idx]
            print(f"  encoded {len(all_idx)} eps in {time.time()-t0:.1f}s")

            projs, r2_agg, r2_per_dim, sigma1, sigma2 = fit_and_project(
                eps_fit, eps_plot, eps_test, alpha=1.0, method=method)
            score_lbl = "σ₁/σ₂" if method == "ridge_svd" else "λ₁/λ₂"
            print(f"  Ridge R² (held-out) = {r2_agg:.3f} | {score_lbl} = {sigma1:.3f}/{sigma2:.3f}")
            plot_panel(ax, projs, eps_plot, label, r2_agg, episode_colors,
                          axis_label=axis_label)
            out_rows.append({"variant": variant, "r2_held_out": r2_agg,
                              "r2_per_dim": r2_per_dim.tolist(),
                              "sigma1": sigma1, "sigma2": sigma2,
                              "n_fit_frames": len(fit_idx)*32, "n_test_frames": len(test_idx)*32})
        except Exception as e:
            print(f"  FAILED: {e}")
            ax.text(0.5, 0.5, f"{label}\nFAILED: {e}", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8)

    # (no suptitle; figure caption lives in the paper LaTeX)
    fig.tight_layout(rect=(0, 0.04, 1, 1))  # leave room for bottom legend
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nsaved -> {out_png}")
    with out_json.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
