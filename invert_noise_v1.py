import os
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

import sys
sys.path.append('./dppo')

from dppo.agent.dataset.sequence import StitchedSequenceDataset
from dppo.model.diffusion.diffusion_eval import DiffusionEval
from dppo.script.download_url import (
    get_dataset_download_url,
    get_checkpoint_download_url,
)
import gdown

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)


def compute_ddim_inverse_noise(model: DiffusionEval, x0: torch.Tensor, cond: dict) -> torch.Tensor:
    """
    DDIM inversion to recover initial latent x_T (noise_action) that produces given x0 under deterministic DDIM.

    Args:
        model: DiffusionEval (predict_epsilon=True, use_ddim=True, eta=0)
        x0: (B, Ta, Da) target action chunk
        cond: dict with 'state': (B, To, Do)

    Returns:
        xT: (B, Ta, Da) inverted initial latent
    """
    assert model.use_ddim, "Inversion requires DDIM schedule"
    device = x0.device
    B = x0.shape[0]
    x = x0.clone()
    T = len(model.ddim_t)
    # Iterate backwards over the DDIM schedule to recover x_T from x_0
    for i in reversed(range(T)):
        if i == 0:
            break  # x at i=0 corresponds to x_T after this loop
        # At index i, current latent x corresponds to x_{i}
        t_scalar = int(model.ddim_t[i].item())
        t_b = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        # Predict epsilon at time i using current latent x_i
        eps = model.actor(x, t_b, cond=cond)

        # Gather alphas for current index i and its previous (i-1)
        alpha_i = model.ddim_alphas[i].to(device)
        alpha_prev = model.ddim_alphas_prev[i].to(device)
        while alpha_i.dim() < x.dim():
            alpha_i = alpha_i.view(-1, *([1] * (x.dim() - 1)))
            alpha_prev = alpha_prev.view(-1, *([1] * (x.dim() - 1)))

        # Invert one deterministic DDIM step: x_{i-1} from x_i
        x = (x - (1.0 - alpha_prev).sqrt() * eps) * (alpha_i.sqrt() / alpha_prev.sqrt()) + (1.0 - alpha_i).sqrt() * eps

    return x


def log_plots_wandb(noises_np: np.ndarray, tag_prefix: str, max_points: int = 5000, labels=None, color_by=None):
    N, D = noises_np.shape
    sel = np.random.choice(N, size=min(N, max_points), replace=False) if N > max_points else np.arange(N)
    X = noises_np[sel]
    labels_sel = labels[sel] if labels is not None else None
    color_by_sel = color_by[sel] if color_by is not None else None

    # Distributions
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(np.linalg.norm(X, axis=1), bins=50)
    ax[0].set_title("Noise L2 norm")
    ax[1].hist(X.flatten(), bins=50)
    ax[1].set_title("Noise values")
    wandb.log({f"{tag_prefix}/histograms": wandb.Image(fig)})
    plt.close(fig)

    # PCA 2D
    try:
        pca = PCA(n_components=2)
        p = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(5, 5))
        if color_by_sel is not None:
            sc = ax.scatter(p[:, 0], p[:, 1], c=color_by_sel, s=3, alpha=0.8, cmap='viridis')
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('time proportion')
        elif labels_sel is None:
            ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.6)
        else:
            uniq = np.unique(labels_sel)
            n_lbl = len(uniq)
            if n_lbl <= 10:
                cmap = plt.cm.get_cmap('tab10', n_lbl)
            elif n_lbl <= 20:
                cmap = plt.cm.get_cmap('tab20', n_lbl)
            else:
                cmap = plt.cm.get_cmap('gist_ncar', n_lbl)
            sc = ax.scatter(p[:, 0], p[:, 1], c=labels_sel, s=3, alpha=0.6, cmap=cmap)
            plt.colorbar(sc, ax=ax)
        ax.set_title("PCA (2D)")
        wandb.log({f"{tag_prefix}/pca2d": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        wandb.log({f"{tag_prefix}/pca2d_error": str(e)})

    # PCA 3D
    try:
        pca3 = PCA(n_components=3)
        p3 = pca3.fit_transform(X)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        if color_by_sel is not None:
            sc = ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=color_by_sel, s=6, alpha=0.7, cmap='viridis')
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('time proportion')
        elif labels_sel is None:
            ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], s=6, alpha=0.7)
        else:
            uniq = np.unique(labels_sel)
            n_lbl = len(uniq)
            cmap = plt.cm.get_cmap('tab20', n_lbl) if n_lbl > 10 else plt.cm.get_cmap('tab10', n_lbl)
            sc = ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=labels_sel, s=6, alpha=0.7, cmap=cmap)
            plt.colorbar(sc, ax=ax, pad=0.1)
        ax.set_title("PCA (3D)")
        wandb.log({f"{tag_prefix}/pca3d": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        wandb.log({f"{tag_prefix}/pca3d_error": str(e)})

    # t-SNE 2D
    try:
        tsne = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=30)
        y = tsne.fit_transform(X)
        if color_by_sel is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(y[:, 0], y[:, 1], c=color_by_sel, s=3, alpha=0.8, cmap='viridis')
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('time proportion')
            ax.set_title("t-SNE (2D) colored by time proportion")
            wandb.log({f"{tag_prefix}/tsne2d": wandb.Image(fig)})
            plt.close(fig)
        elif labels_sel is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(y[:, 0], y[:, 1], s=3, alpha=0.6)
            ax.set_title("t-SNE (2D)")
            wandb.log({f"{tag_prefix}/tsne2d": wandb.Image(fig)})
            plt.close(fig)
        else:
            # Overview colored plot
            uniq = np.unique(labels_sel)
            n_lbl = len(uniq)
            if n_lbl <= 10:
                cmap = plt.cm.get_cmap('tab10', n_lbl)
            elif n_lbl <= 20:
                cmap = plt.cm.get_cmap('tab20', n_lbl)
            else:
                cmap = plt.cm.get_cmap('gist_ncar', n_lbl)
            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(y[:, 0], y[:, 1], c=labels_sel, s=3, alpha=0.6, cmap=cmap)
            plt.colorbar(sc, ax=ax)
            ax.set_title("t-SNE (2D) colored by cluster")
            wandb.log({f"{tag_prefix}/tsne2d": wandb.Image(fig)})
            plt.close(fig)

            # Separate plots per cluster
            for lbl in uniq:
                mask = (labels_sel == lbl)
                fig, ax = plt.subplots(figsize=(5, 5))
                # Background in light grey
                ax.scatter(y[~mask, 0], y[~mask, 1], s=2, alpha=0.1, c='lightgrey')
                # Foreground: this cluster
                ax.scatter(y[mask, 0], y[mask, 1], s=5, alpha=0.9, c='C0')
                ax.set_title(f"t-SNE cluster {int(lbl)} (n={int(mask.sum())})")
                wandb.log({f"{tag_prefix}/tsne2d_cluster_{int(lbl)}": wandb.Image(fig)})
                plt.close(fig)
    except Exception as e:
        wandb.log({f"{tag_prefix}/tsne2d_error": str(e)})

    # t-SNE 3D
    try:
        tsne3 = TSNE(n_components=3, init='random', learning_rate='auto', perplexity=30)
        y3 = tsne3.fit_transform(X)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        if color_by_sel is not None:
            sc = ax.scatter(y3[:, 0], y3[:, 1], y3[:, 2], c=color_by_sel, s=6, alpha=0.7, cmap='viridis')
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('time proportion')
            ax.set_title("t-SNE (3D) colored by time proportion")
        elif labels_sel is None:
            ax.scatter(y3[:, 0], y3[:, 1], y3[:, 2], s=6, alpha=0.7)
            ax.set_title("t-SNE (3D)")
        else:
            uniq = np.unique(labels_sel)
            n_lbl = len(uniq)
            cmap = plt.cm.get_cmap('tab20', n_lbl) if n_lbl > 10 else plt.cm.get_cmap('tab10', n_lbl)
            sc = ax.scatter(y3[:, 0], y3[:, 1], y3[:, 2], c=labels_sel, s=6, alpha=0.7, cmap=cmap)
            plt.colorbar(sc, ax=ax, pad=0.1)
            ax.set_title("t-SNE (3D) colored by cluster")
        wandb.log({f"{tag_prefix}/tsne3d": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        wandb.log({f"{tag_prefix}/tsne3d_error": str(e)})


def run_task(global_cfg: OmegaConf, task_name: str, task_cfg: OmegaConf, device: str):
    tag_prefix = f"invert/{task_name}"

    # Auto-download dataset and checkpoint if missing
    class _DL:
        pass
    dl = _DL()
    dl.env = task_name
    dl.train_dataset_path = task_cfg.train_dataset_path
    if not os.path.exists(task_cfg.train_dataset_path):
        url = get_dataset_download_url(dl)
        out_dir = os.path.dirname(task_cfg.train_dataset_path)
        os.makedirs(out_dir, exist_ok=True)
        gdown.download_folder(url=url, output=out_dir)
    dl.base_policy_path = task_cfg.base_policy_path
    if not os.path.exists(task_cfg.base_policy_path):
        url = get_checkpoint_download_url(dl)
        out_path = task_cfg.base_policy_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdown.download(url=url, output=out_path, fuzzy=True)

    # Data
    dataset = StitchedSequenceDataset(
        dataset_path=task_cfg.train_dataset_path,
        horizon_steps=int(task_cfg.horizon_steps),
        cond_steps=int(task_cfg.cond_steps),
        device=device,
    )
    # Load traj_lengths to compute exact time proportion per sample
    try:
        raw = np.load(task_cfg.train_dataset_path, allow_pickle=False)
        traj_lengths = raw["traj_lengths"].astype(np.int64)
        traj_cumsum = np.cumsum(traj_lengths)
    except Exception:
        traj_lengths = None
        traj_cumsum = None

    # Model
    # Use task-specific network overrides when provided, else fall back to global
    if hasattr(task_cfg, 'network'):
        net_cfg = dict(OmegaConf.to_container(task_cfg.network, resolve=True))
    else:
        net_cfg = dict(OmegaConf.to_container(global_cfg.model.network, resolve=True))
    net_cfg.update({
        "cond_dim": int(task_cfg.obs_dim) * int(task_cfg.cond_steps),
        "horizon_steps": int(task_cfg.horizon_steps),
        "action_dim": int(task_cfg.action_dim),
    })
    policy = DiffusionEval(
        network_path=task_cfg.base_policy_path,
        ft_denoising_steps=0,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        network=hydra.utils.instantiate(net_cfg),
        horizon_steps=int(task_cfg.horizon_steps),
        obs_dim=int(task_cfg.obs_dim),
        action_dim=int(task_cfg.action_dim),
        denoising_steps=int(global_cfg.denoising_steps),
        device=device,
        use_ddim=True,
        ddim_steps=int(global_cfg.ddim_steps),
        controllable_noise=True,
        torch_compile=False,
    )
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.final_action_clip_value = None

    # Iterate dataset and invert (sample up to max_samples for speed)
    max_samples = int(global_cfg.max_samples) if int(global_cfg.max_samples) > 0 else len(dataset)
    num_samples = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    noise_list, action_list, state_list = [], [], []
    t_in_episode_list, t_prop_list = [], []
    for idx in tqdm(indices, desc=f"Inverting {task_name}"):
        batch = dataset[idx]
        actions = batch.actions.unsqueeze(0)  # (1, Ta, Da)
        states = batch.conditions["state"]
        if states.dim() == 3 and states.shape[0] == int(task_cfg.cond_steps):
            states = states.permute(1, 0, 2).contiguous()  # (B, To, Do)
        cond = {"state": states.unsqueeze(0)}  # (1, To, Do)

        xT = compute_ddim_inverse_noise(policy, actions, cond)
        # Round-trip check for numerical sanity
        cond_rt = {"state": cond["state"], "noise_action": xT}
        x0_rt = policy.decode(cond=cond_rt, steps=None, deterministic=True)
        rt_mse = torch.mean((x0_rt - actions) ** 2).item()
        if bool(global_cfg.use_wandb):
            wandb.log({f"{tag_prefix}/roundtrip_mse": rt_mse})
        noise_list.append(xT.detach().cpu().numpy().reshape(1, -1))
        action_list.append(actions.detach().cpu().numpy().reshape(1, -1))
        state_list.append(states.detach().cpu().numpy().reshape(1, -1))
        # time within episode (steps before current start)
        try:
            start, num_before_start = dataset.indices[idx]
            t_in_episode_list.append([int(num_before_start)])
            if traj_cumsum is not None:
                ep_id = int(np.searchsorted(traj_cumsum, start, side='right'))
                ep_len = int(traj_lengths[ep_id]) if ep_id < len(traj_lengths) else int(traj_lengths[-1])
                denom = max(1, ep_len - 1)
                t_prop = float(num_before_start) / float(denom)
            else:
                t_prop = float(num_before_start) / float(num_before_start + int(task_cfg.horizon_steps) + 1e-6)
            t_prop_list.append([t_prop])
        except Exception:
            t_in_episode_list.append([0])
            t_prop_list.append([0.0])

    noises = np.concatenate(noise_list, axis=0)
    actions_arr = np.concatenate(action_list, axis=0)
    states_arr = np.concatenate(state_list, axis=0)

    # Save per task
    out_dir_task = os.path.join(global_cfg.out_dir, task_name)
    os.makedirs(out_dir_task, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir_task, "inverted_noises_v1.npz"),
        noises=noises,
        actions=actions_arr,
        states=states_arr,
        t_in_episode=np.array(t_in_episode_list).astype(np.int32),
        t_prop=np.array(t_prop_list).astype(np.float32),
        horizon_steps=int(task_cfg.horizon_steps),
        action_dim=int(task_cfg.action_dim),
        obs_dim=int(task_cfg.obs_dim),
    )

    # Optionally derive labels from state features (e.g., position bins or clusters)
    labels = None
    if int(global_cfg.kmeans_k) > 0:
        try:
            feats = states_arr[:, global_cfg.state_feature_indices] if not (isinstance(global_cfg.state_feature_indices, str) and global_cfg.state_feature_indices == "all") else states_arr
            kmeans = KMeans(n_clusters=int(global_cfg.kmeans_k), n_init=10, random_state=int(global_cfg.seed))
            labels = kmeans.fit_predict(feats)
            if bool(global_cfg.use_wandb):
                wandb.log({f"{tag_prefix}/cluster_inertia": float(kmeans.inertia_)})
        except Exception as e:
            if bool(global_cfg.use_wandb):
                wandb.log({f"{tag_prefix}/kmeans_error": str(e)})

    # Log stats per task
    if bool(global_cfg.use_wandb):
        wandb.log({
            f"{tag_prefix}/noise_abs_mean": float(np.mean(np.abs(noises))),
            f"{tag_prefix}/noise_rms": float(np.sqrt(np.mean(noises ** 2))),
        })

    # Plots per task (colored by labels or by time proportion)
    if bool(global_cfg.use_wandb):
        color_by = np.array(t_prop_list).reshape(-1) if len(t_prop_list) == noises.shape[0] else None
        log_plots_wandb(noises, tag_prefix=tag_prefix, max_points=int(global_cfg.tsne_max_points), labels=labels, color_by=color_by)


@hydra.main(config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg/robomimic"), config_name="invert_all_v1.yaml", version_base=None)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb.project, name=cfg.name, group=cfg.wandb.group, config=OmegaConf.to_container(cfg, resolve=True))

    device = cfg.device

    # If multi-task config provided and mode is 'invert', iterate tasks; otherwise support plotting-only mode
    mode = getattr(cfg, 'mode', 'invert')
    if hasattr(cfg, "tasks") and mode == 'invert':
        for task_name, task_cfg in cfg.tasks.items():
            run_task(cfg, task_name, task_cfg, device)
        if cfg.use_wandb:
            wandb.finish()
        return

    # Plot-only mode: load saved npz and regenerate plots without inversion
    if mode == 'plot':
        for task_name in cfg.tasks:
            npz_path = os.path.join(cfg.out_dir, task_name, "inverted_noises_v1.npz")
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            noises = data["noises"]
            t_prop = data["t_prop"] if "t_prop" in data.files else None
            labels = None
            # If state_feature_indices provided, cluster on states for labels
            if cfg.kmeans_k > 0 and "states" in data.files:
                states_arr = data["states"]
                feats = states_arr[:, cfg.state_feature_indices] if not (isinstance(cfg.state_feature_indices, str) and cfg.state_feature_indices == "all") else states_arr
                try:
                    kmeans = KMeans(n_clusters=int(cfg.kmeans_k), n_init=10, random_state=int(cfg.seed))
                    labels = kmeans.fit_predict(feats)
                    if cfg.use_wandb:
                        wandb.log({f"invert/{task_name}/cluster_inertia": float(kmeans.inertia_)})
                except Exception:
                    labels = None
            color_by = t_prop.reshape(-1) if t_prop is not None else None
            if cfg.use_wandb:
                log_plots_wandb(noises, tag_prefix=f"invert/{task_name}", max_points=int(cfg.tsne_max_points), labels=labels, color_by=color_by)
        if cfg.use_wandb:
            wandb.finish()
        return

    # Auto-download dataset and checkpoint if missing
    if not os.path.exists(cfg.train_dataset_path):
        url = get_dataset_download_url(cfg)
        out_dir = os.path.dirname(cfg.train_dataset_path)
        os.makedirs(out_dir, exist_ok=True)
        gdown.download_folder(url=url, output=out_dir)
    if not os.path.exists(cfg.base_policy_path):
        url = get_checkpoint_download_url(cfg)
        out_path = cfg.base_policy_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdown.download(url=url, output=out_path, fuzzy=True)

    # Data
    dataset = StitchedSequenceDataset(
        dataset_path=cfg.train_dataset_path,
        horizon_steps=cfg.horizon_steps,
        cond_steps=cfg.cond_steps,
        device=device,
    )
    # Load traj_lengths to compute exact time proportion per sample
    try:
        raw = np.load(cfg.train_dataset_path, allow_pickle=False)
        traj_lengths = raw["traj_lengths"].astype(np.int64)
        traj_cumsum = np.cumsum(traj_lengths)
    except Exception:
        traj_lengths = None
        traj_cumsum = None

    # Model
    policy = DiffusionEval(
        network_path=cfg.base_policy_path,
        ft_denoising_steps=0,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        network=hydra.utils.instantiate(cfg.model.network),
        horizon_steps=cfg.horizon_steps,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        denoising_steps=cfg.denoising_steps,
        device=device,
        use_ddim=True,
        ddim_steps=cfg.ddim_steps,
        controllable_noise=True,
        torch_compile=False,
    )
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    # Ensure no final clamp interferes with inversion consistency
    policy.final_action_clip_value = None

    # Iterate dataset and invert
    num_samples = min(len(dataset), cfg.max_samples) if cfg.max_samples > 0 else len(dataset)
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    noise_list = []
    action_list = []
    state_list = []

    t_in_episode_list = []
    t_prop_list = []
    for idx in tqdm(indices, desc="Inverting demos"):
        batch = dataset[idx]
        actions = batch.actions.unsqueeze(0)  # (1, Ta, Da)
        states = batch.conditions["state"]
        if states.dim() == 3 and states.shape[0] == cfg.cond_steps:
            states = states.permute(1, 0, 2).contiguous()  # (B, To, Do)
        cond = {"state": states.unsqueeze(0)}  # (1, To, Do)

        xT = compute_ddim_inverse_noise(policy, actions, cond)
        # Round-trip check for numerical sanity
        cond_rt = {"state": cond["state"], "noise_action": xT}
        x0_rt = policy.decode(cond=cond_rt, steps=None, deterministic=True)
        rt_mse = torch.mean((x0_rt - actions) ** 2).item()
        if cfg.use_wandb:
            wandb.log({"invert/roundtrip_mse": rt_mse})
        noise_list.append(xT.detach().cpu().numpy().reshape(1, -1))
        action_list.append(actions.detach().cpu().numpy().reshape(1, -1))
        state_list.append(states.detach().cpu().numpy().reshape(1, -1))
        # time within episode (steps before current start)
        try:
            start, num_before_start = dataset.indices[idx]
            t_in_episode_list.append([int(num_before_start)])
            if traj_cumsum is not None:
                # episode id is first cumsum > start
                ep_id = int(np.searchsorted(traj_cumsum, start, side='right'))
                ep_len = int(traj_lengths[ep_id]) if ep_id < len(traj_lengths) else int(traj_lengths[-1])
                denom = max(1, ep_len - 1)
                t_prop = float(num_before_start) / float(denom)
            else:
                # Fallback proxy (less accurate)
                t_prop = float(num_before_start) / float(num_before_start + cfg.horizon_steps + 1e-6)
            t_prop_list.append([t_prop])
        except Exception:
            t_in_episode_list.append([0])
            t_prop_list.append([0.0])

    noises = np.concatenate(noise_list, axis=0)
    actions_arr = np.concatenate(action_list, axis=0)
    states_arr = np.concatenate(state_list, axis=0)

    # Save
    os.makedirs(cfg.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(cfg.out_dir, "inverted_noises_v1.npz"),
        noises=noises,
        actions=actions_arr,
        states=states_arr,
        t_in_episode=np.array(t_in_episode_list).astype(np.int32),
        t_prop=np.array(t_prop_list).astype(np.float32),
        horizon_steps=cfg.horizon_steps,
        action_dim=cfg.action_dim,
        obs_dim=cfg.obs_dim,
    )

    # Optionally derive labels from state features (e.g., position bins or clusters)
    labels = None
    if cfg.kmeans_k > 0:
        try:
            if isinstance(cfg.state_feature_indices, str) and cfg.state_feature_indices == "all":
                feats = states_arr
            else:
                feats = states_arr[:, cfg.state_feature_indices]
            kmeans = KMeans(n_clusters=cfg.kmeans_k, n_init=10, random_state=cfg.seed)
            labels = kmeans.fit_predict(feats)
            wandb.log({"invert/cluster_inertia": float(kmeans.inertia_)}) if cfg.use_wandb else None
        except Exception as e:
            wandb.log({"invert/kmeans_error": str(e)}) if cfg.use_wandb else None

    # Log stats
    wandb.log({
        "invert/noise_abs_mean": float(np.mean(np.abs(noises))),
        "invert/noise_rms": float(np.sqrt(np.mean(noises ** 2))),
    }) if cfg.use_wandb else None

    # Plots (colored by labels if available)
    # color_by time proportion if available
    color_by = np.array(t_prop_list).reshape(-1) if len(t_prop_list) == noises.shape[0] else None
    log_plots_wandb(noises, tag_prefix="invert", max_points=cfg.tsne_max_points, labels=labels, color_by=color_by) if cfg.use_wandb else None

    # Deeper analysis for the observed two clusters
    try:
        if noises.shape[0] > 10:
            k2 = KMeans(n_clusters=2, n_init=10, random_state=cfg.seed)
            noise_labels = k2.fit_predict(noises)
            c0, c1 = k2.cluster_centers_[0], k2.cluster_centers_[1]
            cos_sim = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-8))
            sil = float(silhouette_score(noises, noise_labels))
            if cfg.use_wandb:
                wandb.log({
                    "invert/k2_inertia": float(k2.inertia_),
                    "invert/k2_cosine_centers": cos_sim,
                    "invert/k2_silhouette": sil,
                    "invert/k2_balance": float(np.mean(noise_labels)),
                })

            # Distances to assigned centers (separation strength)
            d0 = np.linalg.norm(noises - c0, axis=1)
            d1 = np.linalg.norm(noises - c1, axis=1)
            d_assigned = np.where(noise_labels == 0, d0, d1)
            d_other = np.where(noise_labels == 0, d1, d0)
            sep_ratio = d_other / (d_assigned + 1e-8)
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(sep_ratio, bins=50)
                ax.set_title("Other/Assigned center distance ratio")
                if cfg.use_wandb:
                    wandb.log({"invert/k2_sep_ratio_hist": wandb.Image(fig)})
                plt.close(fig)
            except Exception:
                pass

            # State features that differ between clusters (standardized difference)
            mu0 = states_arr[noise_labels == 0].mean(axis=0)
            mu1 = states_arr[noise_labels == 1].mean(axis=0)
            sd = states_arr.std(axis=0) + 1e-6
            zdiff = np.abs((mu1 - mu0) / sd)
            top_idx = np.argsort(-zdiff)[: min(30, len(zdiff))]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(np.arange(len(top_idx)), zdiff[top_idx])
            ax.set_xticks(np.arange(len(top_idx)))
            ax.set_xticklabels([str(i) for i in top_idx], rotation=90)
            ax.set_title("Top state features separating noise clusters (z-diff)")
            if cfg.use_wandb:
                wandb.log({"invert/k2_state_top_features": wandb.Image(fig)})
            plt.close(fig)

            # Time-in-episode distribution by cluster
            try:
                t_arr = np.array(t_in_episode_list).reshape(-1)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(t_arr[noise_labels == 0], bins=30, alpha=0.6, label="c0")
                ax.hist(t_arr[noise_labels == 1], bins=30, alpha=0.6, label="c1")
                ax.set_title("Time-in-episode by noise cluster")
                ax.legend()
                if cfg.use_wandb:
                    wandb.log({"invert/k2_time_hist": wandb.Image(fig)})
                plt.close(fig)
            except Exception:
                pass

            # PCA on states colored by noise cluster
            try:
                pca_s = PCA(n_components=2)
                ps = pca_s.fit_transform(states_arr)
                fig, ax = plt.subplots(figsize=(5, 5))
                sc = ax.scatter(ps[:, 0], ps[:, 1], c=noise_labels, s=3, alpha=0.6, cmap=plt.cm.get_cmap('tab20', 2))
                plt.colorbar(sc, ax=ax)
                ax.set_title("States PCA colored by noise cluster (K=2)")
                if cfg.use_wandb:
                    wandb.log({"invert/states_pca_k2": wandb.Image(fig)})
                plt.close(fig)
            except Exception:
                pass

            # Action magnitude differences across clusters (Ta x Da heatmap)
            a0 = actions_arr[noise_labels == 0]
            a1 = actions_arr[noise_labels == 1]
            if a0.size > 0 and a1.size > 0:
                a0m = np.mean(np.abs(a0), axis=0)
                a1m = np.mean(np.abs(a1), axis=0)
                diff = np.abs(a1m - a0m)
                try:
                    Ta, Da = cfg.horizon_steps, cfg.action_dim
                    diff_map = diff.reshape(Ta, Da)
                    fig, ax = plt.subplots(figsize=(max(6, Da), max(4, Ta)))
                    im = ax.imshow(diff_map, aspect='auto', cmap='magma')
                    fig.colorbar(im, ax=ax)
                    ax.set_title("|action| difference by noise cluster (Ta x Da)")
                    if cfg.use_wandb:
                        wandb.log({"invert/k2_action_diff": wandb.Image(fig)})
                    plt.close(fig)
                except Exception:
                    pass
    except Exception as e:
        if cfg.use_wandb:
            wandb.log({"invert/deep_analysis_error": str(e)})

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


