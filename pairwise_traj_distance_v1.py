import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import wandb


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)


def load_robomimic_npz(path: str):
    data = np.load(path, allow_pickle=False)
    states = data["states"]  # (total_steps, obs_dim)
    traj_lengths = data["traj_lengths"].astype(np.int64)  # (num_traj,)
    return states, traj_lengths


def build_traj_index_ranges(traj_lengths: np.ndarray):
    starts = np.concatenate([[0], np.cumsum(traj_lengths)[:-1]])
    ends = np.cumsum(traj_lengths)
    return starts, ends


@hydra.main(config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg/robomimic"), config_name="pairwise_traj_can_v1.yaml", version_base=None)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb.project, name=cfg.name, group=cfg.wandb.group, config=OmegaConf.to_container(cfg, resolve=True))

    states, traj_lengths = load_robomimic_npz(cfg.train_dataset_path)
    num_traj = int(len(traj_lengths))

    print(f"Loaded dataset: {cfg.train_dataset_path}")
    print(f"Total trajectories: {num_traj}")
    print(f"Total transitions: {int(traj_lengths.sum())}")
    if cfg.use_wandb:
        wandb.log({
            "pairwise/num_traj": num_traj,
            "pairwise/total_steps": int(traj_lengths.sum()),
        })

    if cfg.expected_num_traj > 0 and num_traj != int(cfg.expected_num_traj):
        print(f"WARNING: expected {cfg.expected_num_traj} trajectories, found {num_traj}")
        if cfg.use_wandb:
            wandb.log({"pairwise/expected_num_traj_mismatch": 1})

    starts, ends = build_traj_index_ranges(traj_lengths)
    traj_idx = int(cfg.traj_index)
    assert 0 <= traj_idx < num_traj, f"traj_index out of range: {traj_idx} not in [0, {num_traj-1}]"
    s, e = int(starts[traj_idx]), int(ends[traj_idx])
    traj_states = states[s:e]  # (T, Do)
    if cfg.stride > 1:
        traj_states = traj_states[:: int(cfg.stride)]
    if cfg.max_len > 0 and traj_states.shape[0] > int(cfg.max_len):
        traj_states = traj_states[: int(cfg.max_len)]

    # Compute pairwise distances
    D = pairwise_distances(traj_states, metric=cfg.distance_metric)
    print(f"Distance matrix shape: {D.shape}")

    # Plot heatmap
    os.makedirs(cfg.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Pairwise state distances (traj {traj_idx})")
    save_path = os.path.join(cfg.out_dir, f"pairwise_traj_{traj_idx}.png")
    fig.savefig(save_path, bbox_inches="tight")
    if cfg.use_wandb:
        wandb.log({"pairwise/heatmap": wandb.Image(fig)})
    plt.close(fig)

    # Basic stats
    if cfg.use_wandb:
        wandb.log({
            "pairwise/mean": float(np.mean(D)),
            "pairwise/median": float(np.median(D)),
            "pairwise/max": float(np.max(D)),
        })

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()







