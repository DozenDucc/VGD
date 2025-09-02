import os
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import hydra
from omegaconf import OmegaConf
import wandb

import sys
sys.path.append('./dppo')  # to resolve intra-dppo imports like `from model.*` used by DPPO modules

# Safer defaults for headless MuJoCo / OpenGL to avoid GL context crashes
import os as _os
_os.environ.setdefault("MUJOCO_GL", "egl")
_os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dppo.agent.dataset.sequence import StitchedSequenceDataset
from dppo.model.diffusion.diffusion_eval import DiffusionEval
from dppo.model.common.steering_head import SteeringHeadMLP
from dppo.script.download_url import (
    get_dataset_download_url,
    get_checkpoint_download_url,
)
import gdown
from dppo.script.download_url import get_normalization_download_url


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


def build_adamw_weight_decay_only_on_weights(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create AdamW that applies weight decay to weights only (no decay on biases or norm params).

    This follows common practice: disable decay for bias and normalization parameters to avoid harming optimization.
    """
    decay_params = []
    no_decay_params = []

    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, norm_types):
                no_decay_params.append(param)
            elif param_name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # De-duplicate and ensure groups are disjoint
    decay_unique = {id(p): p for p in decay_params}
    no_decay_unique = {id(p): p for p in no_decay_params}
    for pid in list(decay_unique.keys()):
        if pid in no_decay_unique:
            decay_unique.pop(pid)

    param_groups = [
        {"params": list(decay_unique.values()), "weight_decay": float(weight_decay)},
        {"params": list(no_decay_unique.values()), "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=float(lr))

@hydra.main(config_path=os.path.join(base_path, "cfg/robomimic"), config_name="steer_can.yaml", version_base=None)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # Repro
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    device = cfg.device

    # Auto-download dataset if missing
    if not os.path.exists(cfg.train_dataset_path):
        url = get_dataset_download_url(cfg)
        out_dir = os.path.dirname(cfg.train_dataset_path)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Downloading dataset from {url} to {out_dir}")
        gdown.download_folder(url=url, output=out_dir)

    # Auto-download base policy if missing
    if not os.path.exists(cfg.base_policy_path):
        url = get_checkpoint_download_url(cfg)
        if url is None:
            raise ValueError("Unknown checkpoint path; please set base_policy_path to a known pretrained checkpoint.")
        out_path = cfg.base_policy_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"Downloading base policy from {url} to {out_path}")
        gdown.download(url=url, output=out_path, fuzzy=True)

    # Dataset: states and actions already normalized if using preprocessed files
    dataset = StitchedSequenceDataset(
        dataset_path=cfg.train_dataset_path,
        horizon_steps=cfg.horizon_steps,
        cond_steps=cfg.cond_steps,
        device=device,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # Frozen diffusion policy (deterministic DDIM unrolled)
    policy = DiffusionEval(
        network_path=cfg.base_policy_path,
        ft_denoising_steps=0,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        final_action_clip_value=getattr(cfg, "final_action_clip_value", 1.0),
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

    # Steering head
    head = SteeringHeadMLP(
        obs_dim=cfg.obs_dim,
        cond_steps=cfg.cond_steps,
        action_dim=cfg.action_dim,
        horizon_steps=cfg.horizon_steps,
        hidden_dims=cfg.head.hidden_dims,
    ).to(device)

    optimizer = build_adamw_weight_decay_only_on_weights(head, lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            actions = batch.actions  # (B, Ta, Da)
            state_hist = batch.conditions["state"]  # (To, B, Do)? -> per dataset it's (To, B, Do) stacked last
            # sequence dataset returns conditions stacked as (To, B, Do) with To=cond_steps, but code builds as list then stack -> shape (cond_steps, B, Do)
            # However, in this repo, cond is stored as (To, B, Do) then at use they expect (B, To, Do). Ensure transpose
            if state_hist.dim() == 3 and state_hist.shape[0] == cfg.cond_steps:
                state_hist = state_hist.permute(1, 0, 2).contiguous()

            # Steering head outputs initial latent
            w = head(state_hist)
            # Optional tanh shaping of noise into a bounded range
            if getattr(cfg, "use_tanh_noise", False):
                if getattr(cfg, "tanh_range", None) is not None:
                    tanh_min, tanh_max = float(cfg.tanh_range[0]), float(cfg.tanh_range[1])
                    half = 0.5 * (tanh_max - tanh_min)
                    mid = 0.5 * (tanh_max + tanh_min)
                    w = torch.tanh(w) * half + mid
                else:
                    scale = float(getattr(cfg, "tanh_scale", 2.0))
                    w = torch.tanh(w) * scale
            # Optional hard clipping of noise
            if getattr(cfg, "noise_clip", None) is not None:
                w = w.clamp(-cfg.noise_clip, cfg.noise_clip)

            # Decode deterministically through frozen policy with gradients
            cond = {"state": state_hist, "noise_action": w}
            # a_pred = policy.decode(cond=cond, steps=cfg.ddim_steps, deterministic=True)
            # Use the model's configured schedule (DDIM/DDPM) without overriding step count
            a_pred = policy.decode(cond=cond, steps=None, deterministic=True)

            # Losses
            recon = F.mse_loss(a_pred, actions)
            noise_reg = (w.pow(2).mean())
            if getattr(cfg, "use_noise_reg", True):
                loss = recon + cfg.lambda_w * noise_reg
            else:
                loss = recon

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(head.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            if cfg.use_wandb and global_step % cfg.log_interval == 0:
                # Additional diagnostics
                noise_abs_mean = w.detach().abs().mean().item()
                noise_rms = w.detach().pow(2).mean().sqrt().item()
                a_pred_abs_mean = a_pred.detach().abs().mean().item()
                a_pred_rms = a_pred.detach().pow(2).mean().sqrt().item()
                actions_abs_mean = actions.detach().abs().mean().item()
                actions_rms = actions.detach().pow(2).mean().sqrt().item()
                wandb.log({
                    "train/loss": loss.item(),
                    "train/recon": recon.item(),
                    "train/noise_reg": noise_reg.item(),
                    "train/grad_norm": float(sum(p.grad.detach().data.norm(2).item() for p in head.parameters() if p.grad is not None)),
                    "train/noise_abs_mean": noise_abs_mean,
                    "train/noise_rms": noise_rms,
                    "train/a_pred_abs_mean": a_pred_abs_mean,
                    "train/a_pred_rms": a_pred_rms,
                    "train/actions_abs_mean": actions_abs_mean,
                    "train/actions_rms": actions_rms,
                }, step=global_step)

            global_step += 1

        # Evaluation
        if (epoch + 1) % cfg.eval_every == 0:
            eval_results = {}
            for env_name in cfg.eval_envs:
                # Ensure normalization exists per env
                if "${env}" in cfg.normalization_path:
                    norm_path = cfg.normalization_path.replace("${env}", env_name)
                else:
                    norm_path = cfg.normalization_path
                # Prefer dataset-adjacent normalization for the training env to avoid stat mismatch
                ds_norm_path = os.path.join(os.path.dirname(cfg.train_dataset_path), "normalization.npz")
                if env_name == cfg.env and os.path.exists(ds_norm_path):
                    norm_path = ds_norm_path
                if not os.path.exists(norm_path):
                    class _N: pass
                    tmp = _N()
                    tmp.env_name = env_name
                    tmp.normalization_path = norm_path
                    url = get_normalization_download_url(tmp)
                    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
                    gdown.download(url=url, output=norm_path, fuzzy=True)
                else:
                    # Sanity-check normalization dimensions
                    try:
                        import numpy as _np
                        _norm = _np.load(norm_path)
                        if _norm["obs_min"].shape[0] != cfg.obs_dim:
                            if cfg.use_wandb:
                                wandb.log({f"eval/{env_name}/obs_norm_dim_mismatch": 1}, step=global_step)
                            if getattr(cfg, "skip_eval_on_error", True):
                                continue
                    except Exception:
                        pass
                # Build env (skip on error if requested)
                # Build env with isolation to mitigate segfaults from repeated Mujoco initializations
                try:
                    import multiprocessing as mp

                    def _make_env(return_dict):
                        from env_utils import make_robomimic_env, ObservationWrapperRobomimic
                        env_i = make_robomimic_env(
                            env=env_name,
                            normalization_path=norm_path,
                            low_dim_keys=cfg.env_wrappers.robomimic_lowdim.low_dim_keys,
                            dppo_path=cfg.dppo_path,
                        )
                        env_i = ObservationWrapperRobomimic(env_i, reward_offset=cfg.reward_offset)
                        return_dict["ok"] = True

                    manager = mp.Manager()
                    rd = manager.dict()
                    p = mp.Process(target=_make_env, args=(rd,))
                    p.start()
                    p.join()
                    if not rd.get("ok", False):
                        raise RuntimeError("Failed to create env in subprocess")

                    # If subprocess creation succeeded, create in main proc once
                    from env_utils import make_robomimic_env, ObservationWrapperRobomimic, ActionChunkWrapper
                    env = make_robomimic_env(
                        env=env_name,
                        normalization_path=norm_path,
                        low_dim_keys=cfg.env_wrappers.robomimic_low_dim.low_dim_keys if hasattr(cfg.env_wrappers, 'robomimic_low_dim') else cfg.env_wrappers.robomimic_lowdim.low_dim_keys,
                        dppo_path=cfg.dppo_path,
                    )
                    env = ObservationWrapperRobomimic(env, reward_offset=cfg.reward_offset)
                except Exception as e:
                    if cfg.use_wandb:
                        wandb.log({f"eval/{env_name}/error": 1}, step=global_step)
                    if getattr(cfg, "skip_eval_on_error", True):
                        print(f"Skipping eval for {env_name} due to error: {e}")
                        continue
                    raise
                # Provide minimal cfg for ActionChunkWrapper
                class _C: pass
                c = _C()
                c.act_steps = cfg.horizon_steps
                c.obs_dim = cfg.obs_dim
                c.action_dim = cfg.action_dim
                env = ActionChunkWrapper(env, c, max_episode_steps=cfg.max_episode_steps)

                rewards = []
                successes = []
                noise_abs_means = []
                a_pred_abs_means = []
                head.eval()
                with torch.no_grad():
                    for _ in range(cfg.eval_episodes):
                        obs, _ = env.reset()
                        # Initialize history with repeated current obs
                        hist = [obs for _ in range(cfg.cond_steps)]
                        ep_reward = 0.0
                        ever_success = False
                        # Each step issues an action chunk of size horizon_steps
                        for _ in range(cfg.max_episode_steps // cfg.horizon_steps):
                            state_hist = torch.tensor(
                                np.stack(hist[-cfg.cond_steps:]), dtype=torch.float32, device=device
                            ).unsqueeze(0)  # (1, To, Do)
                            w = head(state_hist)
                            if getattr(cfg, "use_tanh_noise", False):
                                if getattr(cfg, "tanh_range", None) is not None:
                                    tanh_min, tanh_max = float(cfg.tanh_range[0]), float(cfg.tanh_range[1])
                                    half = 0.5 * (tanh_max - tanh_min)
                                    mid = 0.5 * (tanh_max + tanh_min)
                                    w = torch.tanh(w) * half + mid
                                else:
                                    scale = float(getattr(cfg, "tanh_scale", 2.0))
                                    w = torch.tanh(w) * scale
                            if getattr(cfg, "noise_clip", None) is not None:
                                w = w.clamp(-cfg.noise_clip, cfg.noise_clip)
                            cond = {"state": state_hist, "noise_action": w}
                            a_pred = policy.decode(cond=cond, steps=None, deterministic=True)
                            # Per-step diagnostics
                            noise_abs_means.append(w.abs().mean().item())
                            a_pred_abs_means.append(a_pred.abs().mean().item())
                            action_np = a_pred.squeeze(0).cpu().numpy()
                            next_obs, reward, done, _, info = env.step(action_np)
                            ep_reward += reward
                            ever_success = ever_success or (reward > -cfg.reward_offset)
                            obs = next_obs
                            hist.append(obs)
                            if done:
                                break
                        rewards.append(ep_reward)
                        successes.append(1.0 if ever_success else 0.0)

                avg_rew = float(np.mean(rewards)) if len(rewards) else 0.0
                succ_rate = float(np.mean(successes)) if len(successes) else 0.0
                avg_noise_abs = float(np.mean(noise_abs_means)) if len(noise_abs_means) else 0.0
                avg_a_pred_abs = float(np.mean(a_pred_abs_means)) if len(a_pred_abs_means) else 0.0
                eval_results[env_name] = {"reward": avg_rew, "success": succ_rate, "noise_abs_mean": avg_noise_abs, "a_pred_abs_mean": avg_a_pred_abs}
                if cfg.use_wandb:
                    wandb.log({
                        f"eval/{env_name}/reward": avg_rew,
                        f"eval/{env_name}/success_rate": succ_rate,
                        f"eval/{env_name}/noise_abs_mean": avg_noise_abs,
                        f"eval/{env_name}/a_pred_abs_mean": avg_a_pred_abs,
                    }, step=global_step)

                # Explicitly close and delete env to avoid leaking GL contexts
                try:
                    if hasattr(env, 'close'):
                        env.close()
                except Exception:
                    pass
                del env
                try:
                    import gc as _gc
                    _gc.collect()
                except Exception:
                    pass

            if len(eval_results) > 0 and cfg.use_wandb:
                wandb.log({
                    "eval/reward": float(np.mean([v["reward"] for v in eval_results.values()])),
                    "eval/success_rate": float(np.mean([v["success"] for v in eval_results.values()])),
                    "eval/noise_abs_mean": float(np.mean([v["noise_abs_mean"] for v in eval_results.values()])),
                    "eval/a_pred_abs_mean": float(np.mean([v["a_pred_abs_mean"] for v in eval_results.values()])),
                }, step=global_step)

        # Save checkpoint per epoch
        if (epoch + 1) % cfg.save_every == 0:
            os.makedirs(cfg.logdir, exist_ok=True)
            save_path = os.path.join(cfg.logdir, f"steering_head_epoch_{epoch+1}.pt")
            torch.save({
                "model": head.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }, save_path)

    # Final save
    os.makedirs(cfg.logdir, exist_ok=True)
    final_path = os.path.join(cfg.logdir, "steering_head_final.pt")
    torch.save({"model": head.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, final_path)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


