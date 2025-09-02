import os
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np

# Patch wandb to avoid the NoneType error in working_set
import wandb.util
original_working_set = wandb.util.working_set

def patched_working_set():
	try:
		for item in original_working_set():
			if item is not None:
				yield item
	except (TypeError, AttributeError, KeyError):
		# Skip packages with None metadata or other issues
		pass

wandb.util.working_set = patched_working_set

import hydra
from omegaconf import OmegaConf
import gym, d4rl
import d4rl.gym_mujoco
import sys
sys.path.append('./dppo')
 
from stable_baselines3 import VGD
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils_vgd import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
	config_path=None, config_name=None, version_base=None
)
def main(cfg: OmegaConf):
	OmegaConf.resolve(cfg)

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	if cfg.use_wandb:
		wandb.init(
			project=cfg.wandb.project,
			name=cfg.name,
			group=cfg.wandb.group,
			monitor_gym=True,
			save_code=True,
			config=OmegaConf.to_container(cfg, resolve=True),
		)

	MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

	num_env = cfg.env.n_envs
	def make_env():
		if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
			# Lazy import d4rl only for gym mujoco tasks to avoid mujoco_py issues otherwise
			try:
				import d4rl  # noqa: F401
				import d4rl.gym_mujoco  # noqa: F401
			except Exception as e:
				print(f"Warning: d4rl import failed ({e}). Gym mujoco tasks may not run.")
			env = gym.make(cfg.env_name)
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
			env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
		env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
		return env

	base_policy = load_base_policy(cfg)
	env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
	env.seed(cfg.seed + 1)
	post_linear_modules = None
	if cfg.train.use_layer_norm:
		post_linear_modules = [torch.nn.LayerNorm]

	net_arch = []
	for _ in range(cfg.train.num_layers):
		net_arch.append(cfg.train.layer_size)
	policy_kwargs = dict(
		net_arch=dict(pi=net_arch, qf=net_arch),
		activation_fn=torch.nn.Tanh,
		log_std_init=0.0,
		post_linear_modules=post_linear_modules,
		n_critics=cfg.train.n_critics,
	)

	model = VGD(
		"MlpPolicy",
		env,
		learning_rate=cfg.train.actor_lr,
		buffer_size=10000000,
		learning_starts=1,
		batch_size=cfg.train.batch_size,
		tau=cfg.train.tau,
		gamma=cfg.train.discount,
		train_freq=cfg.train.train_freq,
		gradient_steps=cfg.train.utd,
		action_noise=None,
		optimize_memory_usage=False,
		tensorboard_log=cfg.logdir,
		verbose=1,
		policy_kwargs=policy_kwargs,
		diffusion_policy=base_policy,
		diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
		critic_backup_combine_type=cfg.train.critic_backup_combine_type,
		guidance_lambda=getattr(cfg.train, 'guidance_lambda', 0.0),
		guidance_warmup_steps=getattr(cfg.train, 'guidance_warmup_steps', 0),
		guidance_time_anneal=getattr(cfg.train, 'guidance_time_anneal', 'linear'),
		guidance_last_k_steps=getattr(cfg.train, 'guidance_last_k_steps', 0),
		guidance_decode_steps=getattr(cfg.train, 'guidance_decode_steps', -1),
	)

	checkpoint_callback = CheckpointCallback(
		save_freq=cfg.save_model_interval, 
		save_path=cfg.logdir+'/checkpoint/',
		name_prefix='ft_policy',
		save_replay_buffer=cfg.save_replay_buffer, 
		save_vecnormalize=True,
	)

	num_env_eval = cfg.env.n_eval_envs
	eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
	eval_env.seed(cfg.seed + num_env + 1) 

	logging_callback = LoggingCallback(
		action_chunk = cfg.act_steps, 
		eval_episodes = int(cfg.num_evals / num_env_eval), 
		log_freq=MAX_STEPS, 
		use_wandb=cfg.use_wandb, 
		eval_env=eval_env, 
		eval_freq=cfg.eval_interval,
		num_train_env=num_env,
		num_eval_env=num_env_eval,
		rew_offset=cfg.env.reward_offset,
		max_steps=MAX_STEPS,
		deterministic_eval=cfg.deterministic_eval,
		record_noise=getattr(cfg, 'record_eval_noise', True),
		noise_out_dir=os.path.join(cfg.logdir, 'artifacts_vgd_noise'),
		stop_after_episodes=getattr(cfg, 'stop_after_episodes', -1),
	)

	# Evaluate and optionally record latents for analysis
	logging_callback.evaluate(model, deterministic=False)
	if cfg.deterministic_eval:
		logging_callback.evaluate(model, deterministic=True)

	# Optional: dump actor latents during evaluation episodes
	if getattr(cfg, 'record_eval_noise', True):
		try:
			out_dir = os.path.join(cfg.logdir, 'artifacts_vgd_noise')
			os.makedirs(out_dir, exist_ok=True)
			# Use single-env eval for recording
			rec_env = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)
			rec_env.seed(cfg.seed + num_env + 999)

			all_noises, all_states, t_in_episode = [], [], []
			max_episodes = int(getattr(cfg, 'record_eval_episodes', 5))
			for ep in range(max_episodes):
				obs = rec_env.reset()
				done = np.array([False])
				step_in_ep = 0
				while not done[0] and step_in_ep < cfg.env.max_episode_steps:
					try:
						noise_latent = model.predict_noise_latent(obs, deterministic=True)
					except Exception:
						noise_latent = np.zeros((1, cfg.act_steps, cfg.action_dim), dtype=np.float32)

					all_noises.append(noise_latent.reshape(1, -1))
					all_states.append(obs.reshape(1, -1))
					t_in_episode.append([step_in_ep])

					action, _ = model.predict_diffused(obs, deterministic=True)
					obs, reward, done, info = rec_env.step(action)
					step_in_ep += cfg.act_steps

			if len(all_noises) > 0:
				noises = np.concatenate(all_noises, axis=0)
				states = np.concatenate(all_states, axis=0)
				t_in_ep = np.array(t_in_episode).astype(np.int32)
				npz_path = os.path.join(out_dir, 'vgd_eval_noises.npz')
				np.savez_compressed(npz_path,
								noises=noises,
								states=states,
								t_in_episode=t_in_ep,
								act_steps=cfg.act_steps,
								action_dim=cfg.action_dim,
								obs_dim=cfg.obs_dim)
		except Exception as e:
			print(f"Noise recording skipped due to error: {e}")
	logging_callback.log_count += 1

	if cfg.load_offline_data:
		load_offline_data(model, cfg.offline_data_path, num_env)
	if cfg.train.init_rollout_steps > 0:
		collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)	
		logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)

	callbacks = [checkpoint_callback, logging_callback]
	# Train the agent
	model.learn(
		total_timesteps=20000000,
		callback = callbacks
	)

	# Save the final model
	if len(cfg.name) > 0:
		model.save(cfg.logdir+"/checkpoint/final")

	# Close environment and wandb
	env.close()
	if cfg.use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()


