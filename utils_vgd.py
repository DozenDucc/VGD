import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
import os


class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		max_steps=-1,
		deterministic_eval=False,
		record_noise=False,
		noise_out_dir=None,
		stop_after_episodes=-1,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval
		self.record_noise = record_noise
		self.noise_out_dir = noise_out_dir
		self.eval_session_idx = 0
		self.stop_after_episodes = stop_after_episodes
		self.total_episodes = 0

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		# accumulate finished episodes across all vectorized environments
		self.total_episodes += int(np.sum(self.locals['dones']))
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					log_vals = {
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / max(1, np.sum(self.episode_completed)),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/episodes": self.total_episodes,
					}
					# add any recorded scalar from SB3 logger if present
					name_to_val = getattr(self.locals['self'].logger, 'name_to_value', {})
					wandb.log(log_vals, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		# early stop when reaching target total episodes across all envs
		if self.stop_after_episodes > 0 and self.total_episodes >= self.stop_after_episodes:
			if self.use_wandb and self.log_count >= 0:
				wandb.log({"train/episodes": self.total_episodes}, step=self.log_count)
			return False
		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				# Per-env counters used for both logging and optional noise recording
				step_counters = np.zeros(self.num_eval_env, dtype=int)
				ep_counters = np.zeros(self.num_eval_env, dtype=int)
				# Recording buffers per eval session (only if recording is enabled)
				if self.record_noise and self.noise_out_dir is not None:
					all_noises, all_states, t_in_episode_list, eval_idx_list, episode_idx_list = [], [], [], [], []
				for i in range(self.eval_episodes):
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						# Use guided policy
						action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						# Optionally record latents
						if self.record_noise and self.noise_out_dir is not None and hasattr(agent, 'predict_noise_latent'):
							try:
								noise_latent = agent.predict_noise_latent(obs, deterministic=deterministic)
							except Exception:
								flat_dim = int(env.action_space.shape[-1])
								a_dim = int(max(1, flat_dim // int(self.action_chunk)))
								noise_latent = np.zeros((obs.shape[0], self.action_chunk, a_dim), dtype=np.float32)
							all_noises.append(noise_latent.reshape(obs.shape[0], -1))
							all_states.append(obs.reshape(obs.shape[0], -1))
							# record current step counters and eval/episode ids per env
							for env_idx in range(obs.shape[0]):
								t_in_episode_list.append([int(step_counters[env_idx])])
								eval_idx_list.append([int(self.eval_session_idx)])
								episode_idx_list.append([int(ep_counters[env_idx] + i)])
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset] = 1
						r.append(reward)
						# update per-env step and episode counters
						step_counters = step_counters + self.action_chunk
						if np.any(done):
							for env_idx in range(len(done)):
								if done[env_idx]:
									step_counters[env_idx] = 0
									ep_counters[env_idx] += 1
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)
				# Save per-eval-session noise dump
				if self.record_noise and self.noise_out_dir is not None and len(all_noises) > 0:
					noises = np.concatenate(all_noises, axis=0)
					states = np.concatenate(all_states, axis=0)
					t_in_episode = np.array(t_in_episode_list).astype(np.int32)
					eval_idx = np.array(eval_idx_list).astype(np.int32)
					episode_idx = np.array(episode_idx_list).astype(np.int32)
					os.makedirs(self.noise_out_dir, exist_ok=True)
					fname = f"vgd_eval_noises_eval{self.eval_session_idx}_det{int(deterministic)}.npz"
					np.savez_compressed(os.path.join(self.noise_out_dir, fname),
						noises=noises,
						states=states,
						t_in_episode=t_in_episode,
						eval_session=eval_idx,
						episode_idx=episode_idx,
						action_chunk=self.action_chunk,
						obs_dim=states.shape[1],
					)
				self.eval_session_idx += 1

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps


def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in range(num_steps):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		action_store = action
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	

def load_offline_data(model, offline_data_path, n_env):
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()


