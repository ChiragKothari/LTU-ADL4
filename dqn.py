import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hyperparams import Hyperparameters as params

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # ✅ Only the first env needs to support rendering for video
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")  # <-- CRITICAL
        else:
            env = gym.make(env_id)  # No render_mode, avoids overhead

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda episode_id: episode_id % 50 == 0
            )

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        # Seeding
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk



class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n)
        )

    def forward(self, x):
        x = x / 255.0  # Normalize input
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(params.env_id, params.seed, 0, params.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        params.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs, _ = envs.reset()
    for global_step in range(params.total_timesteps):
        epsilon = linear_schedule(params.start_e, params.end_e, params.exploration_fraction * params.total_timesteps, global_step)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                q_values = q_network(obs_tensor)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, dones, infos = envs.step(actions)

        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                break

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        obs = next_obs

        if global_step > params.learning_starts:
            if global_step % params.train_frequency == 0:
                data = rb.sample(params.batch_size)

                with torch.no_grad():
                    next_q_values = target_network(data.next_observations)
                    target_max, _ = torch.max(next_q_values, dim=1)
                    td_target = data.rewards.flatten() + (1 - data.dones.flatten()) * params.gamma * target_max

                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(old_val, td_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % params.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if params.save_model:
        model_path = f"runs/{run_name}/{params.exp_name}_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
