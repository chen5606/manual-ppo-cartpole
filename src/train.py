from collections import deque
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch

from src.environment import create_env
from src.ppo.agent import PPOAgent, PPOConfig
from src.ppo.buffer import RolloutBuffer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "manual_ppo_cartpole.pt"
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(total_timesteps=50_000):
    config = PPOConfig()

    set_random_seed(42)

    env = create_env()

    env.action_space.seed(42)

    observation, info = env.reset(seed=42)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        config=config,
    )

    buffer = RolloutBuffer()

    total_steps = 0
    episode_count = 0
    episode_return = 0.0

    recent_returns = deque(maxlen=20)

    while total_steps < total_timesteps:
        buffer.clear()

        rollout_size = min(
            config.rollout_steps,
            total_timesteps - total_steps,
        )

        last_done = False

        for _ in range(rollout_size):
            (
                action,
                log_probability,
                value,
            ) = agent.select_action(observation)

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            raw_reward = float(reward)
            training_reward = raw_reward

            done = terminated or truncated

            # 达到时间上限不是状态本身失败，
            # 因此补充下一状态的价值。
            if truncated and not terminated:
                training_reward += (
                    config.gamma
                    * agent.get_value(next_observation)
                )

            buffer.add(
                observation=observation,
                action=action,
                reward=training_reward,
                done=done,
                log_probability=log_probability,
                value=value,
            )

            observation = next_observation

            total_steps += 1
            episode_return += raw_reward
            last_done = done

            if done:
                episode_count += 1

                recent_returns.append(
                    episode_return
                )

                if episode_count % 10 == 0:
                    mean_return = np.mean(
                        recent_returns
                    )

                    print(
                        f"步数={total_steps:6d} | "
                        f"局数={episode_count:4d} | "
                        f"最近20局平均奖励="
                        f"{mean_return:7.2f}"
                    )

                observation, info = env.reset()

                episode_return = 0.0

        if last_done:
            last_value = 0.0
        else:
            last_value = agent.get_value(
                observation
            )

        training_info = agent.update(
            buffer=buffer,
            last_value=last_value,
            last_done=last_done,
        )

        print(
            f"PPO更新 | "
            f"总步数={total_steps:6d} | "
            f"策略损失="
            f"{training_info['policy_loss']:.4f} | "
            f"价值损失="
            f"{training_info['value_loss']:.4f} | "
            f"策略熵="
            f"{training_info['entropy']:.4f}"
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                agent.network.state_dict(),

            "optimizer_state_dict":
                agent.optimizer.state_dict(),

            "observation_dim":
                int(observation_dim),

            "action_dim":
                int(action_dim),

            "config":
                asdict(config),
        },
        MODEL_PATH,
    )

    env.close()

    print(
        f"训练完成，模型保存到：{MODEL_PATH}"
    )