import torch

from src.environment import create_env
from src.ppo.network import ActorCritic
from src.train import MODEL_PATH


def evaluate(episodes=5):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    network = ActorCritic(
        observation_dim=checkpoint["observation_dim"],
        action_dim=checkpoint["action_dim"],
    )

    network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    network.eval()

    env = create_env(render_mode="human")

    for episode in range(episodes):
        observation, info = env.reset()
        episode_return = 0.0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                action_distribution, _ = network(
                    observation_tensor
                )

                # 播放时直接选择概率最大的动作
                action = torch.argmax(
                    action_distribution.probs,
                    dim=-1,
                ).item()

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_return += float(reward)

        print(
            f"第{episode + 1}局，"
            f"总奖励：{episode_return}"
        )

    env.close()


if __name__ == "__main__":
    evaluate(episodes=5)