from stable_baselines3 import PPO

from src.environment import create_env
from src.train import MODEL_PATH


def evaluate(episodes=5):
    """加载训练好的模型并显示运行效果。"""

    env = create_env(render_mode="human")
    model = PPO.load(MODEL_PATH)

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, state = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

        print(
            f"第 {episode + 1} 局，"
            f"总奖励：{total_reward}"
        )

    env.close()