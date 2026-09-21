from pathlib import Path

from stable_baselines3 import PPO

from src.environment import create_env


MODEL_PATH = Path("models/ppo_cartpole")


def train(total_timesteps=50_000):
    """训练PPO智能体并保存模型。"""

    Path("models").mkdir(exist_ok=True)

    # 训练时不显示画面，提高训练速度
    env = create_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=42,
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(MODEL_PATH)

    env.close()

    print(f"训练完成，模型已保存到：{MODEL_PATH}.zip")

