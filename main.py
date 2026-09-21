from src.play import evaluate
from src.train import train


if __name__ == "__main__":
    train(total_timesteps=50_000)
    evaluate(episodes=5)