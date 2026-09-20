import gymnasium as gym


def create_env(render_mode=None):
    """创建 CartPole 强化学习环境。"""
    return gym.make("CartPole-v1", render_mode=render_mode)