from src.environment import create_env

env = create_env(render_mode="human")

observation, info = env.reset(seed=42)
total_reward = 0

for step in range(200):
    # 暂时随机选择动作，还没有使用强化学习算法
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

env.close()

print(f"运行步数：{step + 1}")
print(f"总奖励：{total_reward}")