from dataclasses import dataclass

import torch

from src.ppo.network import ActorCritic

@dataclass
class PPOConfig:
    #每次更新网络参数的幅度
    learning_rate: float = 3e-4
    #未来奖励的折扣系数
    gamma: float = 0.99
    #GAE优势计算的平滑程度
    gae_lambda: float = 0.95
    #限制新旧策略变化范围
    clip_range: float = 0.2
    #Critic损失在总损失中的权重
    value_coefficient: float = 0.5
    #鼓励策略继续探索
    entropy_coefficient: float = 0.01
    #防止梯度过大
    max_gradient_norm: float = 0.5
    #每次更新前收集的经验步数
    rollout_steps: int = 1024
    #每次更新前收集的经验步数
    batch_size: int = 64
    #同一批经验重复学习多少轮
    update_epochs: int = 10


class PPOAgent:
    def __init__(
            self,
            observation_dim,
            action_dim,
            config=None,
            device='cpu',
    ):

        #如果传入了config配置，就用传入的，如果没有，就用默认的。
        self.config=config or PPOConfig()

        self.device=torch.device(device)

        self.network=ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,

        ).to(self.device)


        #Adam优化器
        self.optimizer=torch.optim.Adam(
            self.network.parameters(),

            #lr是learning rate的缩写
            lr=self.config.learning_rate,
        )

    #动作的决定
    def select_action(self,observation):
        #观测转化为tensor格式
        observation_tensor=torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action_distribution,state_value=self.network(
                observation_tensor,
            )

            #按照动作的概率随机采样一个具体动作
            action=action_distribution.sample()

            #计算这次所选动作的对数概率
            log_probability=action_distribution.log_prob(
                action
            )

        return (
            action.item(),
            log_probability.item(),
            state_value.item(),
        )

    #对某个观测的状态价值预测
    def get_value(self, observation):
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            _, state_value = self.network(
                observation_tensor
            )

        return state_value.item()


    #计算优势值
    def _calculate_advantages(
        self,
        rewards,
        dones,
        values,
        last_value,
        last_done,
    ):
        #创建空的优势tensor
        advantages = torch.zeros_like(rewards)

        #三个单独的数值计算转化成零维的tensor
        last_advantage = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=self.device,
        )

        next_value = torch.tensor(
            last_value,
            dtype=torch.float32,
            device=self.device,
        )

        next_done = torch.tensor(
            float(last_done),
            dtype=torch.float32,
            device=self.device,
        )

        #倒序遍历优势值（前一步的优势值计算需要用到后一步的优势值）
        for step in reversed(range(len(rewards))):
            if step < len(rewards) - 1:
                next_value = values[step + 1]
                next_done = dones[step]

            next_non_terminal = 1.0 - next_done

            #TD误差=当前的奖励+折扣后的下一状态价值-当前的状态价值
            #如果delta>0，说明结果相比critic预测的值更好。
            delta = (
                rewards[step]
                + self.config.gamma                   #折扣系数
                * next_value                          #下一状态价值
                * next_non_terminal
                - values[step]                        #对当前价值的预测
            )

            #当前的优势等于当前的TD误差+折扣后的未来优势
            last_advantage = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_non_terminal
                * last_advantage
            )

            #存储优势值
            advantages[step] = last_advantage

        returns = advantages + values

        return advantages, returns


    def update(
            self,
            buffer,
            last_value,
            last_done,
    ):
        (
            observations,
            actions,
            rewards,
            dones,
            old_log_probabilities,
            old_values,
        )=buffer.as_tensor(self.device)

        advantages, returns = self._calculate_advantages(
            #左边为函数中的参数名字，右边为当前代码中的变量。
            rewards=rewards,
            dones=dones,
            values=old_values,
            last_value=last_value,
            last_done=last_done,
        )

        #标准化优势值，使得平均值约等于0，标准差约等于1.
        advantages = (
                             advantages - advantages.mean()
                     ) / (
                             advantages.std(unbiased=False) + 1e-8
                     )

        sample_count = len(buffer)


        #把缓冲区数据打乱并分成多个小批次，计算ppo更新需要的新动作概率、状态价值等。
        #每个训练样本包含：
        #观测
        #选择的动作
        #状态价值
        #优势值
        #目标回报
        #旧策略的动作对数概率
        #同一批缓冲区数据训练update_epochs轮。
        for _ in range(self.config.update_epochs):

            #使用randperm函数对样本进行随机排列从而实现打乱。
            indices=torch.randperm(
                sample_count,
                device=self.device,
            )

            #sample_count是样本的数量，batch_size是每个批次的数量。
            for start in range(
                    0,
                    sample_count,
                    self.config.batch_size,
            ):

                #对样本进行切片实现分组。
                minibatch_indices=indices[
                    start:start+self.config.batch_size
                ]

                #重新计算得到的动作概率分布和critic重新计算的状态价值预测。
                #distribution和advantage共同作用之后更新actor。
                #values与returns共同作用之后共同作用于critic。

                action_distribution, new_values = self.network(
                    observations[minibatch_indices]  #minibatch:小批次
                )

                #旧动作在当前网格下的对数概率
                new_log_probabilities = (
                    action_distribution.log_prob(
                        actions[minibatch_indices]
                    )
                )

                #entropy为熵：动作概率分布的随机程度。
                #计算当前Actor的动作分布的随机程度。
                entropy = action_distribution.entropy()

                #选择同一个动作在新旧策略下的的概率比（新策略/旧策略）
                probability_ratio = torch.exp(
                    new_log_probabilities
                    - old_log_probabilities[minibatch_indices]
                )

                #根据小批次对应的样本编号，取出对应的优势值。
                minibatch_advantages=advantages[
                    minibatch_indices
                ]

                #不被裁剪的目标=新旧的概率比×优势值。
                unclipped_objective = (
                        probability_ratio
                        * minibatch_advantages
                )

                #限制概率比在1-clip_range到1+clip_range之间
                clipped_ratio = torch.clamp(
                    probability_ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                )

                #使用上一个函数计算出来的裁剪后的概率比计算策略目标。
                clipped_objective = (
                        clipped_ratio
                        * minibatch_advantages
                )

                #计算策略损失。
                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()


                #每个小批次的目标回报
                minibatch_returns = returns[
                    minibatch_indices
                ]


                #价值损失：critic的预测值和目标汇报之间的平均平方误差。
                value_loss = 0.5 * (
                    new_values - minibatch_returns
                ).pow(2).mean()

                #熵奖励：用于鼓励actor的探索能力，对所有动作的熵求平均值。
                entropy_bonus = entropy.mean()

                #总损失：策略损失+价值损失-熵奖励
                total_loss = (
                    policy_loss
                    + self.config.value_coefficient
                    * value_loss
                    - self.config.entropy_coefficient
                    * entropy_bonus
                )

                #清除上一次训练保存的梯度
                self.optimizer.zero_grad()

                #计算 Actor 和 Critic 所有相关参数的梯度。
                total_loss.backward()

                #对 Actor 和 Critic 的梯度进行裁剪。
                #self.config.max_gradient_norm是允许的最大的梯度范数。
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_gradient_norm,
                )

                #更新 Actor 和 Critic 的权重和偏置。
                self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item(),
        }
