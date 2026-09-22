#经验缓冲区

import numpy as np
import torch

class RolloutBuffer:
    def __init__(self):
        self.clear()
    def clear(self):
        self.observations=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]
        self.log_probabilities=[]
        self.values=[]

    def add(self,
            observation,
            action,
            reward,
            done,
            log_probability
            ,value):
        self.observations.append(
            np.asarray(observation,dtype=np.float32)
        )
        #actor选择的动作保存
        self.actions.append(action)
        #奖励保存
        self.rewards.append(reward)
        #这一步后回合是否结束
        self.dones.append(done)
        #策略选择该动作的对数概率
        self.log_probabilities.append(log_probability)
        #critic对状态价值的估计
        self.values.append(value)

    #返回当前保存的经验数量
    def __len__(self):

        return len(self.rewards)

    #把经验转换成PyTorch tensor张量

    def as_tensor(self,device='cpu'):

        #环境观测
        observations = torch.as_tensor(
            np.asarray(self.observations),
            dtype=torch.float32,
            device=device,
        )

        #根据观测选择的动作
        actions = torch.as_tensor(
            self.actions,
            dtype=torch.long,
            device=device,
        )

        #执行动作后返回的奖励
        rewards = torch.as_tensor(
            self.rewards,
            dtype=torch.float32,
            device=device,
        )

        #执行动作后当前回合是否结束
        dones = torch.as_tensor(
            self.dones,
            dtype=torch.float32,
            device=device,
        )

        #旧策略选择该动作的对数概率
        old_log_probabilities=torch.as_tensor(
            self.log_probabilities,
            dtype=torch.float32,
            device=device,
        )
        #旧critic对当前状态的估计
        old_values=torch.as_tensor(
            self.values,
            dtype=torch.float32,
            device=device,
        )
        return (
            observations,
            actions,
            rewards,
            dones,
            old_log_probabilities,
            old_values,
        )



