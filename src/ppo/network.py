import torch
from torch import nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self,observation_dim,action_dim):
        super().__init__()
        #Actor网络
        self.actor=nn.Sequential(
            nn.Linear(observation_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,action_dim),
        )
        #Critic网络
        self.critic=nn.Sequential(
            nn.Linear(observation_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1),
        )
    def forward(self,observation):
        action_logits=self.actor(observation)
        action_distribution=Categorical(
            logits=action_logits
        )
        state_value = self.critic(observation).squeeze(-1)

        return action_distribution, state_value
