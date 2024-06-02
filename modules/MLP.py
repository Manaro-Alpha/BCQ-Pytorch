import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()

        self.in_dim = num_obs
        self.out_dim = num_actions
        self.actor = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ELU(),
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU(),
            nn.Linear(128,self.out_dim)
        )

    def forward(self,obs):
        action = self.actor(obs)
