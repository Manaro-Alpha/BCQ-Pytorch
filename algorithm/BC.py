import copy
import torch
import torch.nn as nn
import torch.optim as optim
# from modules.actor_critic import ActorCritic, VAE
from modules import MLP
# from replay_buffer.replay_buffer import ReplayBuffer
from replay_buffer import ReplayBuffer

class BC:
    def __init__(self,
                 mlp:MLP,
                 num_epochs,
                 num_mini_batches,
                 data,
                 replay_buffer:ReplayBuffer,
                 learning_rate=1e-5,
                 grad_clipping = 1.,
                 device='cpu'):
        
        self.device = device
        self.num_learning_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.data = data
        self.replay_buffer = replay_buffer
        self.lr = learning_rate
        self.actor = mlp
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.grad_clipping = grad_clipping

    def train_mode(self):
        self.actor.train()

    def add_data(self, data):
        self.replay_buffer.observation = data["obs"]
        self.replay_buffer.next_observation = data["next_obs"]
        self.replay_buffer.actions = data["actions"]
        self.replay_buffer.rewards = data["rewards"]
        self.replay_buffer.dones = data["dones"]

    def learn(self):
        loss_mean = 0
        self.add_data(self.data)
        obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = self.replay_buffer.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        target = actions_batch
        target.requires_grad = False
        action = self.actor.actor(obs_batch)
        loss = nn.MSELoss()(action,target)
        loss_mean = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.actor.parameters(),self.grad_clipping)
        self.optimizer.step()

        return loss_mean

