import numpy as np
import torch
from tqdm import tqdm

class ReplayBuffer:
    class Dataset:
        def __init__(self,device):
            ## store data, convert to torch.tensor and shift to gpu
            self.observation = None
            self.next_observation = None
            self.action = None
            self.rewards = None
            self.dones = None
            self.infos = None
            self.device = device

        def convert_2_tensor(self):
            return {"obs":torch.tensor(self.observation,device=self.device,dtype=torch.float),\
                    "next_obs": torch.tensor(self.next_observation,device=self.device,dtype=torch.float),\
                    "actions":torch.tensor(self.action, device=self.device,dtype=torch.float),\
                    "rewards":torch.tensor(self.rewards, device=self.device,dtype=torch.float),\
                    "dones":torch.tensor(self.dones, device=self.device,dtype=torch.float),\
                    }

    def __init__(self, obs_dim, action_dim, num_transitions_per_env, device, max_size):
        self.max_size = max_size
        self.num_transitions_per_env = num_transitions_per_env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.observation = torch.zeros(max_size, obs_dim, device=self.device)
        self.next_observation = torch.zeros(max_size, obs_dim, device=self.device)
        self.actions = torch.zeros(max_size, action_dim, device=self.device)
        self.rewards = torch.zeros(max_size, 1, device = self.device)
        self.returns = torch.zeros(max_size, 1, device = self.device)
        self.values = torch.zeros(max_size, 1, device=self.device)
        # self.advantages = torch.zeros(max_size, 1, device = self.device)
        self.dones = torch.zeros(max_size, 1, device=self.device)

    def sample(self,batch_size, data):
        ind = np.random.randint(low=0,high=self.max_size,size=batch_size)
        print(ind)
        # data = dataset.convert_2_tensor()
        obs = data["obs"]
        next_obs = data["next_obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]

    
        self.observation = obs[ind].to(self.device),
        self.next_observation = next_obs[ind].to(self.device),
        self.actions = actions[ind].to(self.device),
        self.rewards = rewards[ind].to(self.device),
        self.dones = dones[ind].to(self.device)
    
    def clear(self):
        self.__init__(self.obs_dim, self.action_dim,self.device,self.max_size)
    
    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step+1]
            terminal = self.dones[step].float()
            delta = self.rewards[step] + gamma*(1-terminal)*next_values - self.values[step]
            advantage = delta + (1 - terminal)*gamma*lam*advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_returns_Q(self):
        pass

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        mini_batch_size = self.max_size//num_mini_batches
        idx = torch.randint(0,self.max_size,(mini_batch_size,),device=self.device,requires_grad=False)
        obs_batch = self.observation[idx]
        next_obs_batch = self.next_observation[idx]
        actions_batch = self.actions[idx]
        rewards_batch = self.rewards[idx]
        dones_batch = self.dones[idx]

        return obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch



