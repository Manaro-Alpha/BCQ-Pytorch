import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, clip_action, std=0.05):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ELU()
        self.activation3 = nn.Tanh()
        actor_in_dim = obs_dim + action_dim
        actor_out_dim = action_dim
        self.phi = std
        self.clip_action = clip_action
        
        self.perturb_net = nn.Sequential(
            nn.Linear(actor_in_dim, 400),
            self.activation1,
            nn.Linear(400,300),
            nn.Linear(300,actor_out_dim),
            self.activation3
        )

    def forward(self, obs, action):
        obs = torch.cat((obs,action),dim=-1)
        p = self.perturb_net(obs)
        return (self.phi*p*self.clip_action + action).clamp(-self.clip_action,self.clip_action)
    

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, std=0.05):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ELU()
        self.activation3 = nn.Tanh()
        critic_in_dim = obs_dim + action_dim

        self.critic1 = nn.Sequential(
        nn.Linear(critic_in_dim, 400),
        self.activation1,
        nn.Linear(400,300),
        nn.Linear(300,1)
    )
        
        self.critic2 = nn.Sequential(
        nn.Linear(critic_in_dim, 400),
        self.activation1,
        nn.Linear(400,300),
        nn.Linear(300,1)
    )
    

    def critic1_forward(self,obs,action):
        obs = torch.cat((obs,action),dim=-1)
        q1 = self.critic1(obs)
        return q1
    
    def critic2_forward(self,obs,action):
        obs = torch.cat((obs,action),dim=-1)
        q2 = self.critic2(obs)
        return q2
    
    def get_q(self,obs,action):
        obs = torch.cat((obs,action),dim=-1)
        q = self.critic1(obs)
        return q
        
    

class VAE(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim, clip_action, device):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ELU()
        self.activation3 = nn.Tanh()
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim+action_dim,750),
            self.activation1,
            nn.Linear(750,750),
            self.activation1
        )

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(obs_dim+latent_dim,750),
            self.activation1,
            nn.Linear(750,750),
            self.activation1,
            nn.Linear(750,action_dim),
            self.activation3
        )

        self.clip_action = clip_action

    def reparametrise(self,mean,log_std):
        std = torch.exp(log_std)
        z = mean + std*torch.rand_like(std)
        return z
    
    def forward(self,obs,action):
        enc_in = torch.cat((obs,action),dim=-1).float()
        z = self.encoder(enc_in)
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        z = self.reparametrise(mean,log_std)

        decode = self.decode(obs,z)

        return z,decode,mean,log_std
    
    def decode(self,obs,z=None):
        if z is None:
            z = torch.randn(obs.shape[0],self.latent_dim,device=self.device).clamp(-0.5, 0.5)
        
        a = self.decoder(torch.cat((obs,z),dim=-1))

        action = self.clip_action*a

        return action 


