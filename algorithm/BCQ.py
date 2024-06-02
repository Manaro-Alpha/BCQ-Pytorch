import copy
import torch
import torch.nn as nn
import torch.optim as optim
# from modules.actor_critic import ActorCritic, VAE
from modules import Actor,Critic,VAE
# from replay_buffer.replay_buffer import ReplayBuffer
from replay_buffer import ReplayBuffer


class BCQ:
    def __init__(self,
                 actor:Actor,
                 critic:Critic,
                 vae:VAE,
                 num_leanring_epochs,
                 num_mini_batches,
                 data,
                 replay_buffer:ReplayBuffer,
                 tau,
                 gamma=0.99,
                 lam=0.75,
                 learning_rate=1e-3,
                 grad_clipping = 1.,
                 device='cpu'):
        
        
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.lr = learning_rate
        self.lr_2 = learning_rate
        self.num_learning_epochs = num_leanring_epochs
        self.num_mini_batches = num_mini_batches
        self.grad_clipping = grad_clipping
        self.data = data
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        # self.actor = self.actor_critic.perturb_net()
        self.vae = vae
        self.tau = tau

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(),lr=self.lr)
        self.optimizer_vae = optim.Adam(self.vae.parameters(),lr=self.lr)

        # self.data = ReplayBuffer.Dataset(device=self.device)
        # self.data, _ = self.data.convert_2_tensor()
        self.replay_buffer = replay_buffer

    # def init_storage(self, obs_dim, action_dim, num_transitions_per_env, device, max_size):
    #     self.replay_buffer = ReplayBuffer(obs_dim, action_dim, num_transitions_per_env, device, max_size)

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        self.vae.train()

    def add_data(self, data):
        self.replay_buffer.observation = data["obs"]
        self.replay_buffer.next_observation = data["next_obs"]
        self.replay_buffer.actions = data["actions"]
        self.replay_buffer.rewards = data["rewards"]
        self.replay_buffer.dones = data["dones"]
        # self.data.infos = dataset["infos"]
        # self.data_tensor = self.data.convert_2_tensor()
        # return self.data_tensor

    def learn(self):
        vae_loss_mean = 0
        critic_loss_mean = 0
        actor_loss_mean = 0
        self.add_data(self.data)
        # self.replay_buffer.sample(batch_size=100,data=self.data)
        
        obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = self.replay_buffer.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # for obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch in generator:
        mini_batch_size = obs_batch.shape[0]

        z,action,vae_mean,vae_logstd = self.vae.forward(obs_batch,actions_batch)
        vae_loss = nn.MSELoss()(action,actions_batch) + 0.5*(-0.5 * (1 + 2*vae_logstd - vae_mean.pow(2) - vae_logstd.exp().pow(2)).mean())
        vae_loss_mean += vae_loss.item()

        self.optimizer_vae.zero_grad()
        vae_loss.backward()
        self.optimizer_vae.step()

        # critic

        with torch.no_grad():
            next_obs_ = torch.repeat_interleave(next_obs_batch,10,0)
            # print(next_obs_.shape) ##[99990,11]
            target_q1 = self.critic_target.critic1_forward(next_obs_, self.actor_target(next_obs_,self.vae.decode(next_obs_)))
            target_q2 = self.critic_target.critic2_forward(next_obs_, self.actor_target(next_obs_,self.vae.decode(next_obs_)))
            # print("################",target_q1.shape)
            target_q = self.lam*torch.min(target_q1,target_q2) + (1-self.lam)*torch.max(target_q1,target_q2)
            # print(target_q.reshape(mini_batch_size,-1).shape)
            target_q = target_q.reshape(mini_batch_size,-1).max(-1)[0].reshape(-1,1)
            target_q = rewards_batch.reshape(-1,1) + torch.mul((1-dones_batch.reshape(-1,1)),self.gamma*target_q) ##rewards_batch reshaped to avoid broadcasting 
            target_q.requires_grad = False
        
        q1 = self.critic.critic1_forward(obs_batch,actions_batch)
        q2 = self.critic.critic2_forward(obs_batch,actions_batch)
        critic_loss = nn.MSELoss()(q1,target_q) + nn.MSELoss()(q2,target_q)
        critic_loss_mean += critic_loss.item()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        sampled_action_batch = self.vae.decode(obs_batch)
        perutrb_action_batch = self.actor(obs_batch, sampled_action_batch)
        actor_loss = -self.critic.get_q(obs_batch,perutrb_action_batch).mean()
        actor_loss_mean += actor_loss.item()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm(self.actor_critic.parameters(),self.grad_clipping)
        self.optimizer_actor.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau* param.data + (1 - self.tau) * target_param.data)

        # for param, target_param in zip(self.actor_critic.critic2.parameters(), self.actor_critic.critic2_target.parameters()):
        #     target_param.data.copy_(self.tau* param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss_mean, critic_loss_mean, vae_loss_mean

        