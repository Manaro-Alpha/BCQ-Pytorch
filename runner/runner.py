import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from algorithm.BCQ import BCQ
# from modules.actor_critic import ActorCritic, VAE
from modules import Actor, Critic, VAE
# from replay_buffer.replay_buffer import ReplayBuffer
from replay_buffer import ReplayBuffer
# from env.env_cfg import env_cfg, alg_cfg
from env import env_cfg, alg_cfg
import gym
import d4rl
from tqdm import tqdm
import numpy
import wandb

class Runner:
    def __init__(self,env_cfg:env_cfg,alg_cfg:alg_cfg,device='cpu',log_dir=None):
        self.device = device
        self.env_cfg = env_cfg
        self.alg_cfg = alg_cfg
        self.env = gym.make(self.env_cfg.env)
        self.dataset = d4rl.qlearning_dataset(self.env,terminate_on_end=True)
        self.max_size = self.dataset["observations"].shape[0]
        self.env_cfg.clip_action = float(self.env.action_space.high[0])
        print("###########################",self.dataset["observations"].shape[0])
        self.replaybuffer = ReplayBuffer(self.env_cfg.num_obs,self.env_cfg.num_actions,self.alg_cfg.num_transitions_per_env,device,self.max_size)
        # self.alg.init_storage(self.env_cfg.num_obs, self.env_cfg.num_actions, self.alg_cfg.num_transitions_per_env, device, self.alg_cfg.max_size)
        self.data_ = self.replaybuffer.Dataset(device)
        self.data_.observation = self.dataset["observations"]
        self.data_.next_observation = self.dataset["next_observations"]
        self.data_.action = self.dataset["actions"]
        self.data_.rewards = self.dataset["rewards"]
        print(self.dataset.keys())
        self.data_.dones = self.dataset["terminals"]
        # self.data_.infos = self.dataset["infos"]

        self.data_ = self.data_.convert_2_tensor()
        
        # self.actor_critic = ActorCritic(self.env_cfg.num_obs, self.env_cfg.num_actions, self.env_cfg.clip_action).to(self.device)
        self.actor = Actor(self.env_cfg.num_obs, self.env_cfg.num_actions, self.env_cfg.clip_action).to(self.device)
        self.critic = Critic(self.env_cfg.num_obs, self.env_cfg.num_actions).to(self.device)
        self.vae = VAE(self.env_cfg.num_obs, self.env_cfg.num_actions, self.env_cfg.num_latent, self.env_cfg.clip_action, device).to(self.device)
        self.alg = BCQ(self.actor, self.critic, self.vae, self.alg_cfg.num_epochs, self.alg_cfg.num_mini_batches, self.data_, self.replaybuffer, self.alg_cfg.tau)
        self.writer = None
        self.log_dir = log_dir
        self.save_interval = self.alg_cfg.save_interval
        self.total_time = 0
        self.eval_reward_mean = 0

    def evaluate(self):
        obs,done = self.env.reset(),False
        obs = torch.tensor(obs,dtype=torch.float,device=self.device)
        self.eval_reward = 0
        self.eval_reward_mean = 0
        # self.alg.actor_critic.eval()
        # self.alg.vae.eval()
        with torch.no_grad():
            for k in range(100):
                self.eval_reward = 0
                while not done:
                    obs = obs.reshape(1,-1).repeat(100,1)
                    action = self.alg.vae.decode(obs)
                    action = self.alg.actor(obs,action)
                    q1 = self.alg.critic.critic1_forward(obs,action)
                    idx = torch.argmax(q1,0)
                    action = action[idx]
                    action = action.detach()
                    obs, reward, done, info = self.env.step(action.cpu().numpy().flatten())
                    obs = torch.tensor(obs,dtype=torch.float,device=self.device)
                    self.eval_reward += reward
                self.eval_reward_mean += self.eval_reward
        print(f"=======================================Evaluation after 10 epochs==============================\n total_reward = {self.eval_reward_mean/100} \n ====================================================")
        # self.alg.actor_critic.train()
        # self.alg.vae.train()
    
    def learn(self, num_learning_iterations, num_epochs):
        self.alg.train_mode()
        if self.log_dir is not None and self.writer is None:
            # wandb.init(project="BCQ", sync_tensorboard=True, name=self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        for epoch in tqdm(range(num_epochs)):
            for it in tqdm(range(num_learning_iterations)):
                start = time.time()
                actor_loss, critic_loss, vae_loss = self.alg.learn()
                actor_loss /= self.alg_cfg.num_mini_batches
                critic_loss /= self.alg_cfg.num_mini_batches
                vae_loss /= self.alg_cfg.num_mini_batches

                stop = time.time()
                learning_time = stop - start
                # if it % self.alg_cfg.eval_interval == 0:
                #     self.evaluate()
                if self.log_dir is not None:
                    self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir,"model_{}.pt".format(it+(epoch*num_learning_iterations))),it+(epoch*num_learning_iterations))
            self.save(os.path.join(self.log_dir,"model_{}.pt".format(num_learning_iterations*(epoch+1))),num_learning_iterations*(epoch+1))
            if epoch % self.alg_cfg.eval_interval == 0:
                    self.evaluate()
        

    def log(self, locs, width=80, pad=35):
        self.total_timesteps = self.alg_cfg.num_transitions_per_env
        self.total_time += locs['learning_time']
        iteration_time = locs['learning_time']

        ep_string = f''
        # if locs['ep_infos']:
        #     for key in locs['ep_infos'][0]:
        #         infotensor = torch.tensor([], device=self.device)
        #         for ep_info in locs['ep_infos']:
        #             # handle scalar and zero dimensional tensor infos
        #             if not isinstance(ep_info[key], torch.Tensor):
        #                 ep_info[key] = torch.Tensor([ep_info[key]])w
        #             if len(ep_info[key].shape) == 0:
        #                 ep_info[key] = ep_info[key].unsqueeze(0)
        #             infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        #         value = torch.mean(infotensor)
        #         self.writer.add_scalar('Episode/' + key, value, locs['it'])
        #         ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # mean_std = self.alg.actor_critic.std.mean()
        # fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        self.writer.add_scalar('Loss/value_function', locs['critic_loss'], locs['it']+locs['epoch']*locs['num_learning_iterations'])
        self.writer.add_scalar('Loss/surrogate', locs['actor_loss'], locs['it']+locs['epoch']*locs['num_learning_iterations'])
        self.writer.add_scalar('Loss/autoenc_function', locs['vae_loss'], locs['it']+locs['epoch']*locs['num_learning_iterations'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.lr, locs['it']+locs['epoch']*locs['num_learning_iterations'])
        if (locs["epoch"])%1 == 0:
            self.writer.add_scalar('Eval/eval_reward', self.eval_reward_mean/100, locs['epoch'])
        # self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        # self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        # self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learning_time'], locs['it']+locs['epoch']*locs['num_learning_iterations'])
        # if len(locs['rewbuffer']) > 0:
        #     self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
        #     self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
        #     self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.total_time)
        #     self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.total_time)

        str = f" \033[1m Learning iteration {locs['it']+(locs['epoch']*locs['num_learning_iterations'])}/{locs['num_learning_iterations']*(locs['num_epochs'])} \033[0m "

        # if len(locs['rewbuffer']) > 0:
        #     log_string = (f"""{'#' * width}\n"""
        #                   f"""{str.center(width, ' ')}\n\n"""
        #                 #   f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
        #                     # 'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
        #                   f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
        #                   f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
        #                   f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
        #                 #   f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        #                   f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
        #                   f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        #                 #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
        #                 #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        # else:
        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                    #   f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    #     'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {locs['critic_loss']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {locs['actor_loss']:.4f}\n"""
                        f"""{'Autoenc function loss:':>{pad}} {locs['vae_loss']:.4f}\n"""
                    #   f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                    #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                    )
        

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.total_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.total_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.total_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if locs["it"]%100 == 0:
            print(log_string)

    def save(self, path, it, infos=None):
        torch.save({
            'actor_state_dict': self.alg.actor.state_dict(),
            'optimizer_actor_state_dict': self.alg.optimizer_actor.state_dict(),
            'critic_state_dict': self.alg.critic.state_dict(),
            'optimizer_critic_state_dict': self.alg.optimizer_critic.state_dict(),
            "vae_state_dict": self.alg.vae.state_dict(),
            "optimizer_vae": self.alg.optimizer_vae.state_dict(),
            'iter': it,
            'infos': infos,
            }, path)
