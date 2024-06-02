import gym
import pickle
import torch
import os
from algorithm import BCQ
from modules import Actor, Critic, VAE

device = torch.device("cpu")

actor = Actor(11,3,1.0).to(device)
critic = Critic(11,3)
vae = VAE(11,3,6,1.0,device).to(device)
env = gym.make("Hopper-v2")
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'offline', 'envs')
log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs')
log_dir = os.path.join(log_root, "May02_11-53-37_1hopper-medium-v2")
it = 505000
policy = torch.load(os.path.join(log_dir,"model_{}.pt".format(it)))
# print(policy)
# policy = torch.jit.load(policy["model_state_dict"])
actor.load_state_dict(policy["actor_state_dict"])
critic.load_state_dict(policy["critic_state_dict"])
vae.load_state_dict(policy["vae_state_dict"])
actor.eval()
critic.eval()
vae.eval()
# observation = env.reset(seed=42)


obs,done = env.reset(),False
obs = torch.tensor(obs,dtype=torch.float,device=device)
eval_reward = 0
for _ in range(1000):
    obs = torch.tensor(obs).reshape(1,-1).repeat(100,1)
    # print(obs)
    action = vae.decode(obs)
    action = actor(obs,action)
    q1 = critic.get_q(obs,action)
    idx = torch.argmax(q1,0)
    action = action[idx]
    action = action.detach().clamp(-1.,1.)
    obs, reward, done, info = env.step(action.cpu().numpy().flatten())
    env.render(mode='human')
    obs = torch.tensor(obs,dtype=torch.float,device=device)
    eval_reward += reward
    if done:
        break
print(eval_reward)
env.close()