from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union
from env import env_cfg, alg_cfg

class ant_cfg(env_cfg):
    num_actions = 8
    num_obs = 111
    num_latent = num_actions*2
    clip_action = 1.0

class alg_cfg(alg_cfg):
    num_mini_batches = 100
    num_epochs = 10000
    tau = 0.005
    num_transitions_per_env = 25000
    max_size = 999999
    num_iterations = 10000
    

