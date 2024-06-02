from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union
import os
from datetime import datetime

class env_cfg:
    env = 'hopper-medium-v2'
    num_obs = 11
    num_actions = 3
    clip_action = 100
    num_latent = 6
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'offline', 'envs')
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs')
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + 'TD3' + '_' + env)
    max_action = 1.

class alg_cfg:
    num_mini_batches = 256
    num_epochs = 200
    tau = 0.005
    num_transitions_per_env = 1000
    max_size = 999999
    num_iterations = 5000
    save_interval = 5000
    eval_interval = 1
    actor_update_freq = 2
    log_interval = 500