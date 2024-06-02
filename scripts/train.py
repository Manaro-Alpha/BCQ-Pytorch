import numpy as np
import os
from datetime import datetime

import torch
from runner import Runner
from env import env_cfg, alg_cfg

def train(env_cfg:env_cfg, alg_cfg:alg_cfg):
    device = torch.device("cpu")
    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    print("Device set to " + str(torch.cuda.get_device_name(device)))
    runner = Runner(env_cfg,alg_cfg,device,log_dir=env_cfg.log_dir)
    print("###################################")
    runner.learn(alg_cfg.num_iterations,alg_cfg.num_epochs)

if __name__ == '__main__':
    train(env_cfg,alg_cfg)