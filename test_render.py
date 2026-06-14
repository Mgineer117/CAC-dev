import torch
import numpy as np
from utils.get_args import get_args
from utils.utils import get_env

# mock args
class Args:
    pass

args = Args()
args.task = "car"
args.sample_mode = "uniform"
args.reward_mode = "default"
args.num_windows = 1

env = get_env(args)
obs, info = env.reset()
env.render()

for _ in range(5):
    u = env.action_space.sample()
    env.step(u)
    env.render()
