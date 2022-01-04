from l2r.envs.env import RacingEnv
from config import SubmissionConfig, EnvConfig, SimulatorConfig
from stable_baselines3.common.env_checker import check_env
import gym
from gym import Env
from gym.spaces import Box

import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


env = RacingEnv(EnvConfig.__dict__, SimulatorConfig.__dict__)
env.make()
# print(env.reset())
# check_env(env)

net_arch=[dict(pi=[512, 512, 32], vf=[512, 512, 32])]

log_path = os.path.join("2rr_logs")


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch': net_arch})

model.set_env(env)
model.learn(total_timesteps=300000000)