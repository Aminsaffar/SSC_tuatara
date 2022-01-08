# from l2r.envs.env import RacingEnv
from l2r.envs.newenv import RacingEnv
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


import traceback



maxspeed = 0.3
targetVel = 0.3

env = RacingEnv(EnvConfig.__dict__, SimulatorConfig.__dict__)
env.make()
# print(env.reset())
# check_env(env)

done = False
obs = env.reset()
i = 0 
myradians = 0
while(not done):
    # Random action
    # action = env.action_space.sample()
    cu_velx = obs[3]
    delta_to_line = obs[17]
    targetVel = maxspeed * ((1 - abs(delta_to_line - myradians)))

    ax = (targetVel - cu_velx)  * 1

    

    myradians =  obs[12]
    st = -1 *((delta_to_line - myradians)) * 1.5
    if st > 1:
      st = 1
    if st < -1:
      st = -1

    # st = 0
    action = np.array([st,ax])
    prevx , prevy = obs[16], obs[15]
    obs, reward, done, info = env.step(action)
    
    cur_x , cur_y = obs[16], obs[15]
    myradians = math.atan2(cur_y-prevy, cur_x-prevx)


    print(i , obs[17] , obs[12], st)

    i = i + 1
    