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
import pandas as pd
import sys
import traceback
import json
import pandas as pd
sys.path.insert(1, 'global_racetrajectory_optimization')
from global_racetrajectory_optimization.main_globaltraj_function import get_beast_map

maxspeed = 0.3
targetVel = 0.3

env = RacingEnv(EnvConfig.__dict__, SimulatorConfig.__dict__)
env.make()
# print(env.reset())
# check_env(env)

done = False
obs = env.reset()

nix = obs[17]



def pathToCsvFormat(pathname, nix = 2):
  #save
  # /home/rosim/amin/L2R/firstRl/l2r-starter-kit-master/global_racetrajectory_optimization/inputs/tracks
  save_path = "global_racetrajectory_optimization/inputs/tracks/"
  saveFile = save_path + pathname + '.csv'
  #check if file exists
  if os.path.isfile(saveFile):
    print("file exists")
    return

  f = open("racetracks/thruxton/" + pathname + ".json")
  mapdata = json.load(f)
  # inside = np.array(mapdata["Inside"])
  # outside = np.array(mapdata["Outside"])
  centre = np.array(mapdata["Centre"])

  # l = len(centre)
  # centre_data = []
  # for i in range(l):
  #   print((i + nix) % l)
  #   f = centre[int((i + nix) % l)]
  #   centre_data.append(f)

  centre_data = centre

  # x_m 	y_m 	w_tr_right_m 	w_tr_left_m
  # print(centre, 5, 5)

  # print(np.sqrt((outside[:,0] - centre[:,0]) ** 2 +  (outside[:,1] - centre[:,0]) ** 2))
  # print(np.sqrt((inside[:,0] - centre[:,0]) ** 2 +  (inside[:,1] - centre[:,0]) ** 2))


  path = pd.DataFrame(centre_data, columns = ['#x_m','y_m'])
  path['w_tr_right_m'] = 5
  path['w_tr_left_m'] = 5

  path.to_csv(save_path + pathname + '.csv', index=False)
  return path




pathToCsvFormat('ThruxtonOfficial')
print("start_x" , "start_y", obs[16], obs[15])
get_beast_map('ThruxtonOfficial', obs[16], obs[15])

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


    # print(i , obs[17] , obs[12], st)

    i = i + 1
    