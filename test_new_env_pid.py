# from l2r.envs.env import RacingEnv
from l2r.envs.newenv import RacingEnv
from config import SubmissionConfig, EnvConfig, SimulatorConfig
from stable_baselines3.common.env_checker import check_env
import gym
from gym import Env
from gym.spaces import Box

from scipy.spatial import KDTree
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


# ##read csv file with ; as delimiter and # as comment and ["#x_m","y_m"] as column names
# path = pd.read_csv("global_racetrajectory_optimization/outputs/traj_race_cl.csv", delimiter = ';', comment = '#', names = ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"])
# path = path.to_numpy()
# # print(path)
# # exit()


# points =path[:, 1:3]

# T = KDTree(points)
# # idx = T.query_ball_point([-27,-30], r=2)
# idx = T.query([-50,30])[1]
# print(path[idx])
# exit()

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
  path['w_tr_right_m'] = 2#5
  path['w_tr_left_m'] = 2#5

  path.to_csv(save_path + pathname + '.csv', index=False)
  return path


# pathToCsvFormat('ThruxtonFromGoogle_2')
# print("start_x" , "start_y", obs[16], obs[15])
# get_beast_map('ThruxtonFromGoogle_2', obs[16], obs[15])



#read csv file with ; as delimiter and # as comment and ["#x_m","y_m"] as column names
path = pd.read_csv("global_racetrajectory_optimization/outputs/traj_race_cl.csv", delimiter = ';', comment = '#', names = ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"])
path = path.to_numpy()
# print(path)
# exit()


points =path[:, 1:3]




T = KDTree(points)
# idx = T.query_ball_point([1,1], r=2)
# print(path[idx])

i = 0 
myradians = 0
while(not done):
    # idx = T.query_ball_point([-27,-30], r=2)
    cur_x , cur_y = obs[16], obs[15]
    i = T.query([cur_x ,cur_y])[1]
    i = i + 3
    # print(path[idx])


    # print(i)
    # print(path[i])
    theta = path[i][3]
    # Random action
    # action = env.action_space.sample()
    cu_velx = obs[3]
    delta_to_line = obs[17]
    # targetVel = maxspeed * ((1 - abs(delta_to_line - myradians)))

    # targetVel = path[i][5] * 2. / 3.
    targetVel = path[i][5] / 2
    # print(targetVel, cu_velx)
    ax = (targetVel - cu_velx) * 2
    if ax > 1:
      ax = 1
    if ax < -1:
      ax = -1

    # ax = 1

    # ax = path[i][6] * 0.1
    # print(ax)
    theta_to_path = math.atan2(cur_y-path[i][2], cur_x-path[i][1])
    myradians =  obs[12]
    st = -1 * ((-1 * theta - myradians)) * 9#5
    if st > 1:
      st = 1
    if st < -1:
      st = -1

    # st = 0
    action = np.array([st,ax])
    prevx , prevy = obs[16], obs[15]
    obs, reward, done, info = env.step(action)
    
    cur_x , cur_y = obs[16], obs[15]
    # myradians = math.atan2(cur_y-prevy, cur_x-prevx)
    delta_x = cur_x - path[i][1]
    delta_y = cur_y - path[i][2]
    print(i , -1*theta , myradians, st, delta_x, delta_y)

    i = i + 1
    

obs = env.reset()