# ========================================================================= #
# Filename:                                                                 #
#    env.py                                                                 #
#                                                                           #
# Description:                                                              #
#    Reinforcement learning environment for autonomous racing               #
# ========================================================================= #

import numpy as np
import math
from gym.spaces import Box

import os
from datetime import datetime
# import env
from l2r.envs import env

import matplotlib.pyplot as plt
import pandas as pd
import sys
import traceback
import json
from scipy.spatial import KDTree
import pandas as pd
class RacingEnv(env.RacingEnv):
    def __init__(
        self,
        env_kwargs,
        sim_kwargs,
        segm_if_kwargs=False,
        birdseye_if_kwargs=False,
        birdseye_segm_if_kwargs=False,
        zone=False,
        provide_waypoints=False,
        manual_segments=False,
        multi_agent=False,
    ):
        env.RacingEnv.__init__(
            self,
            env_kwargs,
            sim_kwargs,
            segm_if_kwargs,
            birdseye_if_kwargs,
            birdseye_segm_if_kwargs,
            zone,
            provide_waypoints,
            manual_segments,
            multi_agent,
        )

        self.maxV = 0
        self.trajectoryLog = []

        #read csv file with ; as delimiter and # as comment and ["#x_m","y_m"] as column names
        path = pd.read_csv("global_racetrajectory_optimization/outputs/traj_race_cl.csv", delimiter = ';', comment = '#', names = ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"])
        path = path.to_numpy()
        # print(path)
        # exit()


        points =path[:, 1:3]



        self.path = path
        self.pathTree = KDTree(points)

    

    def make(
        self,
        level=False,
        multimodal=False,
        sensors=False,
        camera_params=False,
        driver_params=False,
        segm_params=False,
        birdseye_params=False,
        birdseye_segm_params=False,
        vehicle_params=None,
        multi_agent=False,
        remake=False,
    ):
        env.RacingEnv.make(
            self,
            level,
            multimodal,
            sensors,
            camera_params,
            driver_params,
            segm_params,
            birdseye_params,
            birdseye_segm_params,
            vehicle_params,
            multi_agent,
            remake,
        )
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(33, ), dtype=np.float64)
        # self.observation_space = Box(low=-float(-1.0), high=float(1.0), shape=(30, ), dtype=np.float64)


   


    def step(self, action):
        observation, reward, done, info = env.RacingEnv.step(self, action)  # update the simulator
        _data, _imgs = observation
        
        #log trajectory
        self.trajectoryLog.append(
            {
                "action" : action,
                "obs" : observation,
                "rewared" : reward,
                "done" : done,
                "info" : info
            }
        );
        observation = _data
        reward = self.getRewardBasedOnTrajectory(_data)
        return observation, reward, done, info

    def reset(self, level=None, random_pos=False, segment_pos=True):
        # for speed up stop segmenting
        segment_pos = False 

        #save trajectory
        trajectoryDir = "EnvTrajectory/"
        os.makedirs(trajectoryDir, exist_ok=True)
        filename = trajectoryDir + datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S %f') + ".npy"
        # np.save(filename,  self.trajectoryLog)    
        self.trajectoryLog = []

        self.maxV = 0
        observation = env.RacingEnv.reset(self, level, random_pos, segment_pos)
        _data, _imgs = observation
        observation = _data[0]
        # print(observation)
        return observation


    def _observe(self):
        pose, self.imgs = env.RacingEnv._observe(self)

        # enu_x, enu_y, enu_z = pose[16], pose[15], pose[17] 

        # lookAhead = 3
        # for i in range(lookAhead):
        #     nix = (self.nearest_idx + (i) * 2) % self.n_indices #:_)
        #     centerx, centery = self.centerline_arr[nix]
        #     # print("--" + str(centery - enu_y))
        #     pose[15 + i] = centery - enu_y

        # nix = (self.nearest_idx) % self.n_indices #:_)
        # centerx, centery = self.centerline_arr[nix]
        # print(nix, centerx, centery ,  enu_x , enu_y, centery-enu_y, centerx-enu_x)

        # myradians = math.atan2(centery-enu_y, centerx-enu_x)
        # # mydegrees = math.degrees(myradians)


        # # # print(centerx, centery , enu_x, enu_y , myradians, pose[12])

        # # # print("--" + str(centery - enu_y))
        # # # pose[15] = centery - enu_y
        # # pose[17] = myradians
        # pose[17] = nix

        cur_x , cur_y, cur_z = pose[16], pose[15], pose[17]
        # ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"]
        i = self.pathTree.query([cur_x ,cur_y])[1]
        i = i + 3
        i = i % len(self.path)
        # print(path[idx])


        # print(i)
        # print(path[i])
        sm = self.path[i][0]
        xm = self.path[i][1]
        ym = self.path[i][2]

        
        psi_rad = -1 * self.path[i][3]
        if ((psi_rad - pose[12]) > 1.5) or ((psi_rad - pose[12]) < -1.5):
            if psi_rad < 0:
                psi_rad = psi_rad + math.pi 
            else: 
                psi_rad = psi_rad - math.pi

        kappa = self.path[i][4]
        vx = self.path[i][5]
        ax = self.path[i][6]

        vx = vx / 3.0

        pose[16] = pose[16] - xm
        pose[15] = pose[15] - ym

        pose[17] = psi_rad
        pose = np.concatenate([pose, [kappa, vx, ax]])
        # pose = self.obs_scallar(pose)
        return (pose, self.imgs)

    def getRewardBasedOnTrajectory(self, pose):
        deltax = pose[16]
        deltay = pose[15]

        psi_rad = pose[17]
        rad = pose[12]
        # print("             ", psi_rad, rad)
        delta_theta = psi_rad - rad

        deltaSpeed = pose[31] - pose[3]
        if pose[3] < 0:
            return -4.0
        # print(pose[3])

        error_speed =  (deltaSpeed ** 2)    / 200.0
        error_theta = math.sqrt(delta_theta ** 2)  / 1.57
        error_y = (deltay ** 2) / 25.0
        #scale to one
        # print(error_speed, error_theta, error_y)
        error  =  error_y + error_theta +  error_speed

        r = (2.0 - error) / 2.0
        # print(r, error)
        return r



    def obs_scallar(self, observation):
        env.MIN_OBS_ARR[15] = -10.0  
        env.MIN_OBS_ARR[16] = -10.0
        env.MIN_OBS_ARR[17] = -10.0
        env.MAX_OBS_ARR[15] = 10.0  
        env.MAX_OBS_ARR[16] = 10.0
        env.MAX_OBS_ARR[17] = 10.0   
        for i in range(len(observation)):
            observation[i] = ((observation[i] -  env.MIN_OBS_ARR[i]) / (env.MAX_OBS_ARR[i] - env.MIN_OBS_ARR[i]))
            observation[i] = (observation[i] * 2) - 1

        observation[17] = (self.nearest_idx) % self.n_indices #:_) option for observed 
        return observation




    def _is_complete(self, observation):
        is_complete, info = env.RacingEnv._is_complete(self, observation)

        _data, _imgs = observation

        vx , vy, vz = _data[3], _data[4], _data[5]
        v = math.sqrt(vx**2 + vy**2)



        if(v > self.maxV):
            self.maxV = v
        
        # print(v)

        # if(v < 5 and self.maxV > 10):
        #     is_complete = True
        #     print("early stop!!")

        return is_complete, info
