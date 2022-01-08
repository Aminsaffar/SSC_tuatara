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


# import env
from l2r.envs import env


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
        # self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(30, ), dtype=np.float64)
        self.observation_space = Box(low=-float(-1.0), high=float(1.0), shape=(30, ), dtype=np.float64)


   


    def step(self, action):
        observation, reward, done, info = env.RacingEnv.step(self, action)  # update the simulator
        _data, _imgs = observation
        observation = _data
        
        return observation, reward, done, info

    def reset(self, level=None, random_pos=False, segment_pos=True):
        self.maxV = 0
        observation = env.RacingEnv.reset(self, level, random_pos, segment_pos)
        _data, _imgs = observation
        observation = _data[0]
        # print(observation)
        return observation


    def _observe(self):
        pose, self.imgs = env.RacingEnv._observe(self)

        enu_x, enu_y, enu_z = pose[16], pose[15], pose[17] 

        lookAhead = 3
        # for i in range(lookAhead):
        #     nix = (self.nearest_idx + (i + 1) * 2) % self.n_indices #:_)
        #     centerx, centery = self.centerline_arr[nix]
        #     # print("--" + str(centery - enu_y))
        #     pose[15 + i] = centery - enu_y

        nix = (self.nearest_idx) % self.n_indices #:_)
        centerx, centery = self.centerline_arr[nix]

        myradians = math.atan2(centery-enu_y, centerx-enu_x)
        # mydegrees = math.degrees(myradians)


        # print(centerx, centery , enu_x, enu_y , myradians, pose[12])

        # print("--" + str(centery - enu_y))
        # pose[15] = centery - enu_y
        pose[17] = myradians

        
        pose = self.obs_scallar(pose)
        return (pose, self.imgs)

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

        return observation




    def _is_complete(self, observation):
        is_complete, info = env.RacingEnv._is_complete(self, observation)

        _data, _imgs = observation

        vx , vy, vz = _data[3], _data[4], _data[5]
        v = math.sqrt(vx**2 + vy**2)



        if(v > self.maxV):
            self.maxV = v
        
        # print(v)

        if(v < 0.01 and self.maxV > 0.018):
            is_complete = True
            print("early stop!!")

        return is_complete, info
