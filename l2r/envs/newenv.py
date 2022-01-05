# ========================================================================= #
# Filename:                                                                 #
#    env.py                                                                 #
#                                                                           #
# Description:                                                              #
#    Reinforcement learning environment for autonomous racing               #
# ========================================================================= #

import numpy as np
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
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(30, ), dtype=np.float64)

   


    def step(self, action):
        observation, reward, done, info = env.RacingEnv.step(self, action)  # update the simulator
        _data, _imgs = observation
        observation = _data
        
        return observation, reward, done, info

    def reset(self, level=None, random_pos=False, segment_pos=True):
        observation = env.RacingEnv.reset(self, level, random_pos, segment_pos)
        _data, _imgs = observation
        observation = _data[0]
        print(observation)
        
        return observation


    def _observe(self):
        pose, self.imgs = env.RacingEnv._observe(self)

        enu_x, enu_y, enu_z = pose[16], pose[15], pose[17] 

        lookAhead = 3
        for i in range(lookAhead):
            self.nearest_idx = (self.nearest_idx + (i + 1) * 4) % self.n_indices #:_)
            centerx, centery = self.centerline_arr[self.nearest_idx]
            print("--" + str(centery - enu_y))
            pose[15 + i] = centery - enu_y
        

        return (pose, self.imgs)
