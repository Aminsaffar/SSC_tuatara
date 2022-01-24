# from l2r.envs.env import RacingEnv
from l2r.envs.folowTrajectoryEnv import RacingEnv
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
import traceback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


from gym import Wrapper


class FrameSkipWrapper(Wrapper):
    '''
    Implements a frame kipping wrapper.
    As described by Mnih et al., it returns only every 'k-th' frame
    '''
    def __init__(self, env=None, k=4):
        super(FrameSkipWrapper, self).__init__(env)
        self._k = k

    def step(self, action):
        r = 0.0
        done = False
        # print("frame skip!")

        # Step the env `_k` times and return the last obs
        for _ in range(self._k):
            # print("do")
            obs, reward, done, info = self.env.step(action)
            # print("skippied")
            r += reward
            if done:
                break
        return obs, r, done, info


class ActionEnhancer(Wrapper):
    '''
    '''
    def __init__(self, env=None):
        super(ActionEnhancer, self).__init__(env)

    def step(self, action):
        acc = action[1]
        if acc < 0:
          acc = acc / 2.0
        # newacc = acc * 6.0 if acc > 0 else acc* (-1.0 * -16.0)
        # print(acc, newacc)
        action[1] = acc
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'ppo_autosave_ssc_MlpPolicy_folowTrajs')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed #todo remove this line doesnt need folder
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls < self.check_freq * 4:
            return True #dont save anything at start
            
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True






env = RacingEnv(EnvConfig.__dict__, SimulatorConfig.__dict__)

env.make()
# print(env.reset())
# check_env(env)

log_dir = 'monitor.csv'


env = Monitor(env, log_dir)
env = FrameSkipWrapper(env, 4)
env = ActionEnhancer(env)
log_dir = ''
callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=log_dir)

# net_arch=[dict(pi=[64, 64, 32], vf=[64, 64, 32])]
net_arch=[dict(pi=[256, 256], vf=[128, 128])]

log_path = os.path.join("ssc_learning_logs")

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch': net_arch}, device = 'cpu')


try:
    model = PPO.load('ppo_autosave_ssc_MlpPolicy_folowTrajs.zip', device = 'cpu')
    model.set_env(env)
    model.learn(total_timesteps=300000000, callback=callback)

except:
    print("exception happend")
    print(traceback.format_exc())

model.save("ssc_end_of_learning")