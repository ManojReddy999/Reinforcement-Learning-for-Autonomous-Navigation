import os
import argparse
import time

from stable_baselines3 import SAC, PPO
# from sbx import SAC

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import torch.optim as optim

import car
import numpy as np
# from NavEnv import CarEnv
# from NavTask import Navigation
# from SurvivalEnv import CarEnv
# from ImageOnlyEnv import CarEnv
from Point_Cloud_Env import CarEnv
from SurvivalTask import Survive
# from MazeTask import Maze
# from ButtonEnv import CarEnv
# from ButtonTask import PressWithSpecificForce
from dm_control import viewer,composer

env=CarEnv()

task = env.task
# task = Maze(creature)
# task = PressWithSpecificForce(creature)
original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

saved_model_path='/home/mmkr/Car_LocoTransformer/Mujoco_learning/Pushr_car_simulation/results/models/PPO_PointGNN_300k_r025_F_90N_8_128'
model=PPO.load(saved_model_path, env=env)

def random_policy(time_step):
    
    # pos = time_step.observation['car/body_pose_2d']
    # image = time_step.observation['car/realsense_camera'][::2]
    pc = time_step.observation['car/compute_point_cloud']
    # vec = np.array([np.linalg.norm(time_step.observation['car/body_vel_2d'])])
    # obs = time_step.observation['car/body_vel_2d']

    # obs = np.concatenate([state['vec'][:-2],obs])

    # state = { "vec": obs, "img": image}
    obs = {"pc": pc}
    # state = {"vec": vec, "pc": pc}

    action, _=model.predict(obs,deterministic=True)
    
    return action

viewer.launch(original_env, policy=lambda timestep: random_policy(timestep))
