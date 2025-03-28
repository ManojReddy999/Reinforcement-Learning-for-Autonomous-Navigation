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

from ImageEnv import CarEnv
from NetArch import DepthMapFeatureExtractor, CustomFeaturesExtractor, CustomMultiInputExtractor

def make_env(rank):
    def _init():
        # print(f"Initializing environment {rank} in process {os.getpid()}")
        return CarEnv()
    return _init

if __name__ == '__main__':
    
    tb = os.path.join("/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/tensor","SAC_2DCNN_300k")
    models = os.path.join("/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/models","SAC_2DCNN_300k")
    stats_path = os.path.join("/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/logs","SAC_2DCNN_300k")

    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    # env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    policy_kwargs = dict(
        features_extractor_class=CustomMultiInputExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256,256]
    )

    # model = PPO(
    #     policy='MultiInputPolicy',
    #     n_steps = 2048,
    #     learning_rate=3e-4,
    #     n_epochs=10,
    #     env=env,
    #     batch_size=16,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     vf_coef=0.5,
    #     verbose=1,
    #     tensorboard_log=tb,
    #     seed=1,
    #     policy_kwargs=policy_kwargs
    # )


    model = SAC(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100000,
        learning_starts=1000,
        tensorboard_log=tb,
        policy_kwargs=policy_kwargs,
        verbose=1
    )


    # saved_model_path='/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/models/PPO_GS_100k_32_128'
    # model=PPO.load(saved_model_path, env=env)
    # model.policy.half()

    # with torch.amp.autocast(device_type="cuda"):
    model.learn(total_timesteps=300000)
    env.save(stats_path+'test'+".pkl")
    print("Saving end model")
    model.save(models)

    # model.policy