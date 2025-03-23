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

from Point_Cloud_Env import CarEnv
# from NetArch import DepthMapFeatureExtractor, CustomFeaturesExtractor, DepthMap3DFeatureExtractorWrapper
from GNN_Arch import PointGNNFeatureExtractorWrapper, GSageFeatureExtractorWrapper
import os
from torch.amp import autocast
import torch

# os.environ["MJ_THREADS"] = "4"


def make_env(rank):
    def _init():
        # print(f"Initializing environment {rank} in process {os.getpid()}")
        return CarEnv()
    return _init

if __name__ == '__main__':
    
    tb = os.path.join("/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/tensor","SAC_GC_4N_200k_32_256")
    models = os.path.join("/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/models","SAC_GC_4N_200k_32_256")
    stats_path = os.path.join("/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/logs","SAC_GC_4N_200k_32_256")

    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)


    policy_kwargs = dict(
        features_extractor_class=PointGNNFeatureExtractorWrapper,
        features_extractor_kwargs=dict(input_dim=3,hidden_dim=32,output_dim=256,num_layers=3,use_edgeconv=False),
        net_arch=[256,256]
    )

    model = SAC(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=3e-4,
        batch_size=32,
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
    model.learn(total_timesteps=200000)
    env.save(stats_path+'test'+".pkl")
    print("Saving end model")
    model.save(models)

    # model.policy