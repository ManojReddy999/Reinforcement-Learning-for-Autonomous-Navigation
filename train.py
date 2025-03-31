import os
import argparse
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import torch.optim as optim
from env import CarEnv
from feature_extractors import PointGNNFeatureExtractorWrapper, CNNFeatureExtractor
import yaml

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
config = load_yaml("config/sac_point_cloud_reverse.yaml")
training_params = config.get("training",{})
model_params = config.get("model", {})
model_inputs = training_params['model_inputs']

def make_env(rank,arena_size=10, model_inputs=CarEnv.ALL_MODEL_INPUTS):
    def _init():
        env = CarEnv(arena_size=arena_size, model_inputs=model_inputs)
        # print(f"Initializing environment {rank} in process {os.getpid()}")
        return env
    return _init

if __name__ == '__main__':
    
    tb = os.path.join("results/tensor","SAC_GNN_100k")
    models = os.path.join("results/models","SAC_GNN_100k")
    stats_path = os.path.join("results/logs","SAC_GNN_100k")

    num_envs = training_params['num_envs']
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    if "point_cloud" in model_inputs:
        policy_kwargs = dict(
            features_extractor_class=PointGNNFeatureExtractorWrapper,
            features_extractor_kwargs=dict(input_dim=3,hidden_dim=32,output_dim=256,num_layers=3,use_edgeconv=False),
            net_arch=[256,256]
        )
    elif "depth" in model_inputs:
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256,256]
        )

    model = SAC(
        **model_params,
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb
    )

    # saved_model_path='/home/mmkr/LocoTransformer_v2/Pushr_car_simulation/results/models/PPO_GS_100k_32_128'
    # model=PPO.load(saved_model_path, env=env)
    # model.policy.half()

    # with torch.amp.autocast(device_type="cuda"):
    model.learn(total_timesteps=training_params['total_timesteps'])
    env.save(stats_path+'test'+".pkl")
    print("Saving end model")
    model.save(models)