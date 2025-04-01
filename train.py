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

def make_env(rank,arena_size=10, model_inputs=CarEnv.ALL_MODEL_INPUTS):
    def _init():
        env = CarEnv(arena_size=arena_size, model_inputs=model_inputs)
        # print(f"Initializing environment {rank} in process {os.getpid()}")
        return env
    return _init

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train or Evaluate a model")
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="SAC", help="Algorithm to use")
    parser.add_argument("--config_path", type=str, default="config/sac_point_cloud_reverse.yaml", help="Path to the config file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model to evaluate or continue training")
    parser.add_argument("--log_dir", type=str, default="results", help="Directory to save models and logs")
    parser.add_argument("--file_name", type=str, default="model", help="Name of the saved model file")
    parser.add_argument("--simulate", action="store_true", help="Simulate and evaluate trained model")
    args = parser.parse_args()
    
    tb_dir = os.path.join(args.log_dir, "tensorboard", args.file_name)
    models_dir = os.path.join(args.log_dir, "models")
    logs_dir = os.path.join(args.log_dir, "logs")

    config = load_yaml(args.config_path)
    training_params = config.get("training",{})
    model_params = config.get("model", {})

    model_inputs = training_params['model_inputs']
    algorithm = {"PPO":PPO, "SAC":SAC}[args.algo]
    num_envs = training_params['num_envs']
    
    env = SubprocVecEnv([make_env(i,model_inputs=model_inputs) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    if args.model_path:
        print("Loading modek from {args.model_path}")
        model = algorithm.load(args.model_path, env=env)
    else:
        print("Creating new model")
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

        model = algorithm(
            **model_params,
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir
        )

    # with torch.amp.autocast(device_type="cuda"):
    model.learn(total_timesteps=training_params['total_timesteps'])
    model_save_path = os.path.join(models_dir, f"{args.file_name}.zip")
    env.save(os.path.join(logs_dir, f"{args.file_name}_vecnormalize.pkl"))
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")