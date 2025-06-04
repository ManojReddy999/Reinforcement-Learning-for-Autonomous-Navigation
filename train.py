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
import torch
from env import CarEnv
from feature_extractors import PointGNNFeatureExtractorWrapper, CNNFeatureExtractor, HistoryLocoTransformerExtractor
import yaml
import numpy as np
import pandas as pd
from evaluation import Evaluator
import re

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

BASE_SEED = 20
def make_env(rank, num_obstacles, model_inputs=CarEnv.ALL_MODEL_INPUTS):
    def _init():
        seed = BASE_SEED + rank
        env = CarEnv(num_obstacles=num_obstacles, model_inputs=model_inputs, random_seed=seed)
        # env.reset(seed=rank)
        # print(f"Initializing environment {rank} in process {os.getpid()}")
        return Monitor(env)
    return _init

def policy(timestep, model, model_inputs, evaluator):
    """
    Convert a dm_control TimeStep to the observation dict expected by the
    SB3 policy, then return the deterministic action.

    Parameters
    ----------
    timestep : dm_env.TimeStep
    model    : stable_baselines3 BaseAlgorithm (SAC, PPO, …)
    model_inputs : list[str]
        The same subset of ["velocity", "history", "depth", "point_cloud"]
        that you trained the model with.

    Returns
    -------
    np.ndarray  (action vector)
    """
    # --------------------------------------------------
    # 1. Build the observation dict
    # --------------------------------------------------
    obs = {}

    # ----- velocity branch ------------------------------------------
    if "velocity" in model_inputs:
        cur_vel = timestep.observation['car/body_vel_2d'].astype(np.float32)

        if "history" in model_inputs:
            # Allocate on first call
            if not hasattr(policy, "_vel_hist"):
                policy._vel_hist = np.tile(cur_vel, (5, 1))  # (5, 2)

            # FIFO queue: shift ↖ and append newest
            policy._vel_hist[:-1] = policy._vel_hist[1:]
            policy._vel_hist[-1]  = cur_vel
            obs["vec"] = policy._vel_hist.flatten()          # shape (10,)
        else:
            obs["vec"] = cur_vel                             # shape (2,)

    # ----- visual / geometric branch -------------------------------
    if "point_cloud" in model_inputs:
        obs["cam"] = timestep.observation['car/compute_point_cloud'].astype(np.float32)

    elif "depth" in model_inputs:
        depth = timestep.observation['car/realsense_camera'].astype(np.float32)
        # Optional live preview (comment out if not wanted)
        # cv2.imshow("Depth", cv2.convertScaleAbs(depth, alpha=0.15))
        # cv2.waitKey(1)
        obs["cam"] = depth                                   # shape (H, W, 1)  or (5, H, W, 1)

    evaluator.step(timestep)
    # --------------------------------------------------
    # 2. Predict and return the deterministic action
    # --------------------------------------------------
    action, _ = model.predict(obs, deterministic=True)
    return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Evaluate a model")
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="SAC", help="Algorithm to use")
    parser.add_argument("--config_path", type=str, default="config/sac_pc_reverse.yaml", help="Path to the config file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model to evaluate or continue training")
    parser.add_argument("--log_dir", type=str, default="results", help="Directory to save models and logs")
    parser.add_argument("--file_name", type=str, default="model", help="Name of the saved model file")
    # evaluation flags
    parser.add_argument("--eval", action="store_true", help='Run in evaluation mode')
    parser.add_argument('--seed', type=int, help='The random seed used to generate environment', default=0)
    parser.add_argument('--num_eval_envs', type=int, help='Number of environments to run evaluations in', default=10)
    parser.add_argument('--num_eval_episodes', type=int, help='Number of episodes to run evaluations for in each environment', default=1)
    parser.add_argument("--simulate", action="store_true", help="Simulate and evaluate trained model")
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    training_params = config.get("training",{})
    model_params = config.get("model", {})
    model_inputs = training_params['model_inputs']
    algorithm = {"PPO":PPO, "SAC":SAC}[args.algo]
    num_envs = training_params['num_envs']
    num_obstacles = training_params['num_obstacles']

    tb_dir = os.path.join(args.log_dir, "tensorboard", args.file_name)
    models_dir = os.path.join(args.log_dir, "models")
    logs_dir = os.path.join(args.log_dir, "logs")

    if "point_cloud" in model_inputs:
        print("using GNN")
        policy_kwargs = dict(
            features_extractor_class=PointGNNFeatureExtractorWrapper,
            features_extractor_kwargs=dict(input_dim=3,hidden_dim=32,output_dim=256,num_layers=3,use_edgeconv=False),
            net_arch=[256,256]
        )
    elif "history" in model_inputs:
        print("using LocoTransformer")
        policy_kwargs = dict(
            features_extractor_class=HistoryLocoTransformerExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256,256]
        )
    elif "depth" in model_inputs:
        print("using CNN")
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256,256]
        )

    if args.eval:
        model = algorithm.load(args.model_path,custom_objects={'buffer_size': 1})
        try:
            summary_prefix = re.findall(r"\/([^\/]+).zip", args.model_path)[0]
        except:
            summary_prefix = "CNN_vel"
        print(f"Loading model from '{args.model_path}'")
        evaluator = Evaluator(None, logs_dir, episodes=args.num_eval_episodes, prefix=summary_prefix, time_limit=1000)
        seedgen = np.random.SeedSequence(entropy=args.seed)
        seeds = seedgen.generate_state(args.num_eval_envs)
        for seed in seeds:
            env = CarEnv(num_obstacles=training_params["num_obstacles"], model_inputs=training_params["model_inputs"], random_seed=int(seed))
            evaluator.set_environment(env, seed)
            print(f'Start evaluating in environment with seed {seed}')
            obs,_ = env.reset()
            timestep = env.mj_state
            while not evaluator.check_finish():
                action = policy(timestep, model, model_inputs=training_params["model_inputs"], evaluator = evaluator)
                obs, reward, terminated,truncated,_ = env.step(action)
                timestep = env.mj_state
                # timestep = original_env.step(action)
                # if terminated or truncated:
                #     timestep = env.original_env.reset()
            
        evaluator.export_csv()
    else:
        env = SubprocVecEnv([make_env(i, num_obstacles=num_obstacles, model_inputs=model_inputs) for i in range(num_envs)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        if args.model_path:
            print(f"Loading model from '{args.model_path}'")
            model = algorithm.load(args.model_path, env=env)

        else:
            print("Creating new model")

            model = algorithm(
                **model_params,
                env=env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_dir,
            )

        # with torch.amp.autocast(device_type="cuda"):
        model.learn(total_timesteps=training_params['total_timesteps'])
        model_save_path = os.path.join(models_dir, f"{args.file_name}.zip")
        env.save(os.path.join(logs_dir, f"{args.file_name}_vecnormalize.pkl"))
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")