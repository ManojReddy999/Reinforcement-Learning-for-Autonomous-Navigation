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

def evaluate(model, env, model_inputs, num_obstacles, sheet_name: str, n_episodes: int, xlsx_path: str):
    # env = DummyVecEnv([make_env(0, num_obstacles, model_inputs)])
    raw_env: CarEnv = env.envs[0]

    rows = []
    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        step_cnt, speed_sum, top_speed = 0, 0.0, 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            reward = float(rewards[0])
            done = bool(done[0])
            speed = np.linalg.norm(raw_env.mj_state.observation["car/body_vel_2d"])
            speed_sum += speed
            top_speed = max(top_speed, speed)
            step_cnt += 1

        collision = int(raw_env.obstructed())
        rows.append(
            {
                "episode": ep,
                "episode_length": step_cnt,
                "collision": collision,
                "distance": raw_env.dist,
                "avg_speed": speed_sum / step_cnt if step_cnt else 0.0,
                "top_speed": top_speed,
                "reward": raw_env.rewardAccumulated,
            }
        )
        print(
            f"[Eval] {sheet_name}  Ep {ep}/{n_episodes}  "
            f"len={step_cnt}  coll={collision}  dist={raw_env.dist:.2f}  "
            f"avg_spd={speed_sum/step_cnt:.2f}  top_spd={top_speed:.2f}  "
            f"R={raw_env.rewardAccumulated:.1f}"
        )

    # aggregate row
    agg = {
        "episode": "mean",
        "episode_length": np.mean([r["episode_length"] for r in rows]),
        "collision": np.mean([r["collision"] for r in rows]),
        "distance": np.mean([r["distance"] for r in rows]),
        "avg_speed": np.mean([r["avg_speed"] for r in rows]),
        "top_speed": np.mean([r["top_speed"] for r in rows]),
        "reward": np.mean([r["reward"] for r in rows]),
    }
    rows.append(agg)

    df = pd.DataFrame(rows)

    # ensure directory
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    if os.path.exists(xlsx_path):
        with pd.ExcelWriter(xlsx_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Metrics written to {xlsx_path} → sheet '{sheet_name}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Evaluate a model")
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="SAC", help="Algorithm to use")
    parser.add_argument("--config_path", type=str, default="config/sac_pc_reverse.yaml", help="Path to the config file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model to evaluate or continue training")
    parser.add_argument("--log_dir", type=str, default="results", help="Directory to save models and logs")
    parser.add_argument("--file_name", type=str, default="model", help="Name of the saved model file")
    # evaluation flags
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--xlsx_path", default="results/eval_metrics.xlsx")
    parser.add_argument("--sheet_name", default=None, help="override sheet name")
    parser.add_argument("--simulate", action="store_true", help="Simulate and evaluate trained model")
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    training_params = config.get("training",{})
    model_params = config.get("model", {})
    model_inputs = training_params['model_inputs']
    algorithm = {"PPO":PPO, "SAC":SAC}[args.algo]
    num_envs = training_params['num_envs']
    num_obstacles = training_params['num_obstacles']

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
            features_extractor_kwargs=dict(features_dim=128),
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
        vec_env = DummyVecEnv([make_env(0, num_obstacles, model_inputs)])
        print(f"Loading model from '{args.model_path}'")
        model = algorithm.load(args.model_path, env=vec_env)
        default_sheet = os.path.splitext(os.path.basename(args.model_path))[0]
        sheet_name = args.sheet_name or default_sheet
        evaluate(model,vec_env,model_inputs=model_inputs,num_obstacles=num_obstacles,sheet_name=sheet_name,n_episodes=args.eval_episodes,xlsx_path=args.xlsx_path)
    else:
        env = SubprocVecEnv([make_env(i, num_obstacles=num_obstacles, model_inputs=model_inputs) for i in range(num_envs)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        tb_dir = os.path.join(args.log_dir, "tensorboard", args.file_name)
        models_dir = os.path.join(args.log_dir, "models")
        logs_dir = os.path.join(args.log_dir, "logs")

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