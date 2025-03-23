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

import matplotlib.pyplot as plt
import numpy as np

# from callback.trainingCallback import SaveOnBestTrainingRewardCallback
from feat_extrac.feature_extractor import mlp_img_state
from feat_extrac.customFcn import CustomExtractor
from networks.policy import StateDepthPolicy
# from env import CarEnv
from ButtonEnv import CarEnv
from dm_control.mjcf.physics import Physics
from dm_control import viewer
from task import CarTask

def main(args):

    if not args.eval:

        def make_env():
            return CarEnv()
        tb = os.path.join("./Mujoco_learning/Pushr_car_simulation/results/tensor/"+args.file)
        models = os.path.join("./Mujoco_learning/Pushr_car_simulation/results/models/"+args.file)
        stats_path = os.path.join("./Mujoco_learning/Pushr_car_simulation/results/logs/")
        
        num_envs = 4
        env = make_env()
        env = Monitor(env, stats_path)
        env.reset()
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        # custom_policy = dict(
        #     net_arch = dict(pi = [8, 4], vf = [8, 4]),
        #     activation_fn = torch.nn.ReLU,
        #     features_extractor_class=CustomExtractor,
        #     log_std_init = 0.01
        # )

        model = PPO(
            # policy=StateDepthPolicy,
            policy='MultiInputPolicy',
            n_steps = 1500,
            learning_rate=3e-5,
            n_epochs=50,
            env=env,
            batch_size=300,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=1.5,
            # policy_kwargs=custom_policy,
            verbose=0,
            tensorboard_log=tb,
            seed=1
        )

        model.learn(total_timesteps=300000)
        print("Saving end model")
        model.save(models)
        env.save(stats_path+args.file+".pkl")
    
    else:
        env=CarEnv()
        # env=Monitor(env)
        # env=DummyVecEnv([lambda: env])
        # stats_path=os.path.join("./Mujoco_learning/Pushr_car_simulation/results/logs/")
        # env=VecNormalize.load(stats_path + args.file + ".pkl", env)
        # env.training=False
        # env.norm_reward=False

        models=os.path.join("./Mujoco_learning/Pushr_car_simulation/results/models/"+args.file)
        saved_model_path=os.path.join(models+'.zip')
        model=PPO.load(saved_model_path, env=env)

        state=env.reset()
        def random_policy(time_step):
            nonlocal model
            nonlocal state
            nonlocal env
            action, _=model.predict(state)
            state, reward, done, info=env.step(action)

            return action

        viewer.launch(env.original_env, policy=random_policy)

        # episodes=1
        # actions=[]

        # for episode in range(1, episodes+1):
        #     done=False
        #     score=0
        #     state=env.reset()
        #     start=time.time()
        #     while not done:
        #         action, _=model.predict(state)
        #         actions.append([action[0], action[1]])
        #         state, reward, done, info=env.step(action)
        #         score+=reward
        #     print('Episode:{} Score:{}'.format(episode,score))
        # env.close()

        # actions = np.array(actions)
        # fig, ax = plt.subplots(2)
        # ax[0].plot(range(actions.shape[0]), actions[:, 0])
        # ax[1].plot(range(actions.shape[0]), actions[:, 1])
        # fig.set_size_inches(15, 10)
        # plt.savefig("state.png")
        # plt.close()
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="mj1")
    parser.add_argument('--eval', type=bool, default=True, help='Specify if running in evaluation mode.')
    args = parser.parse_args()
    main(args=args)