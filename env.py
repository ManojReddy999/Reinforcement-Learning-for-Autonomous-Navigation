import numpy as np
import math
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from dm_control import composer
from task import Survive
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

MIN_STEERING = -0.38
MAX_STEERING = 0.38
MIN_THROTTLE = -1.0
MIN_FORWARD_THROTTLE = 0
MAX_THROTTLE = 3.0
HORIZON = 2048
COLLISION_PENALTY = 1e6

class CarEnv(Env):
    ALL_MODEL_INPUTS = ["reverse","velocity","point_cloud"]

    def __init__(self, num_obstacles=120, arena_size=8, model_inputs=ALL_MODEL_INPUTS, random_seed=None):

        self.model_inputs = model_inputs
        self.task = Survive(num_obstacles=num_obstacles, arena_size=arena_size, random_seed=random_seed)
        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
        self.mj_state = self.original_env.reset()
        self.timeElapsed = 0

        if "reverse" in self.model_inputs:
            self.action_space = Box(low = np.array([MIN_STEERING, MIN_THROTTLE]),     
                                    high = np.array([MAX_STEERING, MAX_THROTTLE]),
                                    shape=(2,), dtype=np.float32)
        else:
            self.action_space = Box(low = np.array([MIN_STEERING, MIN_FORWARD_THROTTLE]),     
                                    high = np.array([MAX_STEERING, MAX_THROTTLE]),
                                    shape=(2,), dtype=np.float32)

        # Define Observation Space
        vec_space_low = []
        vec_space_high = []

        if "velocity" in self.model_inputs:
            vec_space_low += [MIN_THROTTLE, MIN_THROTTLE]
            vec_space_high += [MAX_THROTTLE, MAX_THROTTLE]

        vec_space = Box(low = np.array(vec_space_low), high = np.array(vec_space_high), dtype=np.float32) if len(vec_space_low) > 0 else None

        if "depth" in self.model_inputs:
            shape = self.mj_state.observation['car/realsense_camera'].shape
        elif "point_cloud" in self.model_inputs:
            shape = self.mj_state.observation['car/compute_point_cloud'].shape

        cam_space = Box(low = 0, high = np.inf, shape = shape, dtype=np.float32) if shape is not None else None

        spaces = {}
        if vec_space is not None:
            spaces["vec"] = vec_space
        if cam_space is not None:
            spaces["cam"] = cam_space

        self.observation_space = Dict(spaces)
        
        self.dist = 0 
        self.direction = 0
        self.done = False
        self.init_car = self.task._agent.get_pose(self.original_env.physics)[0]
        self.last_pos = self.init_car
   
    def step(self, action):
        self.task._agent.apply_action(self.original_env.physics, action, None)
        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1
        check = self.checkComplete()
        reward = self.getReward()

        if check == 1 or check == 3:
            self.done = True
        
        state_obs = self.get_observations()
        self.update_dist()
        info = {}

        self.rewardAccumulated += reward

        if self.done:
            print(self.timeElapsed, "steps")
            print(check, "Check")
            print(self.rewardAccumulated)

        truncated = False 

        return state_obs, reward, self.done, truncated, info
    
    def getReward(self):

        wheel_speeds = self.mj_state.observation['car/wheel_speeds']
        Speed = np.linalg.norm(self.mj_state.observation['car/body_vel_2d'])
        curr_direction = 0

        if wheel_speeds[2]<0:
            curr_direction -= 1
            reward = -1*np.exp(5*Speed)
        elif wheel_speeds[2]>0:
            reward = np.exp(10*Speed)
            curr_direction += 1
        else:
            reward = -10

        # Apply penalty only when switching between forward & reverse
        # if curr_direction*self.direction<0:
        #     reward -= 0.5  # penalty for switching direction
        # self.direction = curr_direction

        if self.obstructed():
            reward -= COLLISION_PENALTY

        return reward
    
    def get_observations(self):

        combined_obs = {}

        if "velocity" in self.model_inputs:
            vec_obs = self.mj_state.observation['car/body_vel_2d']
            combined_obs["vec"] = vec_obs
        if "point_cloud" in self.model_inputs:
            cam_obs = self.mj_state.observation['car/compute_point_cloud']
            combined_obs["cam"] = cam_obs
        elif "depth" in self.model_inputs:
            cam_obs = self.mj_state.observation['car/realsense_camera']
            combined_obs["cam"] = cam_obs

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.pc)
        # o3d.visualization.draw_geometries([pcd])
        return combined_obs
    
    def obstructed(self):

        if self.task.detect_collisions(self.original_env.physics):
            return True
        return False

    def update_dist(self):

        pos = self.mj_state.observation['car/body_pose_2d']
        self.dist += np.linalg.norm(pos - self.last_pos)
        self.last_pos = pos
        return 
    
    def checkComplete(self):
        
        if self.timeElapsed >= HORIZON: return 1
        if self.task.detect_collisions(self.original_env.physics): return 3
        return 0

    def reset(self,seed=None,options=None):
        
        print("Dist", self.dist)

        self.done = False
        self.mj_state = self.original_env.reset()
        self.rewardAccumulated = 0
        self.timeElapsed = 0
        self.dist = 0 
        self.direction = 0
        observations = self.get_observations()
        info = {}
        # print("[RESET] Generating env...")
        return observations, info

    def render(self):
        pass    

    def close(self):
        self.original_env.close()





