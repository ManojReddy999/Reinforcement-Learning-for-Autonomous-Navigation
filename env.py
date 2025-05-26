import numpy as np
import math
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from dm_control import composer
from task import Survive

MIN_STEERING = -0.38
MAX_STEERING =  0.38
MIN_THROTTLE = -1.0
MIN_FORWARD_THROTTLE = 0
MAX_THROTTLE =  3.0
HORIZON = 1000

class CarEnv(Env):
    # Added "history" to the list of possible model inputs
    ALL_MODEL_INPUTS = ["reverse", "velocity", "depth", "history"]

    def __init__(self, num_obstacles=200, arena_size=8, model_inputs=ALL_MODEL_INPUTS, random_seed=None):

        self.model_inputs = model_inputs
        self.task = Survive(num_obstacles=num_obstacles, arena_size=arena_size, random_seed=random_seed)
        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

        if "history" in self.model_inputs:
            self.vel_history = np.zeros((5, 2), dtype=np.float32)

        self.mj_state = self.original_env.reset()
        self.timeElapsed = 0

        # ======= ACTION SPACE =======
        if "reverse" in self.model_inputs:
            # steering in [-0.38, 0.38], throttle in [-1, 3]
            self.action_space = Box(
                low=np.array([MIN_STEERING, MIN_THROTTLE], dtype=np.float32),
                high=np.array([MAX_STEERING, MAX_THROTTLE], dtype=np.float32),
            )
        else:
            # No reverse => throttle in [0, 3]
            self.action_space = Box(
                low=np.array([MIN_STEERING, MIN_FORWARD_THROTTLE], dtype=np.float32),
                high=np.array([MAX_STEERING, MAX_THROTTLE], dtype=np.float32),
            )

        # ======= OBSERVATION SPACE =======
        # We'll build vector space (velocity) and camera space depending on inputs
        vec_space = None
        spaces_dict = {}

        # Velocity space
        if "velocity" in self.model_inputs:
            if "history" in self.model_inputs:
                # 5 frames of velocity => shape (10,)
                vec_low  = np.array([MIN_THROTTLE]*10, dtype=np.float32)
                vec_high = np.array([MAX_THROTTLE]*10, dtype=np.float32)
            else:
                # Single-step velocity => shape (2,)
                vec_low  = np.array([MIN_THROTTLE, MIN_THROTTLE], dtype=np.float32)
                vec_high = np.array([MAX_THROTTLE, MAX_THROTTLE], dtype=np.float32)
            vec_space = Box(low=vec_low, high=vec_high, dtype=np.float32)
            spaces_dict["vec"] = vec_space

        # Camera or point cloud shape
        cam_space = None
        if "point_cloud" in self.model_inputs:
            shape = self.mj_state.observation['car/compute_point_cloud'].shape
            cam_space = Box(low=0, high=np.inf, shape=shape, dtype=np.float32)
            spaces_dict["cam"] = cam_space
        elif "depth" in self.model_inputs:
            # Not requested, but typical usage if you also had "depth"
            shape = self.mj_state.observation['car/realsense_camera'].shape
            cam_space = Box(low=0, high=np.inf, shape=shape, dtype=np.float32)
            spaces_dict["cam"] = cam_space

        self.observation_space = Dict(spaces_dict)

        # Some internal bookkeeping
        self.dist = 0.0
        self.direction = 0
        self.done = False
        self.init_car = self.task._car.get_pose(self.original_env.physics)[0]
        self.last_pos = self.init_car[:2]
        self.rewardAccumulated = 0.0

    def step(self, action):
        # Apply action to the agent
        # self.task._car.apply_action(self.original_env.physics, action, None)
        # Step the environment
        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1

        # Possibly update velocity history if using "history"
        # if "history" in self.model_inputs and "velocity" in self.model_inputs:
        #     current_vel = self.mj_state.observation['car/body_vel_2d']
        #     # Shift everything up by one step
        #     self.vel_history[:-1] = self.vel_history[1:]
        #     self.vel_history[-1]  = current_vel

        reward = self.mj_state.reward
        collision = self.task.detect_collisions(self.original_env.physics)
        check = self.checkComplete(collision)
        # if check == 1:
        #     truncated = True
        # elif check == 3:
        #     self.done = True
        terminated = collision
        truncated = self.timeElapsed >= HORIZON

        obs = self.get_observations()
        self.update_dist()
        info = {}

        self.rewardAccumulated += reward
        if terminated or truncated:
            print(self.timeElapsed, "steps")
            print("Check:", check)
            print("Total reward:", self.rewardAccumulated)

        return obs, reward, terminated, truncated, info

    def get_observations(self):
        """Assemble the observation dict based on model_inputs."""
        obs = {}

        if "velocity" in self.model_inputs:
            if "history" in self.model_inputs:
                # Return velocity history as shape (10,)
                current_vel = self.mj_state.observation['car/body_vel_2d']
                # Shift everything up by one step
                self.vel_history[:-1] = self.vel_history[1:]
                self.vel_history[-1]  = current_vel
                obs["vec"] = self.vel_history.flatten()
                # print(obs["vec"].shape)
            else:
                # Single-step velocity as shape (2,)
                obs["vec"] = self.mj_state.observation['car/body_vel_2d']

        if "point_cloud" in self.model_inputs:
            obs["cam"] = self.mj_state.observation['car/compute_point_cloud']
        elif "depth" in self.model_inputs:
            obs["cam"] = self.mj_state.observation['car/realsense_camera']

        return obs

    def obstructed(self):
        return self.task.detect_collisions(self.original_env.physics)

    def update_dist(self):
        pos = self.mj_state.observation['car/body_pose_2d'][:2]
        self.dist += np.linalg.norm(pos - self.last_pos)
        self.last_pos = pos

    def checkComplete(self, collision):
        if self.timeElapsed >= HORIZON:
            return 1
        if collision:
            return 3
        return 0

    def reset(self, seed=None, options=None):
        print("Dist traveled before reset:", self.dist)
        self.done = False
        self.mj_state = self.original_env.reset()
        self.rewardAccumulated = 0.0
        self.timeElapsed = 0
        self.dist = 0
        self.direction = 0

        # Reset velocity history if using it
        if "history" in self.model_inputs and "velocity" in self.model_inputs:
            current_vel = self.mj_state.observation['car/body_vel_2d']
            self.vel_history = np.tile(current_vel, (5,1))

        obs = self.get_observations()
        info = {}
        return obs, info

    def render(self):
        pass

    def close(self):
        self.original_env.close()