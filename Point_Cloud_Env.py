import numpy as np
import math
from gymnasium import Env
from gymnasium.spaces import Box, Dict

from dm_control import composer
from SurvivalTask import Survive
import cv2
import car
import matplotlib.pyplot as plt
import open3d as o3d

'''
* Things required for a stable baselines env to work:
    - functions which are required are listed below - __init__, step, reset, render, close (these are for framework)
    - For the network, we need observations (images and states), and means to provide actions
    - additional functions - goalreached along with goal distance, obstructed, checkcomplete, reward, goalpos, carpos, getResetObs, getCurrObs

{

    'car/body_pose_2d': array([ 2.45987755e-01,  1.19822533e-06, -1.90241757e-04]), 
    'car/body_position': array([2.45987755e-01, 1.19822533e-06, 4.00422006e-04]), 
    'car/body_rotation': array([ 9.99999994e-01, -6.15834970e-06, -5.92609224e-05, -9.51205127e-05]), 
    'car/body_rotation_matrix': 
    array([ 9.99999975e-01,  1.90241754e-04, -1.18520672e-04,  0.00000000e+00,
       -1.90240294e-04,  9.99999982e-01,  1.23279732e-05,  0.00000000e+00,
        1.18523016e-04, -1.23054255e-05,  9.99999993e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]), 
        'car/body_vel_2d': array([ 7.36842529e-01, -2.09264098e-04]), 
        car/realsense_camera, [:, :128, :, :]
        'car/sensors_acc': array([ 0.92276181,  0.01851117, 10.11273594]), 
        'car/sensors_gyro': array([ 0.00377797, -0.00224604,  0.00382539]), 
        'car/sensors_vel': array([ 7.36844215e-01, -6.92586860e-05,  1.39557987e-02]), 
        'car/steering_pos': array([0.00091535]), 
        'car/steering_vel': array([-3.60766821e-05]), 
        'car/wheel_speeds': array([14.98710914, 14.9969212 , 14.98587148, 15.00220248])
        
}

'''

class CarEnv(Env):

    def __init__(self):

        self.creature = car.Car()
        self.task = Survive(self.creature,120,arena_size=8)

        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
    
        self.mj_state = self.original_env.reset()

        self.endTime = 2048

        self.timeElapsed = 0

        self.action_space = Box(low = np.array([-0.38, -1]),     
                                high = np.array([0.38, 3]),
                                shape=(2,), dtype=np.float32)

        self.pc = self.mj_state[3]['car/compute_point_cloud']  
        self.dist = 0 
        self.direction = 0
        
        # obs_space_low = np.array([-np.inf])
        # obs_space_high = np.array([np.inf])

        self.observation_space = Dict(
            spaces={
                # "vec": Box(low = obs_space_low, high = obs_space_high, shape = (len(obs_space_high),), dtype=np.float32),
                "pc": Box(-np.inf, np.inf, self.pc.shape, dtype=np.float32)
            }
        )
        
        self.done = False
        self.init_car = self.creature.get_pose(self.original_env.physics)[0]
        self.last_pos = self.init_car
        
        # print("Generating environment... Box - ")

    
    def step(self, action):
        self.task._creature.apply_action(self.original_env.physics, action, None)
        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1
        check = self.checkComplete()

        reward = self.getReward()

        if check == 1:
            self.done = True
        elif check==3:
            self.done = True
        
        state_obs = self.getCurrObs()
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

        wheel_speeds = self.mj_state[3]['car/wheel_speeds']
        Speed = np.linalg.norm(self.mj_state[3]['car/body_vel_2d'])
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
            reward -= 1000000

        return reward
    
    def getRotation(self, theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ])
    
    def getCurrObs(self):

        self.pc = self.mj_state[3]['car/compute_point_cloud']
        
        # self.vec = np.array([np.linalg.norm(self.mj_state[3]['car/body_vel_2d'])])
        # print(self.pc.shape) 

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.pc)
        # o3d.visualization.draw_geometries([pcd])
        # return {"vec": self.vec, "pc": self.pc}
        return {"pc": self.pc}
    
    def getResetObs(self):

        self.pc = self.mj_state[3]['car/compute_point_cloud']
        # self.vec = np.array([np.linalg.norm(self.mj_state[3]['car/body_vel_2d'])])
        # print(self.pc.shape) 
        # return {"vec": self.vec, "pc": self.pc}
        return {"pc": self.pc}
    
    def obstructed(self):

        if self.task.detect_collisions(self.original_env.physics):
            return True
        return False
    
    def out_of_bounds(self):
        pos = self.mj_state[3]['car/body_pose_2d']
        if (pos[0]<8 and pos[0]>-8) and (pos[1]<8 and pos[1]>-8):
            pass
        else:
            return True

    def update_dist(self):
        pos = self.mj_state[3]['car/body_pose_2d']

        self.dist += np.linalg.norm(pos - self.last_pos)
        self.last_pos = pos
        return 
    
    def checkComplete(self):
        
        if self.timeElapsed >= self.endTime: return 1
        # if self.goalReached(): return 2
        if self.obstructed(): return 3
        if self.out_of_bounds(): return 4

        return 0

    def reset(self,seed=None,options=None):
        
        # print("Initial Car - ", self.init_car)

        print("Dist", self.dist)
        # print("Final Coordinates - ", self.mj_state[3]['car/body_pose_2d'])

        self.done = False

        self.mj_state = self.original_env.reset()

        self.rewardAccumulated = 0
        self.timeElapsed = 0
        self.dist = 0 
        self.direction = 0
        observations = self.getResetObs()
        info = {}
        # print("[RESET] Generating env...")
        return observations, info

    def render(self):
        pics = np.array(self.mj_state[3]['car/overhead_cam'])
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
  
        cv2.resizeWindow("Resized_Window", 700, 700) 
        
        # Displaying the image 
        cv2.imshow("Resized_Window", pics) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        return

    def close(self):
        self.original_env.close()
        pass





