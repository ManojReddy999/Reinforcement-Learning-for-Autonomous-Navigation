import numpy as np
import math
from gymnasium import Env
from gymnasium.spaces import Box, Dict

from dm_control import composer
from SurvivalTask import Survive
from obstacle import Obstacle
from MazeTask import Maze
import cv2
import car
import matplotlib.pyplot as plt

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

        creature = car.Car()
        self.task = Survive(creature,30)

        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
    
        self.mj_state = self.original_env.reset()

        self.endTime = 1500

        self.timeElapsed = 0

        self.action_space = Box(low = np.array([-1, 0]),     
                                high = np.array([1, 3]),
                                shape=(2,), dtype=np.float32)
        

        self.state = np.concatenate([np.array([0, 0, 0]), self.mj_state[3]['car/body_vel_2d'], self.mj_state[3]['car/steering_pos']]) 
        for i in range(3):
            self.state = np.concatenate((self.state, np.concatenate([np.array([0, 0, 0]) ,self.mj_state[3]['car/body_vel_2d'], self.mj_state[3]['car/steering_pos']])))

        self.images = np.array(self.mj_state[3]['car/realsense_camera']).astype(np.float32)    

        obs_space_low = np.array([-8, -8, -np.pi, -np.inf, -np.inf, -np.inf])
        obs_space_high = np.array([8, 8, np.pi, np.inf, np.inf, np.inf])

        for i in range(3):
            obs_space_low = np.concatenate((obs_space_low, np.array([-8, -8, -np.pi, -np.inf, -np.inf, -np.inf])))
            obs_space_high = np.concatenate((obs_space_high, np.array([8, 8, np.pi, np.inf, np.inf, np.inf])))
        
        self.observation_space = Dict(
            spaces={
                "vec": Box(low = obs_space_low, high = obs_space_high, shape = (len(obs_space_high),), dtype=np.float32),
                "img": Box(0, np.inf, self.images.shape, dtype=np.float32)
            }
        )
        
        self.done = False
        self.init_car = [0, 0, 0]
        self.time_penalty = 0
        self.action_reward = 0
        print("Generating environment... Box - ")

    
    def step(self, action):
        self.task._creature.apply_action(self.original_env.physics, action, None)
        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1
        check = self.checkComplete()

        # print(self.mj_state.observation['car/steering_pos'])

        reward = self.getReward(action)

        if check == 1:
            self.done = True
        # elif check == 2:
        #     reward += 5000
        #     self.done = True
        elif check==3:
            # reward -= 100
            self.done = True
        elif check==4:
            # reward -= 100
            self.done = True
        
        state_obs = self.getCurrObs()
        info = {}

        self.rewardAccumulated += reward

        if self.done:
            print(self.timeElapsed, "Steps")
            print(check, "Check")
            print(self.rewardAccumulated)

        truncated = False 

        return state_obs, reward, self.done, truncated, info
    
    def getReward(self,action):

        wheel_speeds = self.mj_state[3]['car/wheel_speeds']
        Speed = np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2))

        if wheel_speeds[2]<0:
            Vel = -1*Speed
        else:
            Vel = Speed

        # depth_count = 0
        depth = np.min(self.images[1][:64])

        # if depth>0.5:
        #     depth = 0.5
        # if depth < 3:
        #     depth_count = np.sum(self.images[1]==depth)

        # VelDepth = depth - 0.1*Vel
        # reward = 5*Vel*np.exp(-(((VelDepth-0.2)/3)**2))
        
        # if Speed >2:
        #     reward = 10*Vel
        # elif Speed >1:
        #     reward = 5*Vel
        # else:
        #     reward = Vel
        reward = 5*Vel - 1/(depth+1e-3)

        if self.obstructed():
            reward -= 1000

        if self.out_of_bounds():
            reward -= 500

        return reward
    
    def getRotation(self, theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ])
    
    def getCurrObs(self):

        pos = self.mj_state[3]['car/body_pose_2d']
        curr_state = np.concatenate([pos, self.mj_state[3]['car/body_vel_2d'], self.mj_state[3]['car/steering_pos']])

        self.state = np.concatenate((self.state[:-6],curr_state))
        obs = self.state

        self.images = self.mj_state[3]['car/realsense_camera']
        print(self.images.shape)

        # plt.imshow(self.images[0],cmap='gray')
        # plt.show()
        # print(np.min(self.images[0][60,110]))
        # plt.imshow(self.images[1],cmap='gray')
        # plt.show()
        # print(np.min(self.images[1]))
        depth_image = self.images[2]
        depth_image_normalized = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        plt.imshow(depth_image_normalized, cmap='plasma')
        plt.show()

        # print(depth_image_normalized[150,120])

        return {"img": self.images, "vec": obs}
    
    def getResetObs(self):

        pos = self.mj_state[3]['car/body_pose_2d']
        # car_pos = np.array([[pos[0]], 
        #                     [pos[1]]])

        self.state = np.concatenate([pos, self.mj_state[3]['car/body_vel_2d'], self.mj_state[3]['car/steering_pos']])

        for i in range(3):
            self.state = np.concatenate((self.state, np.concatenate([pos, self.mj_state[3]['car/body_vel_2d'], self.mj_state[3]['car/steering_pos']])))

        obs = self.state

        self.images = self.mj_state[3]['car/realsense_camera']

        return {"img": self.images, "vec": obs}

    
    def obstructed(self):

        collision = self.task.detect_collisions(self.original_env.physics)
        if collision:
            return True
        return False
    
    def out_of_bounds(self):
        pos = self.mj_state[3]['car/body_pose_2d']
        if (pos[0]<8 and pos[0]>-8) and (pos[1]<8 and pos[1]>-8):
            pass
        else:
            return True

    def Dist(self):
        pos = self.mj_state[3]['car/body_pose_2d']

        dist = (pos[0]-self.init_car[0])**2 + (pos[1]-self.init_car[1])**2
        return np.sqrt(dist)
    
    def checkComplete(self):
        
        if self.timeElapsed >= self.endTime: return 1
        # if self.goalReached(): return 2
        if self.obstructed(): return 3
        if self.out_of_bounds(): return 4

        return 0

    def reset(self,seed=None,options=None):
        
        print("Initial Car - ", self.init_car)

        print("Final Distance - ", self.Dist())
        print("Final Coordinates - ", self.mj_state[3]['car/body_pose_2d'])

        self.done = False

        self.mj_state = self.original_env.reset()

        self.rewardAccumulated = 0
        self.timeElapsed = 0
        observations = self.getResetObs()
        info = {}
        print("[RESET] Generating env...")
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





