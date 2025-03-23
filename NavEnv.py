import numpy as np
import math
from gymnasium import Env
from gymnasium.spaces import Box, Dict

from dm_control import composer
from NavTask import Navigation
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
        car/realsense_camera, 
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
        self.task = Navigation(creature)
        # self.task = CarTask()
        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

        self.mj_state = self.original_env.reset()

        self.endTime = 1500

        self.timeElapsed = 0

        self.action_space = Box(low = np.array([-10, 0]),        # Changed the action space of throttle from 0to10 to -10to10
                                high = np.array([10, 10]),
                                shape=(2,), dtype=np.float32)
        

        self.state = np.array([20, 0, np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]]) 
        for i in range(2):
            self.state = np.concatenate((self.state, np.array([20, 0, np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]])))

        self.images = np.array(self.mj_state[3]['car/realsense_camera'])

        obs_space_low = np.array([-20, -20, 0, 0])
        obs_space_high = np.array([20, 20, 10, 10])

        for i in range(2):
            obs_space_low = np.concatenate((obs_space_low, np.array([-20, -20, 0, 0])))
            obs_space_high = np.concatenate((obs_space_high, np.array([20, 20, 10, 10])))
        
        self.observation_space = Dict(
            spaces={
                "vec": Box(low = obs_space_low, high = obs_space_high, shape = (12,), dtype=np.float32),
                "img": Box(0, 255, self.images.shape, dtype=np.uint8)
            }
        )
        
        # obs_space_low = np.array([-20, -20, 0, 0] * 3)  # Shape (12,)
        # obs_space_high = np.array([20, 20, 10, 10] * 3)  # Shape (12,)

        # # Create the observation space
        # self.observation_space = Box(low=obs_space_low, high=obs_space_high, dtype=np.float32)

        # self.images = self.getDepthMap()
        # self.num_images = 4
        # for i in range(self.num_images - 1):
        #     self.images = np.concatenate((self.images, self.getDepthMap()), axis=2)
        
        # self.images = np.transpose(self.images, (2, 0, 1))

        self.done = False
        # self.boxPos = self.task.goal_graph.current_goal
        self.boxPos = np.array(self.task._button_initial_pose[:2])
        self.init_goal = self.boxPos# self.boxPos # figure this out
        self.init_car = [0, 0, 0]

        print("Generating environment... Box - ")

    # def getDepthMap(self):
    #     return np.array(self.mj_state[3]['car/realsense_camera'])
    
    def step(self, action):

        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1
        check = self.checkComplete()

        reward = self.getReward()

        if check == 1:
            self.done = True
        elif check == 2:
            reward += 5000
            self.done = True
        elif check==3:
            reward -= 2000
            self.done = True
        
        state_obs = self.getCurrObs()
        info = {}

        # self.rewardAccumulated = reward
        self.rewardAccumulated += reward
        # print('accumulated:',self.rewardAccumulated)
        # print('dist:',self.goalDist())
        if self.done:
            print(self.timeElapsed, "Steps")
            print(check, "Check")
            print(self.rewardAccumulated)

        truncated = False # look into how to initialize this

        return state_obs, reward, self.done, truncated, info
    
    def getReward(self):

        reward = 0 # reward for taking a step without hitting obstacles
        # goal_reward = -1
        obstacle_reward = 0

        # alpha = 1
        # beta = 0.05

        # reward += alpha * np.exp(-beta * x)

        # # Reward depending on the yaw angle of the car
        pos = self.mj_state[3]['car/body_pose_2d']
        w, x, y, z = self.mj_state[3]['car/body_rotation']
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        car_pos = [pos[0], pos[1]]
        box_pos = self.boxPos # Change

        delta = box_pos - car_pos
        yaw_diff = math.atan2(delta[1], delta[0])

        y = abs(yaw_diff - yaw)
        x = self.goalDist()

        # # Constants
        alpha = 0.05
        beta = 8

        gamma = 1
        eta = 0.5
        
        reward += alpha * np.exp(-beta * x)
        reward += gamma * np.exp(-eta * y)
        # reward += -x-y

        reward += self.task._button._num_activated_steps
        self.task._button._num_activated_steps = 0

        return reward
    
    def getRotation(self, theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ])
    
    def getCurrObs(self):

        pos = self.mj_state[3]['car/body_pose_2d']
        w, x, y, z = self.mj_state[3]['car/body_rotation']
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        R = self.getRotation(yaw)

        car_pos = np.array([[pos[0]], 
                            [pos[1]]])
        box_pos = self.boxPos[:,np.newaxis]
        relative_pos = np.squeeze(R@(box_pos-car_pos))
        self.state = np.concatenate((np.array([relative_pos[0], relative_pos[1], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]]), self.state[:-4]))
        obs = self.state

        self.images = self.mj_state[3]['car/realsense_camera']
        # print(self.images.shape, np.sum(self.images[2]))
        # plt.imshow(self.images[2]/np.max(self.images[2]),cmap='viridis')
        # plt.imshow(self.images[2],cmap='viridis')
        # plt.show()

        return {"img": self.images, "vec": obs}
        # return {"vec": obs}
        # return {"img": self.images}
    
    def getResetObs(self):

        pos = self.mj_state[3]['car/body_pose_2d']
        w, x, y, z = self.mj_state[3]['car/body_rotation']
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        R = self.getRotation(yaw)
        car_pos = np.array([[pos[0]], 
                            [pos[1]]])
        box_pos = self.boxPos[:,np.newaxis]
        relative_pos = np.squeeze(R@(box_pos-car_pos))

        self.state = np.array([relative_pos[0], relative_pos[1], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]])

        for i in range(2):
            self.state = np.concatenate((self.state, np.array([relative_pos[0], relative_pos[1], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]])))
        obs = self.state

        self.images = self.mj_state[3]['car/realsense_camera']

        return {"img": self.images, "vec": obs}
        # return {"vec": obs}
        # return {"img": self.images}

    
    # def obstructed(self):
    #     pos = self.mj_state[3]['car/body_pose_2d']
    #     obs = np.array(self.task.getObstacles())
    #     for i in range(obs.shape[0]):
    #         dist = np.sqrt((pos[0]-obs[i][0])**2 + (pos[1]-obs[i][1])**2)
    #         if dist<0.1:
    #             return True
    #     return False

    def obstructed(self):
        if self.task._button2._is_activated or self.task._button3._is_activated:
            return True
        else:
            return False
    
    def goalReached(self):
        dist = self.goalDist()
        return dist <= 0.01

    # def goalReached(self):
    #     if self.task._button._is_activated:
    #         return True
    #     else:
    #         return False

    def goalDist(self):
        pos = self.mj_state[3]['car/body_pose_2d']

        dist = (pos[0]-self.boxPos[0])**2 + (pos[1]-self.boxPos[1])**2
        return np.sqrt(dist)
    
    def checkComplete(self):
        
        if self.timeElapsed >= self.endTime: return 1
        if self.goalReached(): return 2
        if self.obstructed(): return 3

        return 0
    
    # See whether we need goal and car pose
    # For now we don't
    # def generateRandomBox(self):
    #     #return np.array([10,0])
    #     goal_range = [2,10]
    #     quadrant = np.random.randint(0,2,2)
    #     if quadrant[0] == 0:
    #         quadrant[0] = -1
    #     if quadrant[1] == 0:
    #         quadrant[1] = -1
    #     return np.random.randint(goal_range[0], goal_range[1], 2) * quadrant

    def reset(self,seed=None,options=None):
        # print("Initial Box - ",self.boxPos)
        print("Initial Car - ", self.init_car)

        print("Final Distance - ", self.goalDist())
        print("Final Coordinates - ", self.mj_state[3]['car/body_pose_2d'])

        self.done = False
        

        self.mj_state = self.original_env.reset()
        # self.boxPos = np.array([10,10])
        self.rewardAccumulated = 0
        self.timeElapsed = 0
        observations = self.getResetObs()
        info = {}
        print("[RESET] Generating env...")
        return observations, info

    def render(self):
        pics = np.array(self.mj_state[3]['car/overhead_cam'])
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
  
        # Using resizeWindow() 
        cv2.resizeWindow("Resized_Window", 700, 700) 
        
        # Displaying the image 
        cv2.imshow("Resized_Window", pics) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        return

    def close(self):
        self.original_env.close()
        pass




