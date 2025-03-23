import numpy as np
import math

from gym import Env
from gym.spaces import Box, Dict

from task import CarTask
from dm_control import composer

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

        self.task = CarTask()
        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

        self.mj_state = self.original_env.reset()

        self.endTime = 1500

        self.timeElapsed = 0

        self.action_space = Box(low = np.array([-0.7, 0]),
                                high = np.array([0.7, 10]),
                                shape=(2,), dtype=np.float32)
        

        self.state = np.array([20, 0, np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]]) 
        for i in range(2):
            self.state = np.concatenate((self.state, np.array([20, 0, np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]])))

        self.images = np.array(self.mj_state[3]['car/realsense_camera'])

        obs_space_low = np.array([-20, -20, 0, -0.7])
        obs_space_high = np.array([20, 20, 10, 0.7])

        for i in range(2):
            obs_space_low = np.concatenate((obs_space_low, np.array([-20, -20, 0, -0.7])))
            obs_space_high = np.concatenate((obs_space_high, np.array([20, 20, 10, 0.7])))
        
        self.observation_space = Dict(
            spaces={
                "vec": Box(low = obs_space_low, high = obs_space_high, shape = (12,), dtype=np.float32),
                "img": Box(0, 255, self.images.shape, dtype=np.uint8),
            }
        )

        self.done = False
        self.init_goal = [10, 0] # self.boxPos # figure this out
        self.init_car = [0, 0, 0]
        print("Generating environment... Box - ")

    def step(self, action):

        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1
        check = self.checkComplete()

        # For now use the reward from here
        # Next time try to use the one from mujoco and see difference

        reward = self.getReward()

        if check == 1:
            self.done = True
        elif check == 2:
            reward += 2000
            self.done = True
        elif check==3:
            reward -= 1000
            self.done = True
        
        state_obs = self.getCurrObs()
        info = {}
        self.rewardAccumulated += reward

        if self.done:
            print(self.timeElapsed, "Steps")
            print(check, "Check")
            print(self.rewardAccumulated)


        return state_obs, reward, self.done, info
    
    def getReward(self):

        reward = -1.0

        # Distance
        x = self.goalDist()

        alpha = 1
        beta = 0.05

        reward += alpha * np.exp(-beta * x)

        # # Reward depending on the yaw angle of the car
        # pos, orn = self.getCarPose() # Change
        # yaw = orn[2]

        # car_pos = [pos[0], pos[1]]
        # box_pos = self.boxPos # Change

        # delta = box_pos - car_pos
        # yaw_diff = math.atan2(delta[1], delta[0])

        # y = abs(yaw_diff - yaw)

        # # Constants
        # alpha = config["env"]["dist_rew_cons"][0]
        # beta = config["env"]["dist_rew_cons"][1]

        # gamma = config["env"]["dir_rew_cons"][0]
        # eta = config["env"]["dir_rew_cons"][1]
        
        # reward += alpha * np.exp(-beta * x)
        # reward += gamma * np.exp(-eta * y / x)

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
        box_pos = np.array([[5], [0]])
        relative_pos = np.squeeze(R@(box_pos-car_pos))
        self.state = np.concatenate((np.array([relative_pos[0], relative_pos[1], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]]), self.state[:-4]))
        obs = self.state

        self.images = self.mj_state[3]['car/realsense_camera']


        return {"img": self.images, "vec": obs}
    
    def getResetObs(self):

        pos = self.mj_state[3]['car/body_pose_2d']
        w, x, y, z = self.mj_state[3]['car/body_rotation']
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        R = self.getRotation(yaw)
        car_pos = np.array([[pos[0]], 
                            [pos[1]]])
        box_pos = np.array([[5], [0]])
        relative_pos = np.squeeze(R@(box_pos-car_pos))

        self.state = np.array([relative_pos[0], relative_pos[1], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]])

        for i in range(2):
            self.state = np.concatenate((self.state, np.array([relative_pos[0], relative_pos[1], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]])))
        obs = self.state

        self.mj_state[3]['car/realsense_camera']
        
        return {"img": self.images, "vec": obs}

    
    def obstructed(self):
        pos = self.mj_state[3]['car/body_pose_2d']
        obs = np.array(self.task.getObstacles())
        for i in range(obs.shape[0]):
            dist = np.sqrt((pos[0]-obs[i][0])**2 + (pos[1]-obs[i][1])**2)
            if dist<0.1:
                return True
        return False
    
    def goalReached(self):
        dist = self.goalDist()
        return dist <= 1

    def goalDist(self):
        pos = self.mj_state[3]['car/body_pose_2d']
        dist = (pos[0]-5)**2 + (pos[1]-0)**2
        return np.sqrt(dist)
    
    def checkComplete(self):
        
        if self.timeElapsed >= self.endTime: return 1
        if self.goalReached(): return 2
        if self.obstructed(): return 3

        return 0
    
    # See whether we need goal and car pose
    # For now we don't
        

    def reset(self):
        print("Initial Box - [5, 0]")
        print("Initial Car - ", self.init_car)

        print("Final Distance - ", self.goalDist())
        print("Final Coordinates - ", self.mj_state[3]['car/body_pose_2d'])

        self.done = False

        self.mj_state = self.original_env.reset()
        self.rewardAccumulated = 0
        self.timeElapsed = 0
        observations = self.getResetObs()
        print("[RESET] Generating env...")
        return observations

    def render(self):
        
        pass

    def close(self):
        self.original_env.close()
        pass