from pathlib import Path
import numpy as np
import scipy
from PIL import Image
import os
import struct

from dm_control import composer
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mjcf


class CheckerArena(composer.Arena):
    def __init__(self):
        super(CheckerArena, self).__init__()

    def _build(self):
        self._mjcf_root = mjcf.RootElement()
        self._mjcf_root.asset.add('texture', name = 'grid', type='2d', builtin='checker', width=300,
                            height=300, rgb1=[.1, .2, .3], rgb2=[.2 , .3, .4])
        self._mjcf_root.asset.add('material', name='grid', texture='grid',
                            texrepeat=[8, 8], reflectance=.2)
        self._mjcf_root.worldbody.add('geom', type='plane', size=[8, 8, .1], material='grid')
        self._mjcf_root.worldbody.add('light', ambient=[0.5, 0.5, 0.5],  pos=[0, 0, 10], dir=[0, 0, -1])


        # self._mjcf_root.option.gravity = (0, 0, -9.81)
        # self.num_obstacles = 20
        # self.obstacle_pos = self.generateRandomObstacles(self.num_obstacles)
        # self.walls = []
        # for i in range(self.num_obstacles):
        #     self.walls.append(
        #         self._mjcf_root.worldbody.add('geom',
        #                                 type="sphere",
        #                                 mass="1.2",
        #                                 contype="1",
        #                                 friction="0.4 0.005 0.00001",
        #                                 conaffinity="1",
        #                                 size="0.08",
        #                                 rgba=(0.8, 0.3, 0.3, 1),
        #                                 pos=[self.obstacle_pos[i][0], self.obstacle_pos[i][1], .1]))
        
        # self.goal_pos = self.generateRandomObstacles(1)
        # self.goal_reprs = [
        #     self._mjcf_root.worldbody.add('site',
        #                             type="sphere",
        #                             size="0.08",
        #                             rgba=(0.0, 1.0, 0.0, 0.5),
        #                             pos=[self.goal_pos[0][0], self.goal_pos[0][1], .1])
        # ]
            
    
    def generateRandomObstacles(self, n):
        space_range = [-self.size_val,self.size_val]
        obs = []
        for _ in range(n):
            quadrant = np.random.randint(0,2,2)
            if quadrant[0] == 0:
                quadrant[0] = -1
            if quadrant[1] == 0:
                quadrant[1] = -1
            obs.append(np.random.randint(space_range[0], space_range[1], 2) * quadrant)
        return obs




        