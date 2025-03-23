from pathlib import Path
import numpy as np
import scipy
from PIL import Image
import os
import struct

from dm_control import composer
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mjcf

_ARENA_XML_PATH = '/home/mmkr/TestEnv/Car_LocoTransformer/Mujoco_learning/Pushr_car_simulation/heightfield_arena.xml'

class HeightFieldArena(composer.Arena):
    def __init__(self):
        super(HeightFieldArena, self).__init__()

    def _build(self, name='floor'):
        # Don't call super()._build here because we want our own XML file
        self._mjcf_root = mjcf.from_path(_ARENA_XML_PATH)
        self.size = 20
        if name:
            self._mjcf_root.model = name

        self._mjcf_root.option.gravity = (0, 0, -9.81)
        self._mjcf_root.asset.add('texture', name = 'grid', type='2d', builtin='checker', width=300,
                            height=300, rgb1=[.1, .2, .3], rgb2=[.2 , .3, .4])
        self._mjcf_root.asset.add('material', name='grid', texture='grid',
                            texrepeat=[8, 8], reflectance=.2)
        
        
        self._terrain_geom = self._mjcf_root.worldbody.add(
            'geom', name = 'floorplan', 
            type='plane', size=[self.size, self.size, .1], material='grid')

        
        self._mjcf_root.worldbody.add('camera',
                                      mode="fixed",
                                      pos="0 0 25",
                                      euler="0 0 0")
        
        self.walls = []
        # self.walls.append(
        # self._mjcf_root.worldbody.add('geom', name = "obs1",
        #                                 group = 2,
        #                                 type="sphere",
        #                                 mass="1200",
        #                                 contype="0",
        #                                 friction="0.4 0.005 0.00001",
        #                                 conaffinity="0",
        #                                 size="0.08",
        #                                 rgba=(0.8, 0.3, 0.3, 1),
        #                                 pos=(5,0,0.08)))
        # self.walls.append(
        # self._mjcf_root.worldbody.add('geom', name = "obs2",
        #                                 group = 2,
        #                                 type="sphere",
        #                                 mass="1.2",
        #                                 contype="0",
        #                                 friction="0.4 0.005 0.00001",
        #                                 conaffinity="0",
        #                                 size="0.08",
        #                                 rgba=(0.8, 0.3, 0.3, 1),
        #                                 pos=(0,5,0.08)))
        # self.walls.append(
        # self._mjcf_root.worldbody.add('geom', name = "obs3",
        #                                 group = 2,
        #                                 type="sphere",
        #                                 mass="1.2",
        #                                 contype="0",
        #                                 friction="0.4 0.005 0.00001",
        #                                 conaffinity="0",
        #                                 size="0.08",
        #                                 rgba=(0.8, 0.3, 0.3, 1),
        #                                 pos=(-5,0,0.08)))
        # self.walls.append(
        # self._mjcf_root.worldbody.add('geom', name = "obs4",
        #                                 group = 2,
        #                                 type="sphere",
        #                                 mass="1.2",
        #                                 contype="0",
        #                                 friction="0.4 0.005 0.00001",
        #                                 conaffinity="0",
        #                                 size="0.08",
        #                                 rgba=(0.8, 0.3, 0.3, 1),
        #                                 pos=(0,-5,0.08)))

        

    def height_lookup(self, pos):
        """Returns the height of the terrain at the given position."""
        return 0.08

    def in_bounds(self, pos):
        eps = 0.1
        return np.abs(pos).max() < self.size + eps