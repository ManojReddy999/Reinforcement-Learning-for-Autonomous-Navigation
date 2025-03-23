from dm_control import composer
from dm_control import mjcf
import os


class Sphere(composer.Entity):
    def _build(self, pos, name='sphere', size=[0.2] ,index=0):
        self._mjcf_root = mjcf.RootElement()
        geom_name = f'{name}_{index}'
        self._geom = self._mjcf_root.worldbody.add('geom',
                                      type="sphere",
                                      name=geom_name,
                                      mass="1.2",
                                      contype="1",
                                      friction="0.8 0.02 0.001",
                                      conaffinity="1",
                                      size=size,
                                      rgba=(0.8, 0.3, 0.3, 1),
                                      pos=pos+[0,0,size[0]])

    @property
    def mjcf_model(self):
        return self._mjcf_root

class Wall_x(composer.Entity):
    def _build(self, pos, name='box',size = [0.05,10,0.8], index=0):
        self._mjcf_root = mjcf.RootElement()
        geom_name = f'{name}_{index}'
        self._geom = self._mjcf_root.worldbody.add('geom',
                                      type="box",
                                      name=geom_name,
                                      mass="1.2",
                                      contype="1",
                                      friction="0.4 0.005 0.00001",
                                      conaffinity="1",
                                      size=size,
                                      rgba=(0.8, 0.3, 0.3, 1),
                                      pos=pos+[0,0,size[2]])

    @property
    def mjcf_model(self):
        return self._mjcf_root
    
class Wall_y(composer.Entity):
    def _build(self, pos, name='box',size=[10,0.05,0.4], index=0):
        self._mjcf_root = mjcf.RootElement()
        geom_name = f'{name}_{index}'
        self._geom = self._mjcf_root.worldbody.add('geom',
                                      type="box",
                                      name=geom_name,
                                      mass="1.2",
                                      contype="1",
                                      friction="0.4 0.005 0.00001",
                                      conaffinity="1",
                                      size=size,
                                      rgba=(0.8, 0.3, 0.3, 1),
                                      pos=pos+[0,0,size[2]])

    @property
    def mjcf_model(self):
        return self._mjcf_root
    

class Cube(composer.Entity):
    def _build(self, pos, name='box', size=[0.2,0.2,0.2], index=0):
        self._mjcf_root = mjcf.RootElement()
        geom_name = f'{name}_{index}'
        self._geom = self._mjcf_root.worldbody.add('geom',
                                    type="box",
                                    name=geom_name,
                                    mass="1.2",
                                    contype="1",
                                    friction="0.4 0.005 0.00001",
                                    conaffinity="1",
                                    size=size,
                                    rgba=(0.8, 0.3, 0.3, 1),
                                    pos=pos+[0,0,size[2]])

    @property
    def mjcf_model(self):
        return self._mjcf_root
    

class Cylinder(composer.Entity):
    def _build(self, pos, name='box', size=[0.3,0.3], index=0):
        self._mjcf_root = mjcf.RootElement()
        geom_name = f'{name}_{index}'
        self._geom = self._mjcf_root.worldbody.add('geom',
                                    type="cylinder",
                                    name=geom_name,
                                    mass="1.2",
                                    contype="1",
                                    friction="0.4 0.005 0.00001",
                                    conaffinity="1",
                                    size=size,
                                    rgba=(0.8, 0.3, 0.3, 1),
                                    pos=pos+[0,0,size[1]])

    @property
    def mjcf_model(self):
        return self._mjcf_root
    

class Cuboid1(composer.Entity):
    def _build(self, pos, name='box', size=[0.4,0.1,0.2], index=0):
        self._mjcf_root = mjcf.RootElement()
        geom_name = f'{name}_{index}'
        self._geom = self._mjcf_root.worldbody.add('geom',
                                    type="box",
                                    name=geom_name,
                                    mass="1.2",
                                    contype="1",
                                    friction="0.4 0.005 0.00001",
                                    conaffinity="1",
                                    size=size,
                                    rgba=(0.8, 0.3, 0.3, 1),
                                    pos=pos+[0,0,size[2]])

    @property
    def mjcf_model(self):
        return self._mjcf_root
    

class Cuboid2(composer.Entity):
    def _build(self, pos, name='box', size=[0.1,0.4,0.2], index=0):
        self._mjcf_root = mjcf.RootElement()
        geom_name = f'{name}_{index}'
        self._geom = self._mjcf_root.worldbody.add('geom',
                                      type="box",
                                      name=geom_name,
                                      mass="1.2",
                                      contype="1",
                                      friction="0.4 0.005 0.00001",
                                      conaffinity="1",
                                      size=size,
                                      rgba=(0.8, 0.3, 0.3, 1),
                                      pos=pos+[0,0,size[2]])

    @property
    def mjcf_model(self):
        return self._mjcf_root