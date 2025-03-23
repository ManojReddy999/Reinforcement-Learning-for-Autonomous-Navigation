#@title The `PressWithSpecificForce` task
from dm_control import composer
from dm_control.composer.observation import observable
from heightfield_arena import HeightFieldArena
import numpy as np
from dm_control.utils import transformations

from obstacle import Obstacle,BoxObstacle1, BoxObstacle2

NUM_SUBSTEPS = 25

class Maze(composer.Task):

  def __init__(self, creature):
    self._creature = creature
    self._arena = HeightFieldArena()
    self._arena.add_free_entity(self._creature)
    self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

    
    self._arena.attach(BoxObstacle2([1,0.2,0]))
    # self._arena.attach(BoxObstacle2([5,2,0]))
    # self._arena.attach(BoxObstacle1([5.4,2.4,0]))
    # self._arena.attach(BoxObstacle1([5.4,2.8,0]))
    # self._arena.attach(BoxObstacle2([5.4,-2,0]))
    # self._arena.attach(BoxObstacle1([5.8,-2,0]))
    # Configure initial poses
    self._creature_initial_pose = (0, 0, 0)

    # Configure and enable observables
    self._creature.observables.enable_all()
    # if not include_camera:
    self._creature.observables.get_observable('realsense_camera').enabled = True

    self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

  @property
  def root_entity(self):
    return self._arena

#   @property
#   def task_observables(self):
#     return self._task_observables  

  def initialize_episode(self, physics, random_state):
    self._creature.set_pose(physics, position=self._creature_initial_pose)

  def detect_collisions(self,physics):
    collision = False
    # print('collision data:',physics.data.ncon)
    for i in range(physics.data.ncon):
        contact = physics.data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        if geom1>0 and geom2>0:
          collision = True
          break
    return collision

  def get_reward(self,physics):
    pass  
  
  def getObstacles(self):
    # obs_pos = self.position
    # for wall in self._arena.walls:
    #     obs_pos.append(wall.pos)
    return []



