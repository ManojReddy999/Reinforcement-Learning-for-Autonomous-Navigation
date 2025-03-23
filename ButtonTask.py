#@title The `PressWithSpecificForce` task
from dm_control import composer
from dm_control.composer.observation import observable
from Goal_Button import GoalButton, UniformCircle
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from heightfield_arena import HeightFieldArena
import numpy as np
from dm_control.utils import transformations

NUM_SUBSTEPS = 25

class PressWithSpecificForce(composer.Task):

  def __init__(self, creature):
    self._creature = creature
    self._arena = HeightFieldArena()
    self._arena.add_free_entity(self._creature)
    self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))
    self._button = GoalButton()
    self._arena.attach(self._button)

    # Configure initial poses
    self._creature_initial_pose = (0, 0, 0)
    # button_distance = distributions.Uniform(4, 8)             # Use to randomize the initialization
    # self._button_initial_pose = UniformCircle(button_distance)
    self._button_initial_pose = (3,4,0)

    # Configure variators
    self._mjcf_variator = variation.MJCFVariator()
    self._physics_variator = variation.PhysicsVariator()

    # Configure and enable observables
    self._creature.observables.enable_all()
    # if not include_camera:
    self._creature.observables.get_observable('realsense_camera').enabled = False
    self._button.observables.touch_force.enabled = True

    def to_button(physics):
      button_pos, _ = self._button.get_pose(physics)
      return self._creature.global_vector_to_local_frame(physics, button_pos)

    self._task_observables = {}
    self._task_observables['button_position'] = observable.Generic(to_button)

    for obs in self._task_observables.values():
      obs.enabled = True

    self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

  @property
  def root_entity(self):
    return self._arena

  @property
  def task_observables(self):
    return self._task_observables

  def initialize_episode_mjcf(self, random_state):
    self._mjcf_variator.apply_variations(random_state)

  def initialize_episode(self, physics, random_state):
    self._physics_variator.apply_variations(physics, random_state)
    creature_pose, button_pose = variation.evaluate(
        (self._creature_initial_pose, self._button_initial_pose),
        random_state=random_state)
    self._creature.set_pose(physics, position=creature_pose)
    self._button.set_pose(physics, position=button_pose)

  def get_reward(self,physics):
    return self._button.num_activated_steps / NUM_SUBSTEPS    
  
  def getObstacles(self):
    obs_pos = []
    # for wall in self._arena.walls:
    #     obs_pos.append(wall.pos)
    return np.array(obs_pos)