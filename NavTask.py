#@title The `PressWithSpecificForce` task
from dm_control import composer
from dm_control.composer.observation import observable
from Button import Button, UniformCircle
from Goal_Button import GoalButton
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from heightfield_arena import HeightFieldArena
import numpy as np
from dm_control.utils import transformations


NUM_SUBSTEPS = 25

class Navigation(composer.Task):

  def __init__(self, creature):
    self._creature = creature
    self._arena = HeightFieldArena()
    self._arena.add_free_entity(self._creature)
    self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))
    self._button = GoalButton()
    self._arena.attach(self._button)
    self._button2 = Button()
    self._arena.attach(self._button2)
    self._button3 = Button()
    self._arena.attach(self._button3)

    # position = np.random.uniform([0,0,0.1],[3,3,0.1],(4,3))
    # for i in range(4):
    #   self._arena.attach(Obstacle(position[i]))

    # Configure initial poses
    self._creature_initial_pose = (0, 0, 0)
    # button_distance = distributions.Uniform(4, 8)
    button_distance = 5
    # self._button_initial_pose = UniformCircle(button_distance)
    self._button_initial_pose = (3,4,0.01)

    positions = np.random.uniform([0,0,0.1],[3,3,0.1],(2,3))
    self._button2_initial_pose = positions[0]     #(1,3,0.1)
    self._button3_initial_pose = positions[1]     #(2,2,0.1)

    # Configure variators
    self._mjcf_variator = variation.MJCFVariator()
    self._physics_variator = variation.PhysicsVariator()

    # Configure and enable observables
    self._creature.observables.enable_all()
    # if not include_camera:
    self._creature.observables.get_observable('realsense_camera').enabled = True
    self._button.observables.touch_force.enabled = True
    self._button2.observables.touch_force.enabled = True
    self._button3.observables.touch_force.enabled = True

    def to_button(physics):
      button_pos, _ = self._button.get_pose(physics)
      return self._creature.global_vector_to_local_frame(physics, button_pos)

    def to_button2(physics):
      button_pos, _ = self._button2.get_pose(physics)
      return self._creature.global_vector_to_local_frame(physics, button_pos)
    
    def to_button3(physics):
      button_pos, _ = self._button2.get_pose(physics)
      return self._creature.global_vector_to_local_frame(physics, button_pos)

    self._task_observables = {}
    self._task_observables['button_position'] = observable.Generic(to_button)
    self._task_observables['button2_position'] = observable.Generic(to_button2)
    self._task_observables['button3_position'] = observable.Generic(to_button3)

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
    creature_pose, button_pose, button2_pose, button3_pose = variation.evaluate(
        (self._creature_initial_pose, self._button_initial_pose, self._button2_initial_pose, self._button3_initial_pose),
        random_state=random_state)
    self._creature.set_pose(physics, position=creature_pose)
    self._button.set_pose(physics, position=button_pose)
    self._button2.set_pose(physics, position=button2_pose)
    self._button3.set_pose(physics, position=button3_pose)

  def get_reward(self,physics):
    return self._button.num_activated_steps / NUM_SUBSTEPS    
  
  def getObstacles(self):
    # obs_pos = [self._button2_initial_pose,self._button3_initial_pose]
    obs_pos = []
    # for wall in self._arena.walls:
    #     obs_pos.append(wall.pos)
    return np.array(obs_pos)