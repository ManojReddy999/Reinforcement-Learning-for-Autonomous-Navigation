#@title The `Button` class
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions

import numpy as np

NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.


class Button(composer.Entity):
  """A button Entity which changes colour when pressed with certain force."""
  def _build(self, target_force_range=(0, 1000)):
    self._min_force, self._max_force = target_force_range
    self._mjcf_model = mjcf.RootElement()
    # self._geom = self._mjcf_model.worldbody.add(
    #     'geom', type='cylinder', size=[0.25, 0.5], rgba=[0, 0, 1, 1])
    # self._site = self._mjcf_model.worldbody.add(
    #     'site', type='cylinder', size=self._geom.size*1.01, rgba=[0, 0, 1, 0])
    self._geom = self._mjcf_model.worldbody.add(
        'geom', type='sphere', size=[0.1], rgba=[0, 0, 1, 1])         # Sphere is added here to use it as an obstacle
    self._site = self._mjcf_model.worldbody.add(
        'site', type='sphere', size=self._geom.size*1.01, rgba=[0, 0, 1, 0])
    self._sensor = self._mjcf_model.sensor.add('touch', site=self._site)
    self._num_activated_steps = 0

  def _build_observables(self):
    return ButtonObservables(self)

  @property
  def mjcf_model(self):
    return self._mjcf_model
  # Update the activation (and colour) if the desired force is applied.
  def _update_activation(self, physics):
    current_force = physics.bind(self.touch_sensor).sensordata[0]
    self._is_activated = (current_force > self._min_force and
                          current_force <= self._max_force)
    physics.bind(self._geom).rgba = (
        [1, 0, 0, 1] if self._is_activated else [0, 0, 1, 1])
    self._num_activated_steps += int(self._is_activated)

  def initialize_episode(self, physics, random_state):
    self._reward = 0.0
    self._num_activated_steps = 0
    self._update_activation(physics)

  def after_substep(self, physics, random_state):
    self._update_activation(physics)

  @property
  def touch_sensor(self):
    return self._sensor

  @property
  def num_activated_steps(self):
    return self._num_activated_steps


class ButtonObservables(composer.Observables):
  """A touch sensor which averages contact force over physics substeps."""
  @composer.observable
  def touch_force(self):
    return observable.MJCFFeature('sensordata', self._entity.touch_sensor,
                                  buffer_size=NUM_SUBSTEPS, aggregator='mean')
  

#@title Random initialiser using `composer.variation`


class UniformCircle(variation.Variation):
  """A uniformly sampled horizontal point on a circle of radius `distance`."""
  def __init__(self, distance):
    self._distance = distance
    self._heading = distributions.Uniform(0, 2*np.pi)

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    distance, heading = variation.evaluate(
        (self._distance, self._heading), random_state=random_state)
    return (distance*np.cos(heading), distance*np.sin(heading), 0)