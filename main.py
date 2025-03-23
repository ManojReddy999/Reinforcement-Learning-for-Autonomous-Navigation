from task import CarTask
from dm_control import viewer
from dm_control import composer
import numpy as np
from car import Car
from dm_control.mjcf.physics import Physics
import PIL.Image
import matplotlib.pyplot as plt
import mujoco

# task = CarTask()
# original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

# def random_policy(time_step):
#     del time_step  # Unused.
#     # return np.random.uniform(low=np.array([-0.38, -1.]), high=np.array([0.38, 3.]), size=(2,))
#     return np.array([0 , 0.5])

# viewer.launch(original_env, policy=random_policy)


# for _ in range(5):
#     timestep = original_env.reset()
#     while not timestep.last():
#         # action = np.random.uniform(low=np.array([-0.38, -1.]), high=np.array([0.38, 3.]), size=(2,))
#         action = np.array([0 , 0.5])
#         timestep = original_env.step(action)
#     viewer.launch(original_env)


# Car_Object = CarTask()
# physics = Physics.from_mjcf_model(Car_Object.root_entity.mjcf_model)
# print('timestep', physics.model.opt.timestep)
# print('gravity', physics.model.opt.gravity)
# print(physics.data.time, physics.data.qpos, physics.data.qvel)
# print(physics.named.data.geom_xpos)
# pixels = physics.render()
# imgplot = plt.imshow(pixels)
# plt.show()
# print("Das")

# task = CarTask()
# original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
# viewer.launch(original_env)
# for _ in range(5):
#     timestep = original_env.reset()
#     while not timestep.last():
#         action = np.array([0 , 0.5])
#         timestep = original_env.step(action)
#     viewer.launch(original_env)








# Car_Object = Car()
# Car_Object.observables.enable_all()
# physics = Physics.from_mjcf_model(Car_Object.mjcf_model)
# print('timestep', physics.model.opt.timestep)
# print('gravity', physics.model.opt.gravity)
# print(physics.data.time, physics.data.qpos, physics.data.qvel)
# print(physics.named.data.geom_xpos)
# pixels = physics.render()
# imgplot = plt.imshow(pixels)
# plt.show()
# print("Das")


# task = CarTask()
# original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
# viewer.launch(original_env)
# for _ in range(5):
#   timestep = original_env.reset()
#   while not timestep.last():
#     action = np.random.uniform(low=np.array([-0.38, -1.]), high=np.array([0.38, 3.]), size=(2,))
#     timestep = original_env.step(action)
#     viewer.launch(original_env)


# def random_policy(time_step):
#   del time_step  # Unused.
#   return np.random.uniform(low=np.array([-0.38, -1.]), high=np.array([0.38, 3.]), size=(2,))

# viewer.launch(original_env, policy=random_policy)

# Launch the viewer application.
# viewer.launch(original_env)


