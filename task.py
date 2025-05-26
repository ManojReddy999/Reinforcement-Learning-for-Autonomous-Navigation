#@title The `PressWithSpecificForce` task
from dm_control import composer
import numpy as np
from dm_control.utils import transformations
from dm_control.locomotion.arenas import floors
from car import Car
from obstacle import Wall_x, Wall_y, Sphere, Cube, Cylinder, Cuboid1, Cuboid2

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.005
COLLISION_REWARD = 0
GRAVITY = (0, 0, -9.81)

# np.random.seed(0)

class Survive(composer.Task):

  def __init__(self, num_obstacles=100, arena_size=8, control_timestep=DEFAULT_CONTROL_TIMESTEP, physics_timestep=DEFAULT_PHYSICS_TIMESTEP, random_seed=None):
    super().__init__()
    np.random.seed(random_seed)
    
    self._car = Car()
    self.arena_size = arena_size
    self._arena = floors.Floor(size=(self.arena_size,self.arena_size))
    self._arena.mjcf_model.option.gravity = GRAVITY
    self._arena.add_free_entity(self._car)
    self.num_obstacles = num_obstacles
    self._obstacles = []
    self.add_obstacles()
    self.add_walls()

    # Configure and enable observables
    # self._car.observables.enable_all()
    for obs in self._car.observables.all_observables:
       obs.enabled = True
  
    self.set_timesteps(control_timestep, physics_timestep)

  @property
  def root_entity(self):
    return self._arena 

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    self._arena.initialize_episode(physics, random_state)
    
    # pos = np.random.uniform([-self.arena_size+1,-self.arena_size+1,0],[self.arena_size-1,self.arena_size-1,0])
    # yaw = np.random.uniform(-np.pi,np.pi)
    pos = np.array([0, 0, 0])
    yaw = 0
    self.last_pos = pos[:2]

    # all_positions = np.vstack((self.position))
    # distances = np.linalg.norm(all_positions[:, :2] - pos[:2], axis=1)

    # while np.any(distances<1):
    #    pos = np.random.uniform([-self.arena_size+1,-self.arena_size+1,0],[self.arena_size-1,self.arena_size-1,0])
    #    distances = np.linalg.norm(all_positions[:, :2] - pos[:2], axis=1)
       
    self._car.set_pose(physics, pos, transformations.euler_to_quat([0,0,yaw]))

    self.obstacle_pos = self.generate_obstacle_positions()
    for i, pos in enumerate(self.obstacle_pos):
       self._obstacles[i]._geom.pos = [pos[0], pos[1], 0.2]

    self._creature_geomids = set(physics.bind(self._car.mjcf_model.find_all('geom')).element_id)
    self.obstacle_geom_ids = set(physics.bind([obs._geom for obs in self._obstacles]).element_id)

  def detect_collisions(self,physics):
      
      for contact in physics.data.contact:
          if (contact.geom1 in self._creature_geomids and contact.geom2 in self.obstacle_geom_ids) or \
              (contact.geom2 in self._creature_geomids and contact.geom1 in self.obstacle_geom_ids):
              return True
      return False
  
  def add_walls(self):
    self._walls = []
    # Create walls at the edges of the arena
    wall_1 = Wall_x(np.array([self.arena_size,0,0]),index=1)
    self._arena.attach(wall_1)
    self._obstacles.append(wall_1)
    wall_2 = Wall_x(np.array([-self.arena_size,0,0]),index=2)
    self._arena.attach(wall_2)
    self._obstacles.append(wall_2)
    wall_3 = Wall_y(np.array([0,self.arena_size,0]),index=3)
    self._arena.attach(wall_3)
    self._obstacles.append(wall_3)
    wall_4 = Wall_y(np.array([0,-self.arena_size,0]),index=4)
    self._arena.attach(wall_4)
    self._obstacles.append(wall_4)

  def generate_obstacle_positions(self):
    min_dist = 0.5
    obs_sep = 0.5
    space_min = -self.arena_size
    space_max = self.arena_size
    
    positions = []
    while len(positions) < self.num_obstacles:
        pos = np.random.uniform([space_min, space_min, 0], [space_max, space_max, 0])
        # Check if the obstacle is too close to origin
        if np.linalg.norm(pos[:2]) < min_dist:
            continue   
        # if any(np.linalg.norm(pos[:2]-p[:2]) < obs_sep for p in positions):
        #   continue  
        positions.append(pos)
    return positions
  
  def add_obstacles(self):
    # obstacle_types = [Sphere, Cube, Cylinder, Cuboid1, Cuboid2]
    obstacle_types = [Cube]
    positions = self.generate_obstacle_positions()

    for i, pos in enumerate(positions):
        type = obstacle_types[(len(obstacle_types)*i) // self.num_obstacles]
        obs = type(pos, i+1)
        self._arena.attach(obs)
        self._obstacles.append(obs)

  def get_reward(self,physics):
    # collision = False
    vel = self._car.observables.body_vel_2d(physics)
    speed = np.linalg.norm(vel)
    # cur_pos, _ = self._car.get_pose(physics)
    # # print(cur_pos,self.last_pos)
    # step_dist = np.linalg.norm(cur_pos[:2] - self.last_pos)
    # self.last_pos = cur_pos[:2].copy()

    reward = speed
    # reward = step_dist/DEFAULT_CONTROL_TIMESTEP
    # print("Reward from Task:",reward)

    # if self.detect_collisions(physics):
    #     reward += COLLISION_REWARD
    #     collision = True
    return reward

    # def add_obstacles(self):
    #   obstacle_types = [Sphere, Cube, Cylinder, Cuboid1, Cuboid2]
    #   # obstacle_types = [Cube]
      
    #   # Generate all positions at once
    #   self.position = np.random.uniform(
    #       [-self.arena_size, -self.arena_size, 0],
    #       [self.arena_size, self.arena_size, 0],
    #       (self.num_obstacles, 3)
    #   )

    #   for i in range(self.num_obstacles):
    #       obstacle_type_index = (len(obstacle_types)*i) // self.num_obstacles
    #       # Create the appropriate obstacle type
    #       obs = obstacle_types[obstacle_type_index](self.position[i], index=((len(obstacle_types)*i) % self.num_obstacles) + 1)
    #       self._arena.attach(obs)
    #       self._obstacles.append(obs)
