#@title The `PressWithSpecificForce` task
from dm_control import composer
import numpy as np
from dm_control.utils import transformations
from dm_control.locomotion.arenas import floors
from car import Car
from obstacle import Wall_x, Wall_y, Sphere, Cube, Cylinder, Cuboid1, Cuboid2
import random

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
    # self.generate_maze(rows=20, cols=20, cell_size=0.8,wall_thickness=0.05, wall_height=0.4)         
    self.add_walls()

    # Configure and enable observables
    # self._car.observables.enable_all()
    for obs in self._car.observables.all_observables:
       obs.enabled = True
  
    self.set_timesteps(control_timestep, physics_timestep)
    self._collision_times = []
    self.collided = False
    self.timesteps = 0
    self.reward = 0

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
    
    #for evaluations

    self._collision_times = []
    self.collided = False
    self.timesteps = 0
    self.reward = 0

  def detect_collisions(self,physics):
      
      for contact in physics.data.contact:
          if (contact.geom1 in self._creature_geomids and contact.geom2 in self.obstacle_geom_ids) or \
              (contact.geom2 in self._creature_geomids and contact.geom1 in self.obstacle_geom_ids):
              self.collided = True
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
  
  def generate_maze(self,
                    rows: int,
                    cols: int,
                    cell_size: float,
                    wall_thickness: float = 0.1,
                    wall_height: float = 0.2):
      """
      Carve a maze of size rows×cols inside the square arena, then place
      obstacles as Cuboid1 (horizontal) and Cuboid2 (vertical) wall segments,
      except we skip any segment that would overlap (0,0,0).

      Parameters
      ----------
      rows, cols : int
          Number of cells in the Y (rows) and X (cols) directions.
      cell_size : float
          The full width of each grid‐cell. The entire maze spans
          (cols*cell_size) × (rows*cell_size) in world units, centered at (0,0).
      wall_thickness : float, default=0.1
          The full thickness of each wall segment. This becomes the smaller
          dimension of Cuboid1/Cuboid2.
      wall_height : float, default=0.2
          The full vertical height of each wall segment (Z dimension).
      """
      # 1) Initialize all walls "up" (True). We'll knock some down.
      h_walls = [[True for _ in range(cols)] for _ in range(rows + 1)]
      v_walls = [[True for _ in range(cols + 1)] for _ in range(rows)]
      visited = [[False for _ in range(cols)] for _ in range(rows)]

      # 2) DFS to carve out a perfect maze
      def carve(r: int, c: int):
          visited[r][c] = True
          directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
          random.shuffle(directions)

          for dr, dc in directions:
              nr, nc = r + dr, c + dc
              if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                  if dr == 0 and dc == 1:
                      # remove vertical wall to the right of (r,c)
                      v_walls[r][c + 1] = False
                  elif dr == 0 and dc == -1:
                      # remove vertical wall to the left of (r,c)
                      v_walls[r][c] = False
                  elif dr == 1 and dc == 0:
                      # remove horizontal wall below (r,c)
                      h_walls[r + 1][c] = False
                  elif dr == -1 and dc == 0:
                      # remove horizontal wall above (r,c)
                      h_walls[r][c] = False

                  carve(nr, nc)

      carve(0, 0)

      # 3) Compute world‐coordinates of the maze bounding box
      total_width = cols * cell_size
      total_height = rows * cell_size
      half_wall_h = wall_height / 2.0
      half_thick = wall_thickness / 2.0
      half_cell = cell_size / 2.0

      # 4) Place remaining horizontal walls (h_walls[i][j] == True),
      #    skipping any that would overlap (0,0).
      for i in range(rows + 1):
          for j in range(cols):
              if not h_walls[i][j]:
                  continue

              # Center of that wall in world‐coords:
              wx = - total_width / 2.0 + j * cell_size + half_cell
              wy = - total_height / 2.0 + i * cell_size
              wz = 0.0

              # Check if this horizontal‐wall box would cover (0,0):
              #   - horizontally it spans [wx - half_cell, wx + half_cell]
              #   - vertically (in Y) it spans [wy - half_thick, wy + half_thick]
              if (abs(wx) <= half_cell) and (abs(wy) <= half_thick):
                  # This wall would overlap the origin—skip it.
                  continue

              pos = np.array([wx, wy, wz], dtype=np.float32)
              size = [half_cell, half_thick, half_wall_h]  # [half_len_x, half_len_y, half_len_z]
              idx = len(self._obstacles) + 1
              wall = Cuboid1(pos, name=f'hwall_{i}_{j}', size=size, index=idx)
              self._arena.attach(wall)
              self._obstacles.append(wall)

      # 5) Place remaining vertical walls (v_walls[i][j] == True),
      #    skipping any that would overlap (0,0).
      for i in range(rows):
          for j in range(cols + 1):
              if not v_walls[i][j]:
                  continue

              # Center of that wall in world‐coords:
              wx = - total_width / 2.0 + j * cell_size
              wy = - total_height / 2.0 + i * cell_size + half_cell
              wz = 0.0

              # Check if this vertical‐wall box would cover (0,0):
              #   - horizontally (in X) it spans [wx - half_thick, wx + half_thick]
              #   - vertically it spans [wy - half_cell, wy + half_cell]
              if (abs(wx) <= half_thick) and (abs(wy) <= half_cell):
                  # This wall would overlap the origin—skip it.
                  continue

              pos = np.array([wx, wy, wz], dtype=np.float32)
              size = [half_thick, half_cell, half_wall_h]  # [half_len_x, half_len_y, half_len_z]
              idx = len(self._obstacles) + 1
              wall = Cuboid2(pos, name=f'vwall_{i}_{j}', size=size, index=idx)
              self._arena.attach(wall)
              self._obstacles.append(wall)

  
  def add_obstacles(self):
    obstacle_types = [Sphere, Cube, Cylinder, Cuboid1, Cuboid2]
    # obstacle_types = [Cube]
    positions = self.generate_obstacle_positions()

    for i, pos in enumerate(positions):
        type = obstacle_types[(len(obstacle_types)*i) // self.num_obstacles]
        obs = type(pos, i+1)
        self._arena.attach(obs)
        self._obstacles.append(obs)

  def get_reward(self,physics):
    # collision = False
    dir = np.sign(self._car.observables.wheel_speeds(physics)[2])
    vel = self._car.observables.body_vel_2d(physics)
    speed = np.linalg.norm(vel)
    # cur_pos, _ = self._car.get_pose(physics)
    # # print(cur_pos,self.last_pos)
    # step_dist = np.linalg.norm(cur_pos[:2] - self.last_pos)
    # self.last_pos = cur_pos[:2].copy()

    reward = dir*speed
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
