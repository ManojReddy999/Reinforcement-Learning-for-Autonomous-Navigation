import numpy as np
from env import CarEnv
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import multiprocessing

class ObstacleInfoWrapper(Monitor):
    """Wrapper that adds a method to get obstacle positions directly"""
    def get_obstacle_positions(self):
        # This method will be called inside the subprocess
        if hasattr(self.env, 'task') and hasattr(self.env.task, 'obstacle_pos'):
            return self.env.task.obstacle_pos
        return None

def make_env(rank, num_obstacles=20, random_seed=None):
    """
    Create a function that will create and wrap an environment when called
    """
    def _init():
        # Create environment with or without seeding
        env = CarEnv(num_obstacles=num_obstacles, random_seed=random_seed)
        print(f"Created environment {rank} in process {os.getpid()}")
        # Use our custom wrapper instead of Monitor
        return ObstacleInfoWrapper(env)
    return _init

def test_reset_obstacle_positions():
    """Test whether obstacle positions change when an environment is reset"""
    print("\n=== Testing if obstacle positions change on reset ===")
    
    # Create a single environment directly (not in subprocess)
    env = ObstacleInfoWrapper(CarEnv(num_obstacles=20, random_seed=42))
    
    # Get initial obstacle positions
    _, _ = env.reset()  # First reset to initialize
    initial_positions = env.get_obstacle_positions()
    print("Initial obstacle positions (first 3):", initial_positions[:3])
    
    # Reset multiple times and check if positions change
    for i in range(3):
        print(f"\nReset {i+1}:")
        _, _ = env.reset()
        new_positions = env.get_obstacle_positions()
        print(f"New obstacle positions (first 3): {new_positions[:3]}")
        
        # Compare with initial positions
        are_identical = np.allclose(initial_positions, new_positions)
        print(f"Reset {i+1} positions identical to initial: {are_identical}")
        
        # Check if there's any change from the initial positions
        if not are_identical:
            print("Obstacles change position after reset!")
        else:
            print("Obstacles remain in the same positions after reset.")
    
    env.close()

def test_subproc_identical_envs():
    n_envs = 4  # Number of environments to create in each SubprocVecEnv
    num_obstacles = 20
    
    # Test 1: Create environments without explicit seeding (current approach)
    print("\n=== Testing SubprocVecEnv WITHOUT explicit seeding ===")
    # Create environments without specifying seeds
    envs_no_seed = SubprocVecEnv([make_env(i, num_obstacles=num_obstacles) for i in range(n_envs)])
    
    # Reset environments 
    envs_no_seed.reset()
    
    # Get obstacle positions from all environments
    all_positions = []
    for i in range(n_envs):
        pos = envs_no_seed.env_method('get_obstacle_positions', indices=[i])[0]
        all_positions.append(pos)
        print(f"Sample positions from env{i}:", pos[:3])
    
    # Check if any pair is identical
    identical_pairs = []
    for i in range(n_envs):
        for j in range(i+1, n_envs):
            are_identical = np.allclose(all_positions[i], all_positions[j])
            identical_pairs.append((i, j, are_identical))
            print(f"Environments {i} and {j} have identical obstacle positions: {are_identical}")
    
    # Overall summary
    any_identical = any(is_identical for _, _, is_identical in identical_pairs)
    print(f"Any environments have identical obstacle positions: {any_identical}")
    
    # Close environments
    envs_no_seed.close()
    
    # Test 2: Create environments with different seeds
    print("\n=== Testing SubprocVecEnv WITH explicit different seeds ===")
    # Create environments with different seeds
    envs_with_seeds = SubprocVecEnv([
        make_env(0, num_obstacles=num_obstacles, random_seed=42),
        make_env(1, num_obstacles=num_obstacles, random_seed=43)
    ])
    
    # Reset environments
    envs_with_seeds.reset()
    
    # Get obstacle positions from each environment using our custom method
    pos3 = envs_with_seeds.env_method('get_obstacle_positions', indices=[0])[0]
    pos4 = envs_with_seeds.env_method('get_obstacle_positions', indices=[1])[0]
    
    # Check if they're identical
    are_identical = np.allclose(pos3, pos4)
    print(f"SubprocVecEnv environments with different seeds have identical positions: {are_identical}")
    
    if not are_identical:
        print("Sample positions from env with seed 42:", pos3[:3])
        print("Sample positions from env with seed 43:", pos4[:3])
    
    # Test 3: Create environments with SAME seed
    print("\n=== Testing SubprocVecEnv WITH identical seeds ===")
    # Create environments with the same seed
    envs_same_seed = SubprocVecEnv([
        make_env(0, num_obstacles=num_obstacles, random_seed=42),
        make_env(1, num_obstacles=num_obstacles, random_seed=42)
    ])
    
    # Reset environments
    envs_same_seed.reset()
    
    # Get obstacle positions from each environment using our custom method
    pos5 = envs_same_seed.env_method('get_obstacle_positions', indices=[0])[0]
    pos6 = envs_same_seed.env_method('get_obstacle_positions', indices=[1])[0]
    
    # Check if they're identical
    are_identical = np.allclose(pos5, pos6)
    print(f"SubprocVecEnv environments with identical seeds have identical positions: {are_identical}")
    
    if are_identical:
        print("Sample positions from both environments with same seed (42):", pos5[:3])
    
    # Close environments
    envs_with_seeds.close()
    envs_same_seed.close()

if __name__ == "__main__":
    # Ensure proper process spawning method for multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    # test_subproc_identical_envs()
    test_reset_obstacle_positions()