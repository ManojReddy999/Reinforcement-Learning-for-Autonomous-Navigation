import numpy as np
import random

GOAL_THRESHOLD = 0.3

class GoalGraph():
    """
    A graph of goals that the car can drive to. Once the car arrives at a goal,
    the goal will be changed to one of its successors.
    """
    def __init__(self, scale, goal_threshold=GOAL_THRESHOLD):
        self.goals = [
            (10.0, 0.0),
            (0.0, 10.0),
            (-10.0, 0.0),
            (0.0, -10.0),
        ]

        self.start_headings = [
            (a - 0.1, a + 0.1) for a in [
                np.deg2rad(90),
                np.deg2rad(180),
                np.deg2rad(-90),
                np.deg2rad(0),
            ]
        ]

        self.graph = {
            0: [1],
            1: [2],
            2: [3],
            3: [0],
        }

        self.goal_reprs = []
        self.edge_reprs = []

        self.current_start_idx = 0
        self.current_goal_idx = 0

        self.goal_threshold = goal_threshold
        self.scale = scale

    @property
    def current_start(self):
        return np.array(self.goals[self.current_start_idx]) * self.scale, self.start_headings[self.current_start_idx]

    @property
    def current_goal(self):
        return np.array(self.goals[self.current_goal_idx]) * self.scale

    def is_complete(self):
        return self._ticks_at_current_goal > 0

    def set_goal(self, goal_idx, physics):
        """
        Set a new goal and update the renderables to match.
        """
        for idx, repr in enumerate(self.goal_reprs):
            opacity = 1.0 if idx == goal_idx else 0.0
            if physics:
                physics.bind(repr).rgba = (*repr.rgba[:3], opacity)
            else:
                repr.rgba = (*repr.rgba[:3], opacity)

        self.current_goal_idx = goal_idx
        self._ticks_at_current_goal = 0

    def tick(self, car_pos, physics):
        """
        Update the goal if the car was at the current goal for at least one tick.
        We need the delay so that the car can get the high reward for reaching
        the goal before the goal changes.
        """
        if self.is_complete():
            self.current_start_idx = self.current_goal_idx
            self.set_goal(random.choice(self.graph[self.current_start_idx]), physics)
            return True

        if np.linalg.norm(np.array(car_pos)[:2] - self.current_goal) < self.goal_threshold:
            self._ticks_at_current_goal += 1
        else:
            self._ticks_at_current_goal = 0

        return False

    def reset(self, physics):
        self.current_start_idx = random.randint(0, len(self.goals) - 1)
        self.set_goal(random.choice(self.graph[self.current_start_idx]), physics)
        self._ticks_at_current_goal = 0

    def add_renderables(self, mjcf_root, height_lookup, show_edges=False):
        """
        Add renderables to the mjcf root to visualize the goals and (optionally) edges.
        """
        RENDER_HEIGHT_OFFSET = 0.0

        self.goal_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="sphere",
                                    size="0.08",
                                    rgba=(0.0, 1.0, 0.0, 0.5),
                                    pos=(g[0] * self.scale, g[1] * self.scale, height_lookup((g[0] * self.scale, g[1] * self.scale)) + RENDER_HEIGHT_OFFSET))
            for g in self.goals
        ]

        self.edge_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="cylinder",
                                    size="0.04",
                                    rgba=(1, 1, 1, 0.5),
                                    fromto=(self.goals[s][0] * self.scale, self.goals[s][1] * self.scale, height_lookup((self.goals[s][0] * self.scale, self.goals[s][1] * self.scale)) + RENDER_HEIGHT_OFFSET,
                                            self.goals[g][0] * self.scale, self.goals[g][1] * self.scale, height_lookup((self.goals[g][0] * self.scale, self.goals[g][1] * self.scale)) + RENDER_HEIGHT_OFFSET))
            for s in self.graph for g in self.graph[s]
            if show_edges
        ]
