from feature_extractor import FeatureExtractor
from gridworld import GridWorld
import numpy as np

class GridWorldFeatureExtractor4(FeatureExtractor):

    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 4  # Three existing features + one angle-based feature

    def num_actions(self):
        return len(self.mdp.get_actions())

    def extract(self, state, action):
        goal = (self.mdp.width - 1, self.mdp.height - 1)
        x = 0
        y = 1
        e = 0.01
        feature_values = []

        for a in self.mdp.get_actions():
            if a == action and state != GridWorld.TERMINAL:
                feature_values += [(state[x] + e) / (goal[x] + e)]
                feature_values += [(state[y] + e) / (goal[y] + e)]
                feature_values += [
                    (goal[x] - state[x] + goal[y] - state[y] + e)
                    / (goal[x] + goal[y] + e)
                ]

                # Calculate angle-based feature
                goal_direction = np.array(goal) - np.array(state)
                action_direction = np.array(action)
                angle_radians = np.arccos(np.clip(np.dot(goal_direction, action_direction), -1.0, 1.0))
                angle_degrees = np.degrees(angle_radians)
                feature_values += [angle_degrees]
            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]

        return feature_values

from feature_extractor import FeatureExtractor
from gridworld import GridWorld
import numpy as np

class GridWorldFeatureExtractor5(FeatureExtractor):

    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 4  # Three existing features + one angle-based feature

    def num_actions(self):
        return len(self.mdp.get_actions())



    
    def extract(self, state, action):
        goal = (self.mdp.width - 1, self.mdp.height - 1)
        x = 0
        y = 1
        e = 0.01
        feature_values = []

        # Ensure that state and action are NumPy arrays with float data type
        state = np.array(state, dtype=float)
        action = np.array(action, dtype=float)

        for a in self.mdp.get_actions():
            if np.array_equal(a, action) and not np.array_equal(state, GridWorld.TERMINAL):
                goal_direction = np.array(goal) - state
                action_direction = action

                # Calculate angle-based feature
                angle_radians = np.arccos(np.clip(np.dot(goal_direction, action_direction), -1.0, 1.0))
                angle_degrees = np.degrees(angle_radians)

                feature_values += [(state[x] + e) / (goal[x] + e)]
                feature_values += [(state[y] + e) / (goal[y] + e)]
                feature_values += [(goal[x] - state[x] + goal[y] - state[y] + e) / (goal[x] + goal[y] + e)]
                feature_values += [angle_degrees]
            else:
                feature_values += [0.0] * self.num_features()

        return feature_values
