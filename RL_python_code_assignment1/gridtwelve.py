from feature_extractor import FeatureExtractor
from gridworld import GridWorld

class Grid12(FeatureExtractor):

    def __init__(self, mdp, blocked_states):
        self.mdp = mdp
        self.blocked_states = blocked_states  # 2D matrix with 0 for valid states and 1 for blocked states

    def num_features(self):
        # Include an additional binary feature for each action
        return 11

    def num_actions(self):
        return len(self.mdp.get_actions())

    def is_forbidden_state(self, state):
        # Check if the state is marked as blocked in the grid matrix
        x, y = state
        return self.blocked_states[y][x] == 1

    def extract(self, state, action):
        goal = (self.mdp.width - 1, self.mdp.height - 1)
        x = 0
        y = 1
        e = 0.01
        feature_values = []

        # Check if the state is forbidden based on the grid matrix
        forbidden_state = 1 if self.is_forbidden_state(state) else 0

        for a in self.mdp.get_actions():
            if a == action and state != GridWorld.TERMINAL:
                # Distance-based features
                feature_values += [(state[x] + e) / (goal[x] + e)]
                feature_values += [(state[y] + e) / (goal[y] + e)]
                feature_values += [(goal[x] - state[x] + goal[y] - state[y] + e) / (goal[x] + goal[y] + e)]

                # Binary features to determine if we are in the goal row or column
                feature_values += [1 if goal[x] == state[x] else 0]
                feature_values += [1 if goal[y] == state[y] else 0]

                # Binary feature for being in a terminal state
                feature_values += [1 if state == GridWorld.TERMINAL else 0]

                feature_values += [1 if state[x] % 2 == 0 else 0]  # Example: Is x-coordinate even?
                feature_values += [1 if state[y] % 2 == 0 else 0]  # Example: Is y-coordinate even?
                feature_values += [1 if state[x] + state[y] == goal[x] else 0]  # Example: Sum of coordinates equal to goal x?
                feature_values += [1 if state[y] > state[x] else 0]  # Example: y-coordinate greater than x-coordinate
                feature_values += [forbidden_state]  # Feature indicating if state is forbidden

            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]

        return feature_values
