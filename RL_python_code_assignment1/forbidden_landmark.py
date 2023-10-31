from feature_extractor import FeatureExtractor
from gridworld import GridWorld
class Gr_11_11(FeatureExtractor):

    def __init__(self, mdp, block_states):
        self.mdp = mdp
        self.block_states = block_states 

    def num_features(self):
        landmark_positions = [(1,2),(2,0)] 

        # Include an additional binary feature for each action and one feature per landmark
        return 12 + len(self.landmark_positions)

    def num_actions(self):
        return len(self.mdp.get_actions())

    def extract(self, state, action):
        goal = (self.mdp.width - 1, self.mdp.height - 1)
        x = 0
        y = 1
        e = 0.01
        feature_values = []
        # Check if the agent is on the x-axis
        on_x_axis = 1 if state[y] == 0 else 0

        # Check if the agent is in the upper row of the grid
        upper_row = 1 if state[y] == 0 else 0

        for a in self.mdp.get_actions():
            if a == action and state != GridWorld.TERMINAL:
                # Distance-based features (same as before)
                feature_values += [(state[x] + e) / (goal[x] + e)]
                feature_values += [(state[y] + e) / (goal[y] + e)]
                feature_values += [(goal[x] - state[x] + goal[y] - state[y] + e) / (goal[x] + goal[y] + e)]

                # Binary features (different from before)
                feature_values += [1 if goal[x] == state[x] else 0]
                feature_values += [1 if goal[y] == state[y] else 0]

                # Binary feature for being in a terminal state (same as before)
                feature_values += [1 if state == GridWorld.TERMINAL else 0]

                # Additional binary features (excluding block states)
                feature_values += [1 if state[x] % 3 == 0 and state not in self.block_states else 0]
                feature_values += [1 if state[y] % 3 == 0 and state not in self.block_states else 0]
                feature_values += [1 if state[x] + state[y] == goal[x] and state not in self.block_states else 0]
                feature_values += [1 if state[y] > state[x] and state not in self.block_states else 0]
                feature_values += [on_x_axis]  # Feature indicating if agent is on the x-axis and should not go downwards
                feature_values += [1 if not upper_row else 0]  # Prevent agent from going upwards if already in the upper row

                # Linear features for proximity to landmarks
                for landmark in self.landmark_positions:
                    dist = ((state[x] - landmark[0]) ** 2 + (state[y] - landmark[1]) ** 2) ** 0.5
                    feature_values += [1 / (dist + 1e-6)]  # Avoid division by zero

            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]

        return feature_values
