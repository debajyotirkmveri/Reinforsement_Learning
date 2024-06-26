from feature_extractor import FeatureExtractor
from gridworld import GridWorld


class GridWorldFeatureExtractor3(FeatureExtractor):

    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 3

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
            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]
        return feature_values

class GridWorldFeatureExtractor6(FeatureExtractor):

    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 6

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
                # Features to determine if we are in goal row or column
                feature_values += [1 if goal[x] == state[x] else 0]
                feature_values += [1 if goal[y] == state[y] else 0]
                feature_values += [1 if action == "UP" else 0]
            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]
        return feature_values

class GridWorldFeatureExtractor7(FeatureExtractor):

    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 7

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
                # Features to determine if we are in goal row or column
                feature_values += [1 if goal[x] == state[x] else 0]
                feature_values += [1 if goal[y] == state[y] else 0]
                feature_values += [1 if action == "DOWN" else 0]
                feature_values += [1 if action == "LEFT" else 0]  # Removed extra space
            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]
        return feature_values

    
    
    
    
    
    
    
