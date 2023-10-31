from qfunction import QFunction

class LinearQFunction(QFunction):
    def __init__(self, features, weights=None, default=0.0):
        self.features = features
        if weights == None:
            self.weights = [
                default
                for _ in range(0, features.num_actions())
                for _ in range(0, features.num_features())
            ]

    def update(self, state, action, delta):
        # update the weights
        feature_values = self.features.extract(state, action)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (delta * feature_values[i])

    def get_q_value(self, state, action):
        q_value = 0.0
        feature_values = self.features.extract(state, action)
        for i in range(len(feature_values)):
            q_value += feature_values[i] * self.weights[i]
        return q_value

""" 
from qfunction import QFunction

class LinearQFunction(QFunction):
    def __init__(self, features, default=0.0):
        self.features = features
        self.num_features = features.num_features()
        self.num_actions = features.num_actions()
        self.weights = [default] * (self.num_features * self.num_actions)

    def update(self, state, action, delta):
        # update the weights
        feature_values = self.features.extract(state, action)
        for i in range(self.num_features):
            for j in range(self.num_actions):
                weight_index = i * self.num_actions + j
                self.weights[weight_index] += delta * feature_values[i]

    def get_q_value(self, state, action):
        q_value = 0.0
        feature_values = self.features.extract(state, action)
        for i in range(self.num_features):
            for j in range(self.num_actions):
                weight_index = i * self.num_actions + j
                q_value += feature_values[i] * self.weights[weight_index]
        return q_value


"""
