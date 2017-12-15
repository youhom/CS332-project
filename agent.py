import numpy as np


class Agent:
    def __init__(self, features, policy):
        self.features = features
        self.num_features = len(features[0])
        self.policy = policy
        self.weight = np.array(self.num_features)

    def get_action(self, state):
        """Return action under the policy."""

        return self.policy[state]