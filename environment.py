import numpy as np


class Environment:

    def __init__(self, transition_matrix, rewards, gamma):
        self.transition_matrix = transition_matrix
        self.rewards = rewards
        self.gamma = gamma
        self.num_states = len(transition_matrix)
        self.state = np.random.randint(self.num_states)

    def step(self, action):
        rand_num = np.random.uniform()

        next_state = 0
        while next_state < self.num_states-1 and rand_num > sum(self.transition_matrix[self.state][action][0:(next_state+1)]):
            next_state += 1

        reward = self.rewards[self.state][next_state]
        self.state = next_state

        return self.state, reward
