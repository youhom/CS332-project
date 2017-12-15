import numpy as np
from itertools import product


def policy_values(transition_matrix, rewards, gamma):
    """Return a dictionary where keys are policies and values
    are the value function under the policy."""

    # Number of states
    num_states = len(transition_matrix)

    # Number of actions
    num_actions = len(transition_matrix[0])

    # Output dictionary
    value_function = dict()

    for policy in product(list(range(num_actions)), repeat=num_states):

        policy_trans_matrix = np.array([
            transition_matrix[i][policy[i]] for i in range(num_states)
        ])
        a = gamma * policy_trans_matrix - np.identity(num_states)
        b = np.sum(-1 * policy_trans_matrix * rewards, axis=1)
        value_function[tuple(policy)] = np.linalg.solve(a, b)

    return value_function


def stationary_probability(policy, transition_matrix):
    """Return stationary policy at each state under policy."""

    num_states = len(transition_matrix)
    num_actions = len(transition_matrix[0])

    a = np.array([
        [transition_matrix[0][policy[0]][0]-1, transition_matrix[1][policy[1]][0]],
        [1,1]
    ])
    b = np.array([0, 1])
    return np.linalg.solve(a, b)


def value_difference(states, feat, weights, policy, policy_vals):
    """Calculate the difference [V_w(X)-V_w(Y)]-[V(X)-V(Y)]."""

    return (weights @ np.transpose(feat) - policy_vals[policy]) @ np.transpose(np.array(states))


def sibling(cur_st, act, next_st, transition_matrix):
    """Return sibling state and conditional probability."""

    states = list(range(len(transition_matrix)))
    states.remove(next_st)
    probabilities = np.array([transition_matrix[cur_st][act][s] for s in states])
    probabilities = probabilities / sum(probabilities)

    rand_num = np.random.rand(1)[0]

    i = 0
    while i < len(states)-1 and rand_num > sum(probabilities[0:(i+1)]):
        i += 1

    return states[i], probabilities[i]


def make_new_policy(agent, new_weight, transition_matrix, rewards, gamma):
    """Update policy to act greedily using value approximations defined by self.w."""
    new_policy = [0] * len(agent.policy)

    for st in range(len(agent.policy)):
        action_vals = np.zeros(len(transition_matrix[0]))
        for act, next_states in enumerate(transition_matrix[st]):
            val = 0
            for next_st, pr in enumerate(next_states):
                val += pr * (rewards[st][next_st] + gamma * sum(agent.features[next_st] * new_weight))
            action_vals[act] = val
        new_policy[st] = np.argmax(action_vals)
    return tuple(new_policy)
