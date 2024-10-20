import numpy as np

def reward_function(s, env_size):
    r = ...
    return r

def transition_probabilities(env, s, a, env_size, directions, holes):
    cells = []
    probs = []
    prob_next_state = np.zeros((env_size, env_size))

    prob_next_state = ...

    return prob_next_state
