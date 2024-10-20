import numpy as np

def reward_function(s, env_size):
    #print("ENV_SIZE: " + str(env_size))
    #print("STATE in reward_function is: "  + str(s))
    if (s == [env_size,env_size]).all():
        return 1
    return 0

def transition_probabilities(env, s, a, env_size, directions, holes):
    cells = []
    probs = []
    prob_next_state = np.zeros((env_size, env_size))

    #prob_next_state =  #TODO

    return prob_next_state
