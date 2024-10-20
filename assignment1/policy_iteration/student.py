import numpy as np

def reward_function(s, env_size):
    #print("ENV_SIZE: " + str(env_size))
    #print("STATE in reward_function is: "  + str(s))
    if (s == [env_size,env_size]).all():
        return 1
    return 0

def transition_function(s, a, directions,env_size,holes):
        s_prime = s + directions[a,:]

        if s_prime[0] < env_size and s_prime[1] < env_size and (s_prime >= 0).all():
            if holes[s_prime[0], s_prime[1]] == 0 :
                return s_prime

        return s

def transition_probabilities(env, s, a, env_size, directions, holes):
    cells = []
    probs = []
    prob_next_state = np.zeros((env_size, env_size))
     
    s_prime = transition_function(s, a, directions,env_size,holes)
   
    prob_next_state[s_prime[0],s_prime[1]] =  1
    
    return prob_next_state
