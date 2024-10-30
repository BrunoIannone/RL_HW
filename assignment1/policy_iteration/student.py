import numpy as np
import time
def reward_function(s, env_size):
    
    if (s == [env_size-1,env_size-1]).all():
        return 1
    return 0

def is_state_valid(s_prime,env_size):
    if s_prime[0] < env_size and s_prime[1] < env_size and (s_prime >= 0).all():
        return True
    return False

def transition_probabilities(env, s, a, env_size, directions, holes):
    probs = []

    prob_next_state = np.zeros((env_size, env_size))
    
    s_prime_correct = s + directions[a]
    if (is_state_valid(s_prime_correct,env_size)):
        prob_next_state[s_prime_correct[0],s_prime_correct[1]] = 0.5
        probs.append(0.5)
        
    if(a==0):
        s_prime_random = s + directions[len(directions)-1]
    else:
        s_prime_random = s + directions[a-1]
    
    if (is_state_valid(s_prime_random,env_size)):
        prob_next_state[s_prime_random[0],s_prime_random[1]] =  0.5
        probs.append(0.5)

    sump = sum(probs)
    prob_next_state[s[0], s[1]] = 1 - sump
    return prob_next_state
