import numpy as np

def compute_reward(env,state,action,views,clicks,click,Q):
    # if(views[state, action]-1 > 0):
    #     old_p = (clicks[state, action]-click)/(views[state, action]-1)
    #     old_r = old_p * Q[state, action]
    #     new_p = (clicks[state, action])/(views[state, action])
    #     step_reward = env.CTR[state][action]*click
    #     new_r = new_p*(old_r + step_reward)
    #     return new_r
    # else:
    #     new_p = (clicks[state, action])/(views[state, action])
    #     new_r = new_p*(env.CTR[state][action]*click)
    #     return new_r

    return (env.CTR[state][action]*click)/(1/env.n_actions)

    

def explore_and_commit(env, explore_steps = 50, iters = 200):
    clicks = np.zeros((env.n_states, env.n_actions))
    views = np.zeros((env.n_states, env.n_actions))
    Q = np.zeros((env.n_states, env.n_actions))
    Qs = []
    total_reward = 0.
    regret = 0.

    # Explore
    for i in range(explore_steps):
        state = env.observe()
        action = np.random.randint(0, env.n_actions) #TODO
        print(i,state,action)

        #print(env.CTR[state][action])
        click = env.step(action)
        views[state, action] += 1 #TODO
        print("VIEWS: " + str(views[state, action]))

        clicks[state, action] += click #TODO
        print("CLICKS: " + str(clicks[state, action]))

        Q[state, action] += compute_reward(env,state,action,views,clicks,click,Q)
        print(Q)
        print("ESTIMATOR: " + str(Q[state, action]))

        print("STEP REWARD: " + str(env.CTR[state][action]))
        total_reward += env.CTR[state][action] * click #TODO
        print("TOTAL REWARD: " + str(total_reward))
        best_action = env.CTR[state,:].argmax()
        #print("BEST REWARD: " + str(env.CTR[state][best_action]))
        regret += env.CTR[state][best_action] - env.CTR[state][action] * click #TODO
        #print("TOTAL REGRET: " + str(regret))

        Qs.append(Q.copy())
    #print("START COMMIT")
    # Commit
    for i in range(iters-explore_steps):
        state = env.observe()
        #print(best_action)
        action = Q[state].argmax()
        click = env.step(action)
        total_reward += env.CTR[state][action] * click
        best_action = env.CTR[state,:].argmax()
        regret += env.CTR[state][best_action] - env.CTR[state][action] * click

    return Qs, total_reward, regret


def epsilon_greedy(env, epsilon = 0.1, null_epsilon_after = 50, iters = 200):
    clicks = np.zeros((env.n_states, env.n_actions))
    views = np.zeros((env.n_states, env.n_actions))
    Q = np.zeros((env.n_states, env.n_actions))
    Qs = []
    total_reward = 0.
    regret = 0.

    # "Explore" (epsilon is non-zero)
    for i in range(null_epsilon_after):
        state = env.observe()
        action = ...
        click = env.step(action)
        views[state, action] = ...
        clicks[state, action] = ...
        Q[state, action] = ...
        total_reward = ...
        best_action = env.CTR[state,:].argmax()
        regret += ...
        Qs.append(Q.copy())

    # "Commit" (epsilon set to 0)
    for i in range(iters-null_epsilon_after):
        state = env.observe()
        action = ...
        click = env.step(action)
        total_reward = ...
        best_action = env.CTR[state,:].argmax()
        regret += ...

    return Qs, total_reward, regret