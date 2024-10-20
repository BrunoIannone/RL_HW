import numpy as np


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
        action = ...
        click = env.step(action)
        views[state, action] = ...
        clicks[state, action] = ...
        Q[state, action] = ...
        total_reward = ...
        best_action = env.CTR[state,:].argmax()
        regret += ...
        Qs.append(Q.copy())

    # Commit
    for i in range(iters-explore_steps):
        state = env.observe()
        action = ...
        click = env.step(action)
        total_reward = ...
        regret += ...

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