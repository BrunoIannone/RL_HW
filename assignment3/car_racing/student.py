import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List ##
from collections import namedtuple, deque ## 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cnn
import gc
from exp_replay_buff import *


class Q_network(nn.Module):

    def __init__(self, env,  learning_rate=1e-4):
        super(Q_network, self).__init__()

        n_outputs = env.action_space.n

        self.network = cnn.SimpleCNN(n_outputs)

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=learning_rate)

    def greedy_action(self, state):
       
        qvals = self.get_qvals(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a

    def get_qvals(self, state):
        
        out = self.network(state)
        return out
    
class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make('CarRacing-v2', continuous=False,render_mode="rgb_array")#, render_mode = "human")
        self.reward_threshold = 400
        self.buffer = Experience_replay_buffer()
        #self.agent = DDQN_agent(self.env, self.reward_threshold, self.buffer,self.device)  
        self.learning_rate = 1e-3
        self.network = Q_network(self.env, self.learning_rate).to(self.device)
        self.target_network = Q_network(self.env, self.learning_rate).to(self.device)
        # Or if you want the two networks to start with the same random weights:
        # self.target_network = deepcopy(self.network)
        self.epsilon = 0.5
        self.batch_size = 64
        self.window = 50
        self.step_count = 0
        self.episode = 0

        #### TRAIN PARAMS #### 
        self.gamma=0.99
        self.max_episodes=10000
        self.network_update_frequency=10
        self.network_sync_frequency=200
        ###############################


        #### PARAMS ####
        #self.batch = 10 #minibatch k
        self.step_size = 0.1 # eta (n greca)
        self.replay_period = 10 # K (capital k)
        self.size = 10 # N
        self.alpha = 1
        self.beta = 1
        self.budget = 10 # T
        #self.replay_memory = [] # H, use self.buffer
        self.delta = 0
        self.p1 = 1

        self.initialize()


    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        if mode == 'explore':
            action = self.env.action_space.sample()
        
        else:

            #Assuming self.s_0 has shape 1x84x3x96
            self.s_0 = self.handle_state_shape(self.s_0,self.device)
            
            action = self.network.greedy_action(self.s_0)

        s_1, r, terminated, truncated, _ = self.env.step(action) ##TODO MODIFY FOR CONTINUOUS
        s_1 = self.handle_state_shape(s_1,self.device)

        done = terminated or truncated

        #put experience in the buffer
        self.buffer.append(self.s_0, action, r, terminated, s_1)

        self.rewards += r

        self.s_0 = s_1.detach().clone()

        self.step_count += 1

        if done:
            self.s_0, _ = self.env.reset()
            self.s_0 = self.handle_state_shape(self.s_0,self.device)
        return done
    
        
    def calculate_loss(self, batch):
        #extract info from batch
        states, actions, rewards, dones, next_states = list(batch)

        #transform in torch tensors
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(self.device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(self.device)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)

        ###############
        # DDQN Update #
        ###############
        
        qvals = self.network.get_qvals(states)
        qvals = torch.gather(qvals, 1, actions)

        next_qvals= self.target_network.get_qvals(next_states)
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        target_qvals = rewards + (1 - dones)*self.gamma*next_qvals_max

        #                       Q(s,a) , target_Q(s,a)
        loss = self.loss_function(qvals, target_qvals)

        return loss
    
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)

        loss = self.calculate_loss(batch)

        loss.backward()
        self.network.optimizer.step()

        self.update_loss.append(loss.item())
    
    
    def forward(self, x): # TODO
    
        return x


    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
    
    def handle_state_shape(self,s_0,device):
        if s_0.shape == torch.Size([3, 84, 96]): # Ensures no further crops
            return s_0
        
        s_0 = torch.FloatTensor(s_0)  

        # Permute to change the order of dimensions
        # From (84, 3, 96) to (3, 96, 84)
        s_0 = s_0.permute(2, 1, 0)
        s_0 =  s_0[:, :-12, :]
        
        s_0 = s_0.to(device)
        return s_0
    
    def act(self, state): #returns action for s = env.step(action)
        state = self.handle_state_shape(state,self.device)
        return self.network.greedy_action(state)
         

    def train(self):
               
        self.loss_function = nn.MSELoss()
        self.s_0, _ = self.env.reset()
        self.s_0 = self.handle_state_shape(self.s_0,self.device)

        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
        ep = 0
        training = True
        self.populate = False

        while training: #Begin training
            self.s_0, _ = self.env.reset()
            self.s_0 = self.handle_state_shape(self.s_0,self.device)

            self.rewards = 0
            done = False
            while not done:
                if ((ep % 5) == 0):
                    self.env.render()

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                    # print("explore")
                else:
                    done = self.take_step(mode='exploit')
                    # print("train")
                # Update network
                if self.step_count % self.network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % self.network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    if self.epsilon >= 0.05:
                        self.epsilon = self.epsilon * 0.7
                    ep += 1
                    if self.rewards > 2000:
                        self.training_rewards.append(2000)
                    elif self.rewards > 1000:
                        self.training_rewards.append(1000)
                    elif self.rewards > 500:
                        self.training_rewards.append(500)
                    else:
                        self.training_rewards.append(self.rewards)
                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    mean_loss = np.mean(self.training_loss[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}   mean loss = {:.2f}\t\t".format(
                            ep, mean_rewards, self.rewards, mean_loss), end="")

                    if ep >= self.max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        #break

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
