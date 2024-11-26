import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device

        #### PARAMS ####
        self.batch = 10 #minibatch k
        self.step_size = 0.1 # eta (n greca)
        self.replay_period = 10 # K (capital k)
        self.size = 10 # N
        self.alpha = 1
        self.beta = 1
        self.budget = 10 # T
        self.replay_memory = [] # H
        self.delta = 0
        self.p1 = 1

    def forward(self, x): # TODO
        
        return x
    
    def act(self, state):
        # TODO
        return 

    def train(self):
        env = gym.make('CarRacing-v3', continuous=True)
        print(env.sample())
        for t in range(1,self.budget):
            pass
            #return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
