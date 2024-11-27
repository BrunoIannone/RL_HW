import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils ##
from typing import List ##
from collections import namedtuple, deque ## 
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):  # Puoi cambiare il numero di classi
        super(SimpleCNN, self).__init__()
        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # output: 16 x 84 x 96
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 16 x 42 x 48

        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # output: 32 x 42 x 48
        # Dopo MaxPooling: output: 32 x 21 x 24

        # Terzo blocco convoluzionale
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # output: 64 x 21 x 24
        # Dopo MaxPooling: output: 64 x 10 x 12

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=64 * 10 * 12, out_features=128)  # Primo livello fully connected
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)  # Output finale

    def forward(self, x):
        # Passaggio nei layer convoluzionali con ReLU e MaxPooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> MaxPool
        
        # Flatten per il passaggio al fully connected
        x = x.view(-1, 64 * 10 * 12)  # Flatten (batch_size, features)
        
        # Passaggio nei layer fully connected
        x = F.relu(self.fc1(x))  # Primo fully connected con ReLU
        x = self.fc2(x)  # Output finale
        return x

class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        self.activation_function= nn.Tanh()

        self.layer1 = nn.Linear( #<--- linear layer
            n_inputs, #<----------------#input features
            64,#<-----------------------#output features
            bias=bias)#<----------------bias

        self.layer2 = nn.Linear(
            64,
            32,
            bias=bias)

        self.layer3 = nn.Linear(
                    32,
                    n_outputs,
                    bias=bias)


    def forward(self, x):
        x = self.activation_function( self.layer1(x) )
        x = self.activation_function( self.layer2(x) )
        y = self.layer3(x)

        return y

class Q_network(nn.Module):

    def __init__(self, env,  learning_rate=1e-4):
        super(Q_network, self).__init__()

        n_outputs = env.action_space.n

        #self.network = Net( ?? , ??)
        print( env.observation_space._shape[0], env.action_space.n)
        self.network = Net( env.observation_space._shape[0], env.action_space.n)
        print("Q network:")
        print(self.network)

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          1e-3)#lr=learning_rate)

    def greedy_action(self, state):
        # greedy action = ??
        # greedy_a = 0
        print(state.shape)
        qvals = self.get_qvals(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a

    def get_qvals(self, state):
        #out = ???
        #print(state.shape)
        #state = state[:, :-12, :]
        out = self.network(state)
        return out
    
class Experience_replay_buffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, s_0, a, r, d, s_1):
        self.replay_memory.append(
            self.Buffer(s_0, a, r, d, s_1))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size
    

def from_tuple_to_tensor(tuple_of_np):
    tensor = torch.zeros((len(tuple_of_np), tuple_of_np[0].shape[0]))
    for i, x in enumerate(tuple_of_np):
        tensor[i] = torch.FloatTensor(x)
    return tensor


class DDQN_agent:

    def __init__(self, env, rew_thre, buffer,device, learning_rate=0.001, initial_epsilon=0.5, batch_size= 64):

        self.env = env
        self.device = device

        self.network = Q_network(env, learning_rate).to(self.device)
        #self.target_network = ???
        self.target_network = Q_network(env, learning_rate).to(self.device)
        # Or if you want the two networks to start with the same random weights:
        # self.target_network = deepcopy(self.network)
        self.buffer = buffer
        self.epsilon = initial_epsilon
        self.batch_size = batch_size
        self.window = 50
        self.reward_threshold = rew_thre
        self.initialize()
        self.step_count = 0
        self.episode = 0


    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        #if mode == 'explore':
           # action = self.env.action_space.sample()
        #else:
        # Assuming self.s_0 has shape 1x84x3x96
        self.s_0 = torch.FloatTensor(self.s_0)  # Removes the first dimension -> 84x3x96

        # Permute to change the order of dimensions
        # From (84, 3, 96) to (3, 96, 84)
        self.s_0 = self.s_0.permute(2, 1, 0)

        # Ensure it's moved to the correct device
        self.s_0 = self.s_0.to(self.device)
        #print(self.s_0.shape)
        action = self.network.greedy_action(self.s_0)

        #simulate action
        #print(action)
        s_1, r, terminated, truncated, _ = self.env.step(np.int32(action)) ##TODO MODIFY FOR CONTINUOUS
        done = terminated or truncated

        #put experience in the buffer
        self.buffer.append(self.s_0, action, r, terminated, s_1)

        self.rewards += r

        self.s_0 = s_1.copy()

        self.step_count += 1
        if done:
            self.s_0, _ = self.env.reset()
        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=10000,
              network_update_frequency=10,
              network_sync_frequency=200):
        self.gamma = gamma

        self.loss_function = nn.MSELoss()
        self.s_0, _ = self.env.reset()

        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
        ep = 0
        training = True
        self.populate = False
        while training:
            self.s_0, _ = self.env.reset()

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
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
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

                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        #break
        # save models
        self.save_models()
        # plot
        self.plot_training_rewards()

    def save_models(self):
        torch.save(self.network, "Q_net")

    def load_models(self):
        self.network = torch.load("Q_net")
        self.network.eval()

    def plot_training_rewards(self):
        plt.plot(self.mean_training_rewards)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        plt.show()
        plt.savefig('mean_training_rewards.png')
        plt.clf()

    def calculate_loss(self, batch):
        #extract info from batch
        states, actions, rewards, dones, next_states = list(batch)

        #transform in torch tensors
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(self.device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(self.device)
        states = from_tuple_to_tensor(states).to(self.device)
        next_states = from_tuple_to_tensor(next_states).to(self.device)

        ###############
        # DDQN Update #
        ###############
        # Q(s,a) = ??
        qvals = self.network.get_qvals(states)
        qvals = torch.gather(qvals, 1, actions)

        # target Q(s,a) = ??
        next_qvals= self.target_network.get_qvals(next_states)
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        target_qvals = rewards + (1 - dones)*self.gamma*next_qvals_max

        # loss = self.loss_function( Q(s,a) , target_Q(s,a))
        loss = self.loss_function(qvals, target_qvals)

        return loss


    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)

        loss.backward()
        self.network.optimizer.step()

        self.update_loss.append(loss.item())

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0

    def evaluate(self, eval_env):
        done = False
        s, _ = eval_env.reset()
        rew = 0
        while not done:
            action = self.network.greedy_action(torch.FloatTensor(s).to(self.device))
            s, r, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            rew += r

        print("Evaluation cumulative reward: ", rew)

class Policy(nn.Module):
    continuous = True # you can change this

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
        return 0

    def train(self):
        env = gym.make('CarRacing-v2', continuous=False)#, render_mode = "human")
        print(env.action_space)
        #print(env.observation_space.sample())
        rew_threshold = 400
        buffer = Experience_replay_buffer()
        agent = DDQN_agent(env, rew_threshold, buffer,self.device)
        agent.train()
        eval_env = gym.make("CarRacing-v2", continuous = False, render_mode="human")
        agent.evaluate(eval_env)

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
