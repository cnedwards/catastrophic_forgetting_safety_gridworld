# ECE Final Project
#Based off https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from ai_safety_gridworlds.environments.distributional_shift import DistributionalShiftEnvironment 
from ai_safety_gridworlds.environments.safe_interruptibility import SafeInterruptibilityEnvironment 
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment
from ai_safety_gridworlds.environments.shared import safety_ui
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared.safety_game import Actions

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time

import random
import math
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# ===============================================================
# Environment constants
WIDTH = 9
HEIGHT = 7
NUM_OBJECTS = 8
oneHotMap = {}
oneHotMap[0, 110, 119] = [1, 0, 0, 0, 0, 0, 0, 0] # Box
oneHotMap[255, 0, 0] = [0, 1, 0, 0, 0, 0, 0, 0] # Lava
oneHotMap[219, 219, 219] = [0, 0, 1, 0, 0, 0, 0, 0] # Path
oneHotMap[0, 180, 255] = [0, 0, 0, 1, 0, 0, 0, 0] # Agent
oneHotMap[152, 152, 152] = [0, 0, 0, 0, 1, 0, 0, 0] # Wall
oneHotMap[0, 210, 50] = [0, 0, 0, 0, 0, 1, 0, 0] # Goal
oneHotMap[110, 69, 210] = [0, 0, 0, 0, 0, 0, 1, 0] # Button
oneHotMap[255, 30, 255] = [0, 0, 0, 0, 0, 0, 0, 1] # Interruption

# Tried to match the catastrophic forgetting paper
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_LENGTH = 900000 # Number of timesteps 
TARGET_UPDATE = 10
MAX_TIMESTEPS = 1000000
device = torch.device("cpu")
n_actions = 4 # up, down, left, right
# ===============================================================
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))
# ===============================================================
class ReplayMemory(object):
    def __init__(self, capacity):
        """
        Initializes a ReplayMemory instance
        Args:
            capacity: Int representing the size of the memory buffer
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
    # ===============================================
    def push(self, *args):
        """
        Saves a single transition
        Args:
            *args: Transition tuple that defines state, action, next_state, and reward
        """
        if len(self.memory) < self.capacity:    # If there is room for new transition
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    # ===============================================
    def sample(self, batch_size):
        """
        Samples Transitions from the ReplayMemory
        Args:
            batch_size: size of the batch to sample
        Returns:
            Samples from the ReplayMemory
        """
        return random.sample(self.memory, batch_size)
    # ===============================================
    def __len__(self):
        """
        Gets the length of the ReplayMemory
        Returns:
            The length of the ReplayMemory
        """
        return len(self.memory)
# ===============================================================
class DQN(nn.Module):
    def __init__(self, n_actions):
        """
        Initializes a DQN instance
        Args:
            n_actions: Number of possible actions in the environment
        """
        super(DQN, self).__init__()
        
        self.inputSize = WIDTH*HEIGHT
        self.hidden1Size = 100
        self.hidden2Size = 100
        self.outputSize = n_actions
        self.hidden1 = nn.Linear(self.inputSize, self.hidden1Size) # construct hidden and output layers for the network
        self.hidden2 = nn.Linear(self.hidden1Size, self.hidden2Size)
        self.output = nn.Linear(self.hidden2Size, self.outputSize)
    # ===============================================   
    def forward(self, x):
        """
        Passes a state through the network
        Args:
            x: current state
        Returns:
            The Q Values for each action
        """
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x) # Can't have ReLU here because then we couldn't have negative action values
        return x 
# ===============================================================
def optimize_model():
    """
    Optimizes the model using Adam
    """
    if len(memory) < BATCH_SIZE:
        return
        
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple([not a for a in batch.terminal]), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state])[non_final_mask]
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask.bool()] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #MSE Loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
# ===============================================================
def get_onehot_state(timestep): 
    """
    Converts the state to one-hot state
    Args:
        timestep: An instance of AI safety gridworld environment timestep
    Returns:
        The one-hot encoded state
    """
    gridHeight = len(timestep.observation['RGB'][0])   # True height of the environment grid
    gridWidth = len(timestep.observation['RGB'][0][0]) # True width of the environment grid

    grid = timestep.observation['RGB']  # Get current grid configuration
    state = np.zeros((HEIGHT, WIDTH, NUM_OBJECTS))
    for i in range(HEIGHT): # One hot encoding grid into state
        for j in range(WIDTH):
            if i >= gridHeight or j >= gridWidth: # If the grid is smaller than the state size set as "wall"
                state[i][j] = oneHotMap[152, 152, 152]
            else:
                values = oneHotMap[grid[0][i][j], grid[1][i][j], grid[2][i][j]]
                for k in range(len(values)):
                    state[i][j][k] = values[k]
    s = torch.from_numpy(state).reshape(-1) # reshape to vector for input to NN
    tmp = torch.zeros(WIDTH*HEIGHT*NUM_OBJECTS) 
    for i in range(state.shape[0]):
        tmp[i] = s[i] 
    return tmp
# ===============================================================
def select_action(state):
    """
    Selects the best action for the given state
    Args:
        state: current state
    Returns:
        The best action for the given state
    """
    sample = random.random()
    global steps_done
    eps_threshold = np.max([EPS_END, EPS_END + (EPS_START - EPS_END) * (1 - steps_done / EPS_LENGTH)]) #linearly anneal
    if sample > eps_threshold: # Greedy action
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net.forward(state).argmax().view(1,1)
    else:   # Non-greedy action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
# ===============================================================
def moving_average(data, window_size=100): #used this approach https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """
    Calculates a moving average for all the data
    Args:
        data: set of values
        window_size: number of data points to consider in window
    Returns:
        Moving average of the data
    """    
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

def plot_episode():
    """
    Prints an episode starting at initial position
    """   
    timestep = env.reset()
    while 'termination_reason' not in env._environment_data:
        # Select and perform an action
        print(target_net.forward(state))
        action = target_net.forward(state).argmax().view(1,1)
        #Observe new state
        timestep = env.step(action)
        print(action)
        print(timestep.observation['board'])
# ===============================================================

env = DistributionalShiftEnvironment(is_testing=False)
#env = SafeInterruptibilityEnvironment()
#env = SideEffectsSokobanEnvironment()
policy_net = DQN(n_actions).to(device) # Initialize the behavior policy network
target_net = DQN(n_actions).to(device)  # Initialize the target policy network
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()
memory = ReplayMemory(10000)
global steps_done
steps_done = 0  
timestep = env.reset()  # Set the environment to initial state
gridHeight = len(timestep.observation['RGB'][0])   # True height of the environment grid
gridWidth = len(timestep.observation['RGB'][0][0]) # True width of the environment grid
episodeHistory = [deepcopy(timestep.observation['board'])]  # History of all grid configurations encountered      

#Training loop
episode_durations = []
episode_return = []
steps_at_episode = []
t0 = time.time()
i_episode = 0
while steps_done < MAX_TIMESTEPS: 
    i_episode +=1
    # Initialize the environment and state
    timestep = env.reset()
    state = torch.from_numpy(timestep.observation['board']).reshape(-1)
    t = 0 # Count number of states in episode
    returns = 0
    
    while 'termination_reason' not in env._environment_data: # While not a terminal state
        # Select and perform an action
        action = select_action(state)
        # Observe new state
        timestep = env.step(action)
        next_state = torch.from_numpy(timestep.observation['board']).reshape(-1)
        reward = torch.tensor([timestep.reward], device=device)
        t+=1
        returns += timestep.reward
        # Store the transition in memory
        memory.push(state, action, next_state, reward, 'termination_reason' in env._environment_data)
        # Move to the next state
        state = next_state
        optimize_model()
    else:
        steps_done += t
        episode_durations.append(t)
        episode_return.append(returns)
        steps_at_episode.append(steps_done)
        #plot_durations()
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if i_episode % 100 == 0: 
        print(i_episode, steps_done, time.time()-t0)
    
    if i_episode % 1000 == 0:  # Save image of training every 1000 episodes
        #print(episode_return)
        fig = plt.figure()
        plt.plot(steps_at_episode, episode_return)
        plt.plot(steps_at_episode[99:], moving_average(episode_return))
        plt.xlabel("Steps")
        plt.ylabel("Return")
        plt.savefig('figures/out_noonehot_'+str(steps_done)+'.png')

print("Time: ", time.time()-t0)
print('Complete')
plt.figure()
plt.plot(steps_at_episode, episode_return)
plt.plot(steps_at_episode[99:], moving_average(episode_return))
plt.xlabel("Steps")
plt.ylabel("Return")
plt.savefig('out.png')
plot_episode()

