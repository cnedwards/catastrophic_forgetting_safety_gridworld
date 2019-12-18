#Used https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html as a guide

from mylavaworld import LavaWorld

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
WIDTH = 4
HEIGHT = 4
NUM_OBJECTS = 4
# ===============================================================

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        self.inputSize = WIDTH*HEIGHT*NUM_OBJECTS
        self.hidden1Size = 100
        self.hidden2Size = 100
        self.outputSize = n_actions

        self.hidden1 = nn.Linear(self.inputSize, self.hidden1Size)
        self.hidden2 = nn.Linear(self.hidden1Size, self.hidden2Size)
        self.output = nn.Linear(self.hidden2Size, self.outputSize)
        
        #self.W1 = torch.randn(self.inputSize, self.hidden1Size) # 10 X ? tensor
        #self.W2 = torch.randn(self.hidden1Size, self.hidden2Size) # 3 X 2 tensor
        #self.W3 = torch.randn(self.hidden2Size, self.outputSize) # 3 X 1 tensor
    
    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        
        #print("at x", x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x) #Can't have ReLU here because then we couldn't have negative action values
        
        #x = self.hidden1(x)
        #x = self.hidden2(x)
        #x = self.output(x)
        
        return x#self.head(x.view(x.size(0), -1))
        
        

env = LavaWorld()
#env = SafeInterruptibilityEnvironment()
#env = SideEffectsSokobanEnvironment()
#totalReturn, episodeHistory = runEpisode(env)

#for episode in episodeHistory:
#    print(episode)

agent = Agent()


#dqn = DQN()
#print(dqn.forward(torch.randn(WIDTH*HEIGHT*NUM_OBJECTS)))

#Tried to match paper
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
TARGET_UPDATE = 10
MAX_TIMESTEPS = 25000
EPS_LENGTH = int(0.9 * MAX_TIMESTEPS) #timesteps 

device = torch.device("cpu")

# Get number of actions from gym action space
n_actions = 4

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#print(policy_net.forward(torch.randn(WIDTH*HEIGHT*NUM_OBJECTS)))

#optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()
memory = ReplayMemory(1000)

steps_done = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
        
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
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

    #print(state_action_values, "\n", expected_state_action_values.unsqueeze(1), "\n")
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

    
#Function for converting state
timestep = env.reset()  # Set the environment to initial state
#gridHeight = len(timestep.observation['RGB'][0])   # True height of the environment grid
#gridWidth = len(timestep.observation['RGB'][0][0]) # True width of the environment grid
#episodeHistory = [deepcopy(timestep.observation['board'])]  # History of all grid configurations encountered
    

    
#timestep = env.reset()
#s = get_onehot_state(timestep)
#print(torch.randn(WIDTH*HEIGHT*NUM_OBJECTS).shape)
#print(policy_net.forward(torch.randn(WIDTH*HEIGHT*NUM_OBJECTS)))
#print(s)
#print(policy_net.forward(s)) #this breaks for some reason

#This doesn't:
#tmp_test = torch.randn(WIDTH*HEIGHT*NUM_OBJECTS)
#for i in range(s.shape[0]):
#    tmp_test[i] = s[i]

#zz

#print(timestep)
#print(get_onehot_state(timestep))

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = np.max([EPS_END, EPS_END + (EPS_START - EPS_END) * (1 - steps_done / EPS_LENGTH)]) #linearly anneal...
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            return policy_net.forward(state).argmax().view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def moving_average(data, window_size=100): #used this approach https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

        
#print(select_action(torch.randn(WIDTH*HEIGHT*NUM_OBJECTS)))
#zz
#Training loop
episode_durations = []
episode_return = []
steps_at_episode = []

t0 = time.time()
#num_episodes = 110
#for i_episode in range(num_episodes):
i_episode = 0
while steps_done < MAX_TIMESTEPS: 
    i_episode +=1
    # Initialize the environment and state
    #Stupid workaround
    tmp = torch.tensor(env.reset().reshape(-1), device = device)
    state = torch.zeros(WIDTH*HEIGHT*NUM_OBJECTS) 
    for i in range(tmp.shape[0]):
        state[i] = tmp[i]
    
    #for t in count():
    t = 0 #Count number of states in episode
    returns = 0
    
    while not env.check_end():
        #print(t)
        # Select and perform an action
        action = select_action(state)
        #Observe new state
        next_state, reward = env.step(action.data)
        
        #Stupid workaround
        tmp = torch.tensor(next_state.reshape(-1), device = device)
        next_state = torch.zeros(WIDTH*HEIGHT*NUM_OBJECTS) 
        for i in range(tmp.shape[0]):
            next_state[i] = tmp[i]
        #next_state = torch.tensor([next_state], device = device)
        
        returns += reward
        reward = torch.tensor([reward], device=device)
        t+=1
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        #t1=time.time()
        optimize_model()
        #print(time.time()-t1)
    else:
        steps_done += t+1
        episode_durations.append(t+1)
        episode_return.append(returns)
        steps_at_episode.append(steps_done)
        #plot_durations()
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if i_episode % 100 == 0: 
        print(i_episode, steps_done, time.time()-t0)

        #timestep = env.reset()
        #state = torch.from_numpy(timestep.observation['board']).reshape(-1)
        #print(policy_net(state))
    
    if i_episode % 1000 == 0: 
        #print(episode_return)
        plt.figure()
        plt.plot(steps_at_episode, episode_return)
        plt.plot(steps_at_episode[99:], moving_average(episode_return))
        plt.xlabel("Steps")
        plt.ylabel("Return")
        plt.savefig('figures/out_mylavaworld_'+str(steps_done)+'.png')

print("Time: ", time.time()-t0)
 
print('Complete')

#print(episode_return)
plt.figure()
plt.plot(steps_at_episode, episode_return)
plt.plot(steps_at_episode[99:], moving_average(episode_return))
plt.xlabel("Steps")
plt.ylabel("Return")
#plt.title(r"zzz")
plt.savefig('out.png')


def plot_episode():
    tmp = torch.tensor(env.reset().reshape(-1), device = device)
    state = torch.zeros(WIDTH*HEIGHT*NUM_OBJECTS) 
    for i in range(tmp.shape[0]):
        state[i] = tmp[i]
        
    env.print_state()
    
    while not env.check_end():
        # Select and perform an action
        #print(target_net(state))
        Q = target_net(state)
        action = Q.argmax().view(1,1)
        #Observe new state
        state, reward = env.step(action.data)
        tmp = torch.tensor(state.reshape(-1), device = device)
        state = torch.zeros(WIDTH*HEIGHT*NUM_OBJECTS) 
        for i in range(tmp.shape[0]):
            state[i] = tmp[i]
        
        print(Q)
        print(action, reward)
        env.print_state()

plot_episode()

