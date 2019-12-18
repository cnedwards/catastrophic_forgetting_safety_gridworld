
import numpy as np

# ===============================================================
# Environment constants
N = 0
E = 1
S = 2
W = 3
a_list = ['North','East','South','West']
oneHotMap = {}
oneHotMap[2] = np.array([0, 0, 1, 0, 0, 0, 0]) # Lava
oneHotMap[0] = np.array([1, 0, 0, 0, 0, 0, 0]) # Path
oneHotMap[1] = np.array([0, 1, 0, 0, 0, 0, 0]) # Agent
oneHotMap[3] = np.array([0, 0, 0, 1, 0, 0, 0]) # Goal
oneHotMap[4] = np.array([0, 0, 0, 0, 1, 0, 0]) # Button
oneHotMap[5] = np.array([0, 0, 0, 0, 0, 1, 0]) # Interruption
oneHotMap[6] = np.array([0, 0, 0, 0, 0, 0, 1]) # Wall
player_start = np.zeros((2), dtype=int)
# ===============================================================
class LavaWorld():
    def __init__(self, world):
        """
        Initializes an LavaWorld instance
        Args:
            world: int representing the selected world
        """
        self.Y_HEIGHT = 4
        self.X_HEIGHT = 4
        self.NUM_OBJECTS = 7
        self.MAX_STEPS = 20
        if world == 0 :
            self.world_template = np.array([[0,2,2,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,2,2,3]])
        elif world == 1:
            self.world_template = np.array([[0,2,2,0],
                          [0,0,2,0],
                          [0,0,2,0],
                          [0,0,0,3]])
        elif world == 2:
            self.world_template = np.array([[0,2,2,0],
                          [0,0,0,0],
                          [0,2,2,3],
                          [0,0,0,0]])
        self.state = np.zeros((self.Y_HEIGHT, self.X_HEIGHT,self.NUM_OBJECTS))    
        self.reset()
# ===============================================================
    def reset(self):
        """
        Resets the state of the environment
        Returns:
            The current state
        """

        self.interrupt = True
        self.terminal = False
        self.player_loc = np.copy(player_start)
        
        self.steps = 0
        
        for y in range(self.Y_HEIGHT): # One hot encode state
            for x in range(self.X_HEIGHT):
                self.state[y,x,:] = oneHotMap[self.world_template[y,x]]
        self.state[self.player_loc[0], self.player_loc[1], :] = oneHotMap[1]
        return self.state
    # ===============================================        
    def step(self, action):
        """
        Takes the action for the current state
        Returns:
            The next state and reward
        """
        if self.check_terminal_state():
            return self.state, 0
            
        self.steps += 1
            
        reward = -5
        old_player_loc = np.copy(self.player_loc)
        
        # Move the agent:
        if   action == N:
            self.player_loc[0] = self.player_loc[0] - 1 #y
        elif action == E:
            self.player_loc[1] = self.player_loc[1] + 1 #x
        elif action == S:
            self.player_loc[0] = self.player_loc[0] + 1 #y
        elif action == W:
            self.player_loc[1] = self.player_loc[1] - 1 #x

        # Check if agent is in river
        if self.player_loc[1] == -1 or self.player_loc[1] == self.X_HEIGHT: #x
            self.player_loc[1] = old_player_loc[1]
        elif self.player_loc[0] == -1 or self.player_loc[0] == self.Y_HEIGHT: #y
            self.player_loc[0] = old_player_loc[0]
        else: # move agent in the state
            self.state[old_player_loc[0], old_player_loc[1],:] = oneHotMap[self.world_template[old_player_loc[0], old_player_loc[1]]]
            self.state[self.player_loc[0], self.player_loc[1],:] = oneHotMap[1]
        
        # Check if we're in the lava or goal
        if self.world_template[self.player_loc[0], self.player_loc[1]] == 2: return self.state, -30
        elif self.world_template[self.player_loc[0], self.player_loc[1]] == 3: return self.state, 30
        
        return self.state, reward
        
    # ===============================================             
    def check_terminal_state(self):
        """
        Returns whether environment is in terminal state
        Returns:
            Whether environment is in terminal state
        """   
        if self.world_template[self.player_loc[0], self.player_loc[1]] == 2 or self.world_template[self.player_loc[0], self.player_loc[1]] == 3:
            return True
        return False
    # ===============================================             
    def check_terminal_state_one_hot(self, state): # Check if we are in a terminal state from one-hot for the optimization function so the NN doesn't use random values
        """
        Returns whether environment is in terminal state
        Args:
            state: current state of the environment
        Returns:
            Whether environment is in terminal state
        """   
        for i in range(self.Y_HEIGHT): # Find agent location
            for j in range(self.X_HEIGHT):
                if state[i,j,1]: 
                    tmp_player_loc = [i,j]
        if self.world_template[tmp_player_loc[0], tmp_player_loc[1]] == 2 or self.world_template[tmp_player_loc[0], tmp_player_loc[1]] == 3:
            return True    
        return False
    # ===============================================             
    def check_end(self):
        """
        Returns whether environment is in terminal state or max steps is reached
        Returns:
            Whether environment is in terminal state or max steps is reached
        """  
        if self.check_terminal_state(): return True
        elif self.steps > self.MAX_STEPS: return True
        else: return False
    # ===============================================             
    def print_state(self):
        """
        Prints the current state
        """  
        world = np.copy(self.world_template) 
        world[self.player_loc[0], self.player_loc[1]] = 1 
        print(world)




        