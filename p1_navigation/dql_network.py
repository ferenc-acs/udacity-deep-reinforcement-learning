import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    """My Deep Q Network"""
    
    # Go for an architecture that worked for the lunar lander mini project
    # Had a simple architecture with two dropout layers.
    def __init__( self, state_size, action_size, seed, fc_units = (128, 64, 32) ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units ( tuple(int), dim = (3) ): Hidden Layers one to four: number of neurons
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.dr1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.dr2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(fc_units[1], fc_units[2])
        self.fc4 = nn.Linear(fc_units[2], action_size)
        
    # Define forward propagation through the network    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state))
        x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.dr2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        