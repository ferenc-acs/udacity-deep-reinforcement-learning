import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64, fc3_units=32): # C&P
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        #################### COPY PASTE & MODIFY START ####################
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.dr1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dr2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        
        #################### COPY PASTE & MODIFY STOP ####################
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        #################### COPY PASTE & MODIFY START ####################
        x = F.relu(self.fc1(state))
        x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.dr2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        #################### COPY PASTE & MODIFY STOP ####################

    
