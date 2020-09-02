import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Format: IN_Num [Layer 1] (OUT_Num = IN_Num) [Layer 2] OUT_Num = ...
HIDDEN_DIMS_DEFAULT = {
    'shared' : (512, 512, 256, 256), #Three hidden layers
    'actor' : (256, 128, 128, 64), #Three hidden layers
    'critic' : (256, 128, 128, 64) #Three hidden layers
}

# Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
# Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb
class A2CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, max_grad_norm = 1, hidden_dims = HIDDEN_DIMS_DEFAULT):
        
        torch.manual_seed(20200808) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        super(A2CNetwork, self).__init__()
        
        self.max_grad_norm = max_grad_norm
        
        self.hlayers = dict()
        
        self.hlayers['shared'] = nn.ModuleList()
        self.hlayers['actor'] = nn.ModuleList()
        self.hlayers['critic'] = nn.ModuleList()
        
        # Input layer
        self.input_layer = nn.Linear( input_dim, hidden_dims['shared'][0] )
        
        # Hidden layers shared
        for i in range( len(hidden_dims['shared'] ) -1 ):
            self.hlayers['shared'].append( nn.Linear( hidden_dims['shared'][i], hidden_dims['shared'][i+1] ) )
        
        # Actor layers
        for i in range( len(hidden_dims['actor']) ):
            if i == 0:
                self.hlayers['actor'].append( nn.Linear( hidden_dims['shared'][-1], hidden_dims['actor'][i] ) )
            else:
                self.hlayers['actor'].append( nn.Linear( hidden_dims['actor'][i-1], hidden_dims['actor'][i] ) )
        self.actor_out_layer = nn.Linear( hidden_dims['actor'][-1], output_dim )
                
        #Critic layers
        for i in range( len(hidden_dims['critic']) ):
            if i == 0:
                self.hlayers['critic'].append( nn.Linear( hidden_dims['shared'][-1], hidden_dims['critic'][i] ) )
            else:
                self.hlayers['critic'].append( nn.Linear( hidden_dims['critic'][i-1], hidden_dims['critic'][i] ) )
        self.critic_out_layer = nn.Linear( hidden_dims['critic'][-1], 1 ) 
        
    # Prevents non Pytorch Tensor Object entering the processing stream
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x
    
    def forward(self, states):
        check_tensor = lambda x: isinstance(x, torch.Tensor)
        x_act = True 
        x_crit = True

        
        x = self._format(states)
        x = F.relu( self.input_layer(x) )
        for label in ['shared', 'actor', 'critic']:
            for self.hlayer in self.hlayers[label]:
                if label == 'shared':
                    x = F.relu( self.hlayer(x) )
                if label == 'actor':
                    x_act = F.relu( self.hlayer(x_act) )
                if label == 'critic':
                        x_crit = F.relu( self.hlayer(x_crit) )
                        
            if ( type(x_act) == bool ):
                x_act = x.clone()  # Create an Inplace copy...
            if ( type(x_crit) == bool ):
                x_crit = x.clone() # ...after processing shared layers

       
        return self.actor_out_layer(x_act), self.critic_out_layer(x_crit) 
    
    def fullpass(self, states):
        
        import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        _, value = self.forward(states)
        #dist = torch.distributions.Categorial( logits = logits )
        #dist = torch.distributions.categorical.Categorical( logits = logits ) #PyTorch 0.4.0
        #dist = torch.distributions.normal.Normal( logits, torch.std(logits) )
        #action = dist.sample()
        action = self.select_action(states)
        #logprob = dist.log_prob(action).unsqueeze(-1)
        #logprob = 0.1 # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #entropy = dist.entropy().unsqueeze(-1)
        #entropy = 0.1 # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #action = action.item() if len(action) == 1 else action.data.numpy()
        #action = F.hardtanh(logits) #FA: Limit to values between -1 and 1 
        #is_exploratory = action != np.argmax( logits.detach().numpy(), axis = int( len(state) != -1) )
        #is_exploratory = False # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        return action, value #, logprob, entropy, is_exploratory
    
    def select_action(self, states):
        logits, _ = self.forward(states)
        #dist = torch.distributions.Categorical(logits = logits)
        #action = dist.sample()
        action = F.hardtanh(logits) #FA: Limit to values between -1 and 1 
        #action = (action * 2) - 1 #FA: Give the normalized values a range between -1 and 1
        #action = action.item() if len(action) == 1 else action.data.numpy()
        return action
    
    def evaluate_state(self, states):
        _, value = self.forward(states)
        return value
    

               
                
        