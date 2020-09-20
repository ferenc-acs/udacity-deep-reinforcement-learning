import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pdb

# Format: IN_Num [Layer 1] (OUT_Num = IN_Num) [Layer 2] OUT_Num = ...
HIDDEN_DIMS_DEFAULT = {
    'shared' : (512, 512, 256, 256), #Three hidden layers
    'actor' : (256, 128, 128, 64), #Three hidden layers
    'critic' : (256, 128, 128, 64) #Three hidden layers
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
# Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb
# Thx2: https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
# Thx2: https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
# Thx2: https://discuss.pytorch.org/t/how-to-set-dtype-for-nn-layers/31274/2


class A2CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, max_grad_norm = 1, hidden_dims = HIDDEN_DIMS_DEFAULT):
        
        torch.manual_seed(20200808) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #torch.manual_seed(456454618181) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        super(A2CNetwork, self).__init__()
        
        self.to(DEVICE)
        
        self.max_grad_norm = max_grad_norm
        
        self.hlayers = dict()
        
        self.hlayers['shared'] = nn.ModuleList().to(DEVICE)
        self.hlayers['actor'] = nn.ModuleList().to(DEVICE)
        self.hlayers['critic'] = nn.ModuleList().to(DEVICE)
        
        # Input layer
        self.input_layer = nn.Linear( input_dim, hidden_dims['shared'][0] ).to(DEVICE)
        
        # Hidden layers shared
        for i in range( len(hidden_dims['shared'] ) -1 ):
            self.hlayers['shared'].append( nn.Linear( hidden_dims['shared'][i], hidden_dims['shared'][i+1] ).to(DEVICE) )
        
        # Actor layers
        for i in range( len(hidden_dims['actor']) ):
            if i == 0:
                self.hlayers['actor'].append( nn.Linear( hidden_dims['shared'][-1], hidden_dims['actor'][i] ).to(DEVICE) )
            else:
                self.hlayers['actor'].append( nn.Linear( hidden_dims['actor'][i-1], hidden_dims['actor'][i] ).to(DEVICE) )
        self.actor_out_layer = nn.Linear( hidden_dims['actor'][-1], output_dim ).to(DEVICE)
                
        #Critic layers
        for i in range( len(hidden_dims['critic']) ):
            if i == 0:
                self.hlayers['critic'].append( nn.Linear( hidden_dims['shared'][-1], hidden_dims['critic'][i] ).to(DEVICE) )
            else:
                self.hlayers['critic'].append( nn.Linear( hidden_dims['critic'][i-1], hidden_dims['critic'][i] ).to(DEVICE) )
        self.critic_out_layer = nn.Linear( hidden_dims['critic'][-1], 1 ).to(DEVICE) 
        
    # Prevents non Pytorch Tensor Object entering the processing stream
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, device=DEVICE)
            #x = torch.tensor(x, dtype=torch.double, device=DEVICE)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x
    
    def forward(self, states):
        #check_tensor = lambda x: isinstance(x, torch.Tensor)
        x_act = True 
        x_crit = True
        
        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        x = self._format(states)
        #x = torch.tensor(states, dtype=torch.float32).to(DEVICE) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        x = F.relu( self.input_layer(x) )
        for label in ['shared', 'actor', 'critic']:
            for hlayer in self.hlayers[label]:
                if label == 'shared':
                    x = F.relu( hlayer(x) )
                if label == 'actor':
                    x_act = F.relu( hlayer(x_act) )
                if label == 'critic':
                        x_crit = F.relu( hlayer(x_crit) )
                        
            if ( type(x_act) == bool ):
                x_act = x.clone()  # Create an Inplace copy...
            if ( type(x_crit) == bool ):
                x_crit = x.clone() # ...after processing shared layers

       
        return self.actor_out_layer(x_act), self.critic_out_layer(x_crit) 
    
    def fullpass(self, states):
        
        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        logits, value = self.forward(states)
        
        #dist = torch.distributions.Categorial( logits = logits )
        #dist = torch.distributions.categorical.Categorical( logits = logits ) #PyTorch 0.4.0
        dist = torch.distributions.normal.Normal( torch.mean(logits), torch.std(logits) )
        #action = dist.sample()
        #action = self.select_action(states)
        #action = action.item() if len(action) == 1 else action.data.numpy()
        action = F.hardtanh(logits) #FA: Limit to values between -1 and 1 
        logprob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        #is_exploratory = action != np.argmax( logits.detach().numpy(), axis = int( len(state) != -1) )
        return action, value, logprob, entropy #, , , is_exploratory
        #return action * 4, value, logprob, entropy # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
    
    def select_action(self, states):
        logits, _ = self.forward(states)
        
        #dist = torch.distributions.Categorical(logits = logits)
        #action = dist.sample()
        action = F.hardtanh(logits) #FA: Limit to values between -1 and 1 
        #action = (action * 2) - 1 #FA: Give the normalized values a range between -1 and 1
        #action = action.item() if len(action) == 1 else action.data.numpy()
        return action
        #return action * 4 # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
    
    def evaluate_state(self, states):
        _, value = self.forward(states)
        return value
    

               
                
        