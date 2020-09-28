import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pdb # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
# Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb
# Thx2: https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
# Thx2: https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
# Thx2: https://discuss.pytorch.org/t/how-to-set-dtype-for-nn-layers/31274/2


class A2CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, max_grad_norm = 1):
        
        torch.manual_seed(20200808) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #torch.manual_seed(456454618181) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        super(A2CNetwork, self).__init__()
        
        self.to(DEVICE)
        
        self.max_grad_norm = max_grad_norm
        
        
        # Input layer
        self.input_layer = nn.Linear( input_dim, 512 ).double().to(DEVICE)
        
        # Hidden layers shared
        self.hlayer1 = nn.Linear( 512, 256 ).double().to(DEVICE)
        self.hlayer2 = nn.Linear( 256, 256 ).double().to(DEVICE)

        # Actor layers
        self.actolayer1 = nn.Linear( 256, 128 ).double().to(DEVICE)
        self.actolayer2 = nn.Linear( 128, 64 ).double().to(DEVICE)
        self.actor_out_layer = nn.Linear( 64, output_dim ).double().to(DEVICE)
        
            
        #Critic layers
        self.critlayer1 = nn.Linear( 256, 128 ).double().to(DEVICE)
        self.critlayer2 = nn.Linear( 128, 64 ).double().to(DEVICE)
        self.critic_out_layer = nn.Linear( 64, 1 ).double().to(DEVICE) 
    
        
    # Prevents non Pytorch Tensor Object entering the processing stream
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            #x = torch.tensor(x, dtype=torch.float, device=DEVICE)
            x = torch.tensor(x, dtype=torch.double, device=DEVICE)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x
    
    # Scales the absolute values of actions, like glasses for eyes
    def _scale_action_np(self, action):
        act_enh = F.hardtanh(torch.mul(action, 7))
        return act_enh
    
    def forward(self, states):
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        x = self._format(states)
        
        # Normalize Input
        x = F.normalize(x)
        #x = torch.tensor(states, dtype=torch.float32).to(DEVICE) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        x = F.tanh( self.input_layer(x) )
        
        x = F.relu( self.hlayer1(x) )
        #x = F.tanh( self.hlayer1(x) )
        
        x = F.relu( self.hlayer2(x) )
        #x = F.tanh( self.hlayer2(x) )
        
        x_act = x.clone()  # Create an Inplace copy...
        x_crit = x.clone() # ...after processing shared layers
        
        x_act = F.relu( self.actolayer1(x_act) )
        #x_act = F.tanh( self.actolayer1(x_act) )
        
        x_act = F.relu( self.actolayer2(x_act) )
        #x_act = F.tanh( self.actolayer2(x_act) )
        
        x_crit = F.relu( self.critlayer1(x_crit) )
        #x_crit = F.tanh( self.critlayer1(x_crit) )
        
        x_crit = F.relu( self.critlayer2(x_crit) )
        #x_crit = F.tanh( self.critlayer2(x_crit) )
        
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
        #action = F.hardtanh(logits) #FA: Limit to values between -1 and 1 
        action = self._scale_action_np(logits)
        logprob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        #is_exploratory = action != np.argmax( logits.detach().numpy(), axis = int( len(state) != -1) )
        return action, value, logprob, entropy #, , , is_exploratory
        #return action * 4, value, logprob, entropy # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
    
    def select_action(self, states):
        logits, _ = self.forward(states)
        
        #dist = torch.distributions.Categorical(logits = logits)
        #action = dist.sample()
        #action = F.hardtanh(logits) #FA: Limit to values between -1 and 1 
        #action = (action * 2) - 1 #FA: Give the normalized values a range between -1 and 1
        #action = action.item() if len(action) == 1 else action.data.numpy()
        return self._scale_action_np(logits).detach().cpu().numpy()
        #return action * 4 # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
    
    def evaluate_state(self, states):
        _, value = self.forward(states)
        return value
    

               
                
        