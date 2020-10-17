import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pdb # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Format: IN_Num [Layer 1] (OUT_Num = IN_Num) [Layer 2] OUT_Num = ...
HIDDEN_DIMS_DEFAULT = {
    'shared' : (512, 256, 256), #Two hidden layers
    'actor' : (256, 128, 64), #Two actor layers
    'critic' : (256, 128, 64) #Two crtic layers
} 

# Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
# Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb
# Thx2: https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
# Thx2: https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
# Thx2: https://discuss.pytorch.org/t/how-to-set-dtype-for-nn-layers/31274/2
# Thx2: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/tree/master/a2c_ppo_acktr

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class A2CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, max_grad_norm = 1, hidden_dims = False):
        
        #torch.manual_seed(20200808) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        if not hidden_dims == False:
            self.hidden_dims = hidden_dims # When hidden dims are passed use them...
        else:
            self.hidden_dims = HIDDEN_DIMS_DEFAULT # Otherwise use default
        
        super(A2CNetwork, self).__init__()
        
        self.to(DEVICE)
        
        self.max_grad_norm = max_grad_norm
        
        if self.hidden_dims == HIDDEN_DIMS_DEFAULT: #If default values not modified go into this for debugging
        
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

        else:
            self.hlayers = dict()
        
            self.hlayers['shared'] = nn.ModuleList().double().to(DEVICE)
            self.hlayers['actor'] = nn.ModuleList().double().to(DEVICE)
            self.hlayers['critic'] = nn.ModuleList().double().to(DEVICE)

            # Input layer
            self.input_layer = nn.Linear( input_dim, hidden_dims['shared'][0] ).double().to(DEVICE)

            # Hidden layers shared
            for i in range( len(hidden_dims['shared'] ) -1 ):
                self.hlayers['shared'].append( nn.Linear( hidden_dims['shared'][i], hidden_dims['shared'][i+1] ).double().to(DEVICE) )

            # Actor layers
            for i in range( len(hidden_dims['actor']) ):
                if i == 0:
                    self.hlayers['actor'].append( nn.Linear( hidden_dims['shared'][-1], hidden_dims['actor'][i] ).double().to(DEVICE) )
                else:
                    self.hlayers['actor'].append( nn.Linear( hidden_dims['actor'][i-1], hidden_dims['actor'][i] ).double().to(DEVICE) )
            self.actor_out_layer = nn.Linear( hidden_dims['actor'][-1], output_dim ).double().to(DEVICE)

            #Critic layers
            for i in range( len(hidden_dims['critic']) ):
                if i == 0:
                    self.hlayers['critic'].append( nn.Linear( hidden_dims['shared'][-1], hidden_dims['critic'][i] ).double().to(DEVICE) )
                else:
                    self.hlayers['critic'].append( nn.Linear( hidden_dims['critic'][i-1], hidden_dims['critic'][i] ).double().to(DEVICE) )
            self.critic_out_layer = nn.Linear( hidden_dims['critic'][-1], 1 ).double().to(DEVICE) 
            
            
        #Mean Layer for Probability calculation
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        
        self.fc_mean = init_(nn.Linear(input_dim, output_dim).double().to(DEVICE))
        self.logstd = AddBias(torch.zeros(output_dim).double().to(DEVICE))
    
        
    # Prevents non Pytorch Tensor Object entering the processing stream
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double, device=DEVICE)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x
    
    # Scales the absolute values of actions, like glasses for eyes
    def _scale_action_np(self, action):
        act_enh = F.hardtanh(torch.mul(action, 1))
        return act_enh
    
    def forward(self, states):
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        x_act = True #Init
        x_crit = True #Init
        
        x = self._format(states)
        
        # Normalize Input
        x = F.normalize(x)
        
        if self.hidden_dims == HIDDEN_DIMS_DEFAULT: #If default values not modified go into this for debugging
            
            x = F.hardtanh( self.input_layer(x) )

            x = F.relu( self.hlayer1(x) )
            #x = F.hardtanh( self.hlayer1(x) )

            x = F.relu( self.hlayer2(x) )
            #x = F.hardtanh( self.hlayer2(x) )

            x_act = x.clone()  # Create an Inplace copy...
            x_crit = x.clone() # ...after processing shared layers

            #x_act = F.relu( self.actolayer1(x_act) )
            #x_act = F.hardtanh( self.actolayer1(x_act) )
            x_act = F.tanh( self.actolayer1(x_act) )

            #x_act = F.relu( self.actolayer2(x_act) )
            #x_act = F.hardtanh( self.actolayer2(x_act) )
            x_act = F.tanh( self.actolayer2(x_act) )

            #x_crit = F.relu( self.critlayer1(x_crit) )
            #x_crit = F.hardtanh( self.critlayer1(x_crit) )
            x_crit = F.tanh( self.critlayer1(x_crit) )

            #x_crit = F.relu( self.critlayer2(x_crit) )
            #x_crit = F.hardtanh( self.critlayer2(x_crit) )
            x_crit = F.tanh( self.critlayer2(x_crit) )
            
            x_act = self.actor_out_layer(x_act)
            x_crit = self.critic_out_layer(x_crit)
            
        else:
            True
            x = F.tanh( self.input_layer(x) )
        
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

            x_act = F.tanh( self.actor_out_layer(x_act) )
            x_crit = F.tanh( self.critic_out_layer(x_crit) )            
            
        
        return x_act, x_crit
    
    
    def fullpass(self, states):
        
        
        logits, value = self.forward(states)
        
        action_mean = self.fc_mean(self._format(states))
        zeros = torch.zeros(action_mean.size()).double().to(DEVICE)
        action_logstd = self.logstd(zeros)
        dist = torch.distributions.normal.Normal( action_mean, action_logstd.exp() )

        
        action = self._scale_action_np(logits)
        logprob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return action, value, logprob, entropy
        
    
    def select_action(self, states):
        logits, _ = self.forward(states)
        
        return self._scale_action_np(logits).detach().cpu().numpy()
    
    
    def evaluate_state(self, states):
        _, value = self.forward(states)
        return value
    

               
                
        