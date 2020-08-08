import torch
import torch.nn as nn
import torch.nn.functional as F

# Thx2: https://emacs.stackexchange.com/a/13483
import imp # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
imp.reload(torch) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
torch.manual_seed(20200808) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!

# Format: IN_Num [Layer 1] (OUT_Num = IN_Num) [Layer 2] OUT_Num = ...
HIDDEN_DIMS_DEFAULT = {
    'shared' : (512, 512, 256, 256), #Three hidden layers
    'actor' : (256, 128, 128, 64), #Three hidden layers
    'critic' : (256, 128, 128, 64) #Three hidden layers
}

# thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb
class A2CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = HIDDEN_DIMS_DEFAULT):
        super(A2CNetwork, self).__init__()
        
        self.hlayers = dict()
        
        self.hlayers['shared'] = nn.ModuleList()
        self.hlayers['actor'] = nn.ModuleList()
        self.hlayers['critic'] = nn.ModuleList()
        self.hlayer = nn.Linear(1,1) #Temporary layer for iterations
        
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
    
    def forward(self, state):
        check_tensor = lambda x: isinstance(x, torch.Tensor)
        #last_x_shared_actor = True
        #last_x_shared_critic = True
        x_act = True 
        x_crit = True

        
        x = self._format(state)
        x = F.relu( self.input_layer(x) )
        for label in ['shared', 'actor', 'critic']:
            for self.hlayer in self.hlayers[label]:
                if label == 'shared':
                    x = F.relu( self.hlayer(x) )
                if label == 'actor':
                    #if check_tensor(last_x_shared_actor):
                    #    x_act = F.relu( self.hlayer(last_x_shared_actor) )
                    #    last_x_shared_actor = False
                    #else:
                    x_act = F.relu( self.hlayer(x_act) )
                if label == 'critic':
                    #if check_tensor(last_x_shared_critic):
                    #    x_crit = F.relu( self.hlayer(last_x_shared_critic) )
                    #    last_x_shared_critic = False
                    #else:
                        x_crit = F.relu( self.hlayer(x_crit) )
                        
            # Thx2: https://discuss.pytorch.org/t/copy-deepcopy-vs-clone/55022
            #if not check_tensor(last_x_shared_actor) and ( last_x_shared_actor == True ):
            #    last_x_shared_actor = x.clone()  # Create an Inplace copy...
            #if not check_tensor(last_x_shared_critic)  and ( last_x_shared_critic == True ):
            #    last_x_shared_critic = x.clone() # ...after processing shared layers
            if ( type(x_act) == bool ):
                x_act = x.clone()  # Create an Inplace copy...
            if ( type(x_crit) == bool ):
                x_crit = x.clone() # ...after processing shared layers

       
        return self.actor_out_layer(x_act), self.critic_out_layer(x_crit) 

               
                
        