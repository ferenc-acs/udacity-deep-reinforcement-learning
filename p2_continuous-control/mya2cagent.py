#Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
#Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb

import torch
import numpy as np
from itertools import count
import mya2cnet as mynet

import pdb # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
import pprint as pp # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENV_IS_TRAIN_MODE = False #Set to 'False' only for debug purposes!

class a2cagent():
    def __init__(self, numworkers, env, brain, max_steps = 10000, policy_loss_weight = 1.0,
                 value_loss_weight = 0.6, entropy_loss_weight = 0.001, max_n_steps = 10):
        #assert numworkers > 1
        assert  'unityagents.environment.UnityEnvironment' in str( type(env) )
        assert  'unityagents.brain.BrainParameters' in str( type(brain) )
        
        self.numworkers = numworkers
        self.env = env
        self.brain = brain
        self.max_steps = max_steps
        self.max_n_steps = max_n_steps
        self.brain_inf = None
        
        #self.rewards = list(list())
        self.logpas = list()
        self.entropies = list()
        self.rewards = list()
        self.values = list()
        
        #self.actions = np.zeros( (max_steps, numworkers, self.brain.vector_action_space_size) ) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        self.running_reward = 0.0
        self.running_timestep = 0
        self.running_exploration = 0.0
        
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight        
        
    def train(self, gamma = 0.99, tau = 0.95):
        self.gamma = gamma
        self.tau = tau
        
        self.a2c_net = mynet.A2CNetwork(self.brain.vector_observation_space_size, \
                                        self.brain.vector_action_space_size)
        #self.a2c_net = self.a2c_net.float()
        self.a2c_net = self.a2c_net.double()
        self.a2c_net = self.a2c_net.to(DEVICE)
        self.a2c_opt = torch.optim.Adam(self.a2c_net.parameters())
        
        self.brain_inf = self.env.reset(train_mode=ENV_IS_TRAIN_MODE)[self.brain.brain_name] # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        # self.brain_inf = self.env.reset(train_mode=True)[self.brain.brain_name]

        states = self.brain_inf.vector_observations
        
        lastoptim = 0
        n_steps_start = 0

        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        for step in count(start=1):
            
            #if step == 33: # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
            #    pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
            states, is_terminals = self.interaction_step(states)
            
            if ( step - n_steps_start == self.max_n_steps ):
                lastoptim = step
                # Insert MORE CODE HERE!
                #next_values = self.a2c_net.evaluate_state(states).detach().cpu().numpy()
                #self.rewards.append(next_values)
                #self.values.append(torch.Tensor(next_values).double().to(DEVICE))
                
                self.optimize_model()
                
                self.logpas = []
                self.entropies = []
                self.rewards = []
                self.values = []
                n_steps_start = step
                
            if np.any(is_terminals):
                print(f' --> Environment reset at iteration: {step}')
                self.brain_inf = self.env.reset(train_mode=ENV_IS_TRAIN_MODE)[self.brain.brain_name]
                states = self.brain_inf.vector_observations
                self.logpas = []
                self.entropies = []
                self.rewards = []
                self.values = []
                n_steps_start = step
                
            
            print(f'\rTraining iteration: {step} ', f'last optimization: {lastoptim}'.rjust(30), end = (lambda x: '#' if x%2 == 0 else '+')(step) )
            
            
            
            #next_values = [self.a2c_net.evaluate_state(state).detach().numpy() for state in states] # * ( 1 - ( is_terminals.sum() > 0 ) )

            #self.rewards.append(next_values)
            #self.values.append( torch.Tensor(next_values) )

 
            if step >= self.max_steps:
                break
        #return(self.actions) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
    def optimize_model(self): 
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        logpas = torch.stack(self.logpas).squeeze().double().to(DEVICE)
        #logpas = torch.stack( tuple( torch.from_numpy( np.array(self.logpas) ) ) ).squeeze() #Because Pytorch 0.4.0 %-O
        entropies = torch.stack(self.entropies).squeeze().double().to(DEVICE)
        #entropies = torch.stack( tuple( torch.from_numpy( np.array(self.entropies) ) ) ).squeeze() #Because Pytorch 0.4.0 %-O
        values = torch.stack(self.values).squeeze().double().to(DEVICE)
        #values = torch.stack( tuple( [x[0] for x in self.values[0]] ) ).squeeze() #Because Pytorch 0.4.0 %-O
        
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint = False) #Calculate Monte Carlo like discounts
        #returns = np.array( [[np.sum( discounts[:T-t] * self.rewards[t:, w] ) for t in range(T)] for w in range(self.numworkers) ] )   
        
        if self.numworkers > 1:
            returns = np.array( [[np.sum( np.array(discounts)[:T-t] * np.array(self.rewards)[t:,w] )\
                                  for t in range(T)] for w in range(self.numworkers)] ).flatten() #Discounted returns
        else:
            returns = np.array( [np.sum( np.array(discounts)[:T-t] * np.array(self.rewards)[t:] ) for t in range(T)] ).flatten() #Discounted returns
                                 
                
        rewards = np.array(self.rewards).squeeze()
        np_values = values.data.cpu().numpy()
        tau_discounts = np.logspace(0, T-1, num=T-1, base=self.gamma*self.tau, endpoint = False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        
        if self.numworkers > 1:
            gaes = np.array( [ [np.sum(tau_discounts[:T-1-t] * advs[t:, w]) for t in range(T-1)] for w in range(self.numworkers) ] )
        else:
            gaes = np.array( [np.sum(tau_discounts[:T-1-t] * advs[t:] ) for t in range(T-1) ] )
            
        #gaes = np.array([np.sum(tau_discounts[:T-1-t] * advs[t]) for t in range(T-1)])

        #discount_gaes = discounts[:-1] * gaes
        #discount_gaes = torch.FloatTensor(discount_gaes.T).view(-1).unsqueeze(1)
        #discount_gaes = (lambda x: x.unsqueeze(1) if len (x) > 0 else x)(torch.FloatTensor(discounts[:-1].T).view(-1))
        #discount_gaes = torch.FloatTensor(discounts[:-1].T).view(-1).unsqueeze(1).to(DEVICE)
        discount_gaes = torch.DoubleTensor(discounts[:-1].T).view(-1).unsqueeze(1).double().to(DEVICE)
        
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        #np_values = values.data.numpy()
        value_error = returns - values.detach().cpu().numpy().flatten() #Because Pytorch 0.4.0 %-O
        value_loss = np.mean( np.multiply( np.power(value_error, 2), 0.5 ) )
        
        #policy_loss = -1 * torch.mean( discount_gaes.detach() * torch.mean(logpas, 2)[-T:-1])
        if self.numworkers > 1:
            policy_loss = -1 * torch.mean( discount_gaes.detach() * torch.mean(logpas, 2)[-T:-1] )
        else:
            policy_loss = -1 * torch.mean( discount_gaes.detach() * torch.mean(logpas) )
            
        entropy_loss = -1 * np.mean( entropies.detach().cpu().numpy() ) 
        entropy_loss = -1 * entropies.mean()

        loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss + self.entropy_loss_weight * entropy_loss

        self.a2c_opt.zero_grad()
        
        #loss.backward(retain_graph=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_( self.a2c_net.parameters(), self.a2c_net.max_grad_norm )

        self.a2c_opt.step()
        
    def interaction_step(self, states):  
        # thx2: https://github.com/udacity/deep-reinforcement-learning/blob/master/python/unityagents/brain.py
        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        #actionsL = list()
        #is_exploratoriesL = list()
        #valuesL = list()
        #logpassesL = list()
        #entropiesL = list()
        
        #FA; BMSoT: OBSOLETE
        #for state in states:
        #actions, is_exploratories, logpasses, entropies, values = self.a2c_net.fullpass(states)
        actions, values, logpasses, entropies = self.a2c_net.fullpass(states)
        #print(' --> Actions:'); print( 'mean:', actions.detach().mean().cpu().numpy(), 'sdev:', actions.detach().std().cpu().numpy() ) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #self.actions =
        #actionsL.append(actions)
        #is_exploratoriesL.append(is_exploratories)
        #valuesL.append(values)
        #logpassesL.append(logpasses)
        #entropiesL.append(entropies)

        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        #FA; BMSoT: OBSOLETE
        # Make a Tensor out of the list of value tensors
        # Thx2: https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868
        #values = torch.cat(values_l)
        
        
        # Thx2: https://stackoverflow.com/a/6383390/12171415
        #try:
        self.logpas.append(logpasses)
        #except AttributeError:
        #    self.logpas = torch.stack( torch.Tensor(logpas) )

        #try:
        self.entropies.append(entropies)
        #except AttributeError:
        #    self.entropies = torch.stack( torch.Tensor(entropies) )
        
        #new_states = env.step( [x.cpu().detach().numpy() for x in actions] ) = env.step( [x.cpu().detach().numpy() for x in actions] )
        #self.brain_inf = self.env.step( [x.cpu().detach().numpy() for x in actions] )[self.brain.brain_name]
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #actions = np.random.randn(self.numworkers, self.brain.vector_action_space_size) # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #self.brain_inf = self.env.step( actions )[self.brain.brain_name] # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        self.brain_inf = self.env.step( actions.cpu().detach().numpy() )[self.brain.brain_name]
        #self.brain_inf = self.env.step( actions.cpu().detach().numpy() )['ReacherBrain'] # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        new_states = self.brain_inf.vector_observations
        is_terminals = self.brain_inf.local_done
        rewards = np.array(self.brain_inf.rewards).reshape(-1, 1)
        
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        self.rewards.append(rewards)
        self.values.append(values)
        
        self.running_reward += np.mean(rewards)
        self.running_timestep += 1
        #self.running_exploration += is_exploratory[:,np.newaxis].astype(np.int)
        #self.running_exploration = False # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!

        return new_states, is_terminals
            
            
                    
                    
                    
