#Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
#Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb

import torch
import numpy as np
from itertools import count
import mya2cnet as mynet
from os.path import join
from utilities import get_time_string

import pdb # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
import pprint as pp # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENV_IS_TRAIN_MODE = True #Set to 'False' only for debug purposes!
VERSION = '0.01a' #Version information for saving and loading agent files

class a2cagent():
    def __init__(self, numworkers, env, brain, max_steps = 10000, policy_loss_weight = 1.0, load_agent=False,
                 value_loss_weight = 0.6, entropy_loss_weight = 0.001, max_n_steps = 10, hidden_dims = False):
        
        assert  'unityagents.environment.UnityEnvironment' in str( type(env) )
        assert  'unityagents.brain.BrainParameters' in str( type(brain) )

        self.numworkers = numworkers
        self.env = env
        self.brain = brain

        
        #no filename was given so build everything from scratch
        if load_agent == False: 
            self.max_steps = max_steps
            self.max_n_steps = max_n_steps
            self.brain_inf = None

            self.logpas = list()
            self.entropies = list()
            self.rewards = list()
            self.values = list()

            self.running_reward = 0.0
            self.running_timestep = 0
            self.running_exploration = 0.0

            self.policy_loss_weight = policy_loss_weight
            self.value_loss_weight = value_loss_weight
            self.entropy_loss_weight = entropy_loss_weight    

            self.hidden_dims = hidden_dims

            
        
    def train(self, gamma = 0.99, tau = 0.95):
        self.gamma = gamma
        self.tau = tau
        
        if not self.hidden_dims == False:
            self.a2c_net = mynet.A2CNetwork(self.brain.vector_observation_space_size, \
                                            self.brain.vector_action_space_size)
        else:
            self.a2c_net = mynet.A2CNetwork(self.brain.vector_observation_space_size, \
                                            self.brain.vector_action_space_size, hidden_dims = self.hidden_dims)
            

        self.a2c_net = self.a2c_net.double()
        self.a2c_net = self.a2c_net.to(DEVICE)
        
        #lr according to Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning"
        #self.a2c_opt = torch.optim.Adam(self.a2c_net.parameters(), lr=0.00005)
        self.a2c_opt = torch.optim.Adam(self.a2c_net.parameters())
        
        self.brain_inf = self.env.reset(train_mode=ENV_IS_TRAIN_MODE)[self.brain.brain_name]

        states = self.brain_inf.vector_observations
        
        lastoptim = 0
        n_steps_start = 0
        
        for step in count(start=1):
            
            states, is_terminals = self.interaction_step(states)
            
            if ( step - n_steps_start == self.max_n_steps ):
                lastoptim = step
                
                self.optimize_model()
                
                self.logpas = []
                self.entropies = []
                self.rewards = []
                self.values = []
                n_steps_start = step
                
            if np.any(is_terminals):
                print(f' --> Environment reset at iteration: {step}')
                self.brain_inf = self.env.reset(train_mode=ENV_IS_TRAIN_MODE)[self.brain.brain_name]
                self.logpas = []
                self.entropies = []
                self.rewards = []
                self.values = []
                n_steps_start = step
                
            
            print(f'\rTraining iteration: {step} ', f'last optimization: {lastoptim}'.rjust(30), end = (lambda x: '#' if x%2 == 0 else '+')(step) )
            
 
            if step >= self.max_steps:
                break

        
    def optimize_model(self): 
        #pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        logpas = torch.stack(self.logpas).squeeze().double().to(DEVICE)
        entropies = torch.stack(self.entropies).squeeze().double().to(DEVICE)
        values = torch.stack(self.values).squeeze().double().to(DEVICE)
        
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint = False) #Calculate Monte Carlo like discounts
        
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
            
        discount_gaes = torch.DoubleTensor(discounts[:-1].T).view(-1).unsqueeze(1).double().to(DEVICE)

        value_error = returns - values.detach().cpu().numpy().flatten() #Because Pytorch 0.4.0 %-O
        value_loss = np.mean( np.multiply( np.power(value_error, 2), 0.5 ) )
        
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

        actions, values, logpasses, entropies = self.a2c_net.fullpass(states)
        self.logpas.append(logpasses)
        self.entropies.append(entropies)
        self.brain_inf = self.env.step( actions.cpu().detach().numpy() )[self.brain.brain_name]

        new_states = self.brain_inf.vector_observations
        is_terminals = self.brain_inf.local_done
        rewards = np.array(self.brain_inf.rewards).reshape(-1, 1)
        
        self.rewards.append(rewards)
        self.values.append(values)
        
        self.running_reward += np.mean(rewards)
        self.running_timestep += 1

        return new_states, is_terminals

    def save_agent(self, path):
        state = {
        'version' : VERSION,
        'logpas' : self.logpas,
        'entropies' : self.entropies,
        'rewards' : self.rewards,
        'values' : self.values,
        
        'running_reward' : self.running_reward,
        'running_timestep' : self.running_timestep,
        'running_exploration' : self.running_exploration,
        
        'policy_loss_weight' : self.policy_loss_weight,
        'value_loss_weight' : self.value_loss_weight,
        'entropy_loss_weight' : self.entropy_loss_weight,
        
        'hidden_dims' : self.hidden_dims,
        'gamma' : self.gamma,
        'tau' : self.tau,
        
        'a2c_net' : self.a2c_net.state_dict(),
        'a2c_opt' : self.a2c_opt.state_dict(),
        'brain_inf' : self.brain_inf
        }
        
        filename = f'drlnd-reacher-{VERSION}-{get_time_string()}.pth'
        torch.save( state, join(path, filename) )
        return filename
    
    
    def load_agent(self, path, filename):
        # thx2: https://github.com/pytorch/pytorch/issues/10622#issuecomment-474733769
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        checkpoint = torch.load(join(path, filename), map_location=map_location)
        
        if checkpoint['version'] == VERSION:
            self.a2c_net = mynet.A2CNetwork(self.brain.vector_observation_space_size, \
                                            self.brain.vector_action_space_size)
            self.a2c_net.load_state_dict(checkpoint['a2c_net'])
            
            self.a2c_opt = torch.optim.Adam(self.a2c_net.parameters())
            self.a2c_opt.load_state_dict(checkpoint['a2c_opt'])
            self.a2c_net.eval()
        else:
            print(f'Error loading file {filename}: Wrong version!')
            print(f"{filename} was generated with version {checkpoint['version']} but {VERSION} required!")
                   
    
                    
                    
                    
