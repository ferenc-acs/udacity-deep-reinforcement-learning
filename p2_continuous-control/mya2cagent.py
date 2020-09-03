#Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
#Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb

import torch
import numpy as np
from itertools import count
import mya2cnet as mynet

import pdb # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!


class a2cagent():
    def __init__(self, numworkers, env, brain, max_steps = 10000, policy_loss_weight = 1.0, value_loss_weight = 0.6, entropy_loss_weight = 0.001):
        assert numworkers > 1
        assert  'unityagents.environment.UnityEnvironment' in str( type(env) )
        assert  'unityagents.brain.BrainParameters' in str( type(brain) )
        
        self.numworkers = numworkers
        self.env = env
        self.brain = brain
        self.max_steps = max_steps
        self.brain_inf = None
        
        self.logpas = list()
        self.entropies = list()
        
        self.rewards = list(list())
        self.values = list()
        
        self.running_reward = 0.0
        self.running_timestep = 0
        self.running_exploration = 0.0
        
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight        
        
    def train(self, states, gamma = 0.99, tau = 0.95):
        self.gamma = gamma
        self.tau = tau
        
        self.a2c_net = mynet.A2CNetwork(self.brain.vector_observation_space_size, self.brain.vector_action_space_size)
        self.a2c_opt = torch.optim.Adam(self.a2c_net.parameters())
        
        self.brain_inf = self.env.reset(train_mode=True)[self.brain.brain_name]

        #states = self.env.reset()
        #states = self.env.vector_observations

        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        for step in count(start=1):
            
            if step > 10: # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
                pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
            
        #for step in range(1): # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!       
            print(f'\rTraining epoch: {step} ', end = (lambda x: '#' if x%2 == 0 else '+')(step) )
            states, is_terminals = self.interaction_step(states, self.env)
            
            #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!

            # BEGIN Not necessary: The Unity environment takes care of syncronizing 
            #if ( is_terminals.sum() > 0 ) or ( step - n_start >= self.max_steps ):
            #if ( np.sum(is_terminals) > 0 ): # or ( step - n_start >= self.max_steps ):
            # END Not necessary: The Unity environment takes care of syncronizing
            #next_values = [self.a2c_net.evaluate_state(state).detach().numpy() for state in states] # * ( 1 - ( is_terminals.sum() > 0 ) )

            #self.rewards.append(next_values)
            #self.values.append( torch.Tensor(next_values) )
            if step > 10: #Gather some data first, otherwise GAE estimation gets funny!
                #if step == 3: # Debug! Debug! Debug! Debug! Debug! Debug! 
                #    import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! 
                self.optimize_model()

            #self.logpas = list()
            #self.entropies = list()
            #self.rewards = list()
            #self.values = list()

            n_start = step
            if step >= self.max_steps:
                break
        
    def optimize_model(self): 
        logpas = torch.stack(self.logpas).squeeze()
        #logpas = torch.stack( tuple( torch.from_numpy( np.array(self.logpas) ) ) ).squeeze() #Because Pytorch 0.4.0 %-O
        #entropies = torch.stack(self.entropies).squeeze()
        #entropies = torch.stack( tuple( torch.from_numpy( np.array(self.entropies) ) ) ).squeeze() #Because Pytorch 0.4.0 %-O
        values = torch.stack(self.values).squeeze()  
        #values = torch.stack( tuple( [x[0] for x in self.values[0]] ) ).squeeze() #Because Pytorch 0.4.0 %-O
        
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint = False) #Calculate Monte Carlo like discounts
        #returns = np.array( [[np.sum( discounts[:T-t] * self.rewards[t:, w] ) for t in range(T)] for w in range(self.numworkers) ] )   
        
        returns = np.array( [[np.sum( np.array(discounts)[:T-t] * np.array(self.rewards)[t:,w] ) \
                              for t in range(T)] for w in range(self.numworkers)] ).flatten() #Discounted returns
        rewards = np.array(self.rewards).squeeze()
        np_values = values.data.numpy()
        tau_discounts = np.logspace(0, T-1, num=T-1, base=self.gamma*self.tau, endpoint = False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]

        gaes = np.array([[np.sum(tau_discounts[:T-1-t] * advs[t:, w]) for t in range(T-1)] for w in range(self.numworkers)])
        #gaes = np.array([np.sum(tau_discounts[:T-1-t] * advs[t]) for t in range(T-1)])

        #discount_gaes = discounts[:-1] * gaes
        #discount_gaes = torch.FloatTensor(discount_gaes.T).view(-1).unsqueeze(1)
        #discount_gaes = (lambda x: x.unsqueeze(1) if len (x) > 0 else x)(torch.FloatTensor(discounts[:-1].T).view(-1))
        discount_gaes = torch.FloatTensor(discounts[:-1].T).view(-1).unsqueeze(1)
        
        #np_values = values.data.numpy()
        value_error = returns - values.detach().numpy().flatten() #Because Pytorch 0.4.0 %-O
        value_loss = np.mean( np.multiply( np.power(value_error, 2), 0.5 ) )
        
        policy_loss = -1 * torch.mean( discount_gaes.detach() * torch.mean(logpas, 2)[-T:-1])
            
        #entropy_loss = -1 * np.mean(entropies.numpy()) #Not good! We need a Tensor!
        #entropy_loss = -1 * entropies.mean()

        #loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss + self.entropy_loss_weight * entropy_loss
        loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss 

        self.a2c_opt.zero_grad()
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_( self.a2c_net.parameters(), self.a2c_net.max_grad_norm )

        self.a2c_opt.step()
        
    def interaction_step(self, states, env):  
        
        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        #actionsL = list()
        #is_exploratoriesL = list()
        #valuesL = list()
        #logpassesL = list()
        #entropiesL = list()
        
        #FA; BMSoT: OBSOLETE
        #for state in states:
        #actions, is_exploratories, logpasses, entropies, values = self.a2c_net.fullpass(states)
        actions, values, logpasses = self.a2c_net.fullpass(states)
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
        #self.entropies.append(entropiesL)
        #except AttributeError:
        #    self.entropies = torch.stack( torch.Tensor(entropies) )
        
        #new_states = env.step( [x.cpu().detach().numpy() for x in actions] ) = env.step( [x.cpu().detach().numpy() for x in actions] )
        self.brain_inf = env.step( [x.cpu().detach().numpy() for x in actions] )[self.brain.brain_name]
        new_states = self.brain_inf.vector_observations
        is_terminals = self.brain_inf.local_done
        rewards = np.array(self.brain_inf.rewards).reshape(-1, 1)
        
        #import pdb; pdb.set_trace() # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        
        self.rewards.append(rewards)
        self.values.append(values)
        
        self.running_reward += np.mean(rewards)
        self.running_timestep += 1
        #self.running_exploration += is_exploratory[:,np.newaxis].astype(np.int)
        #self.running_exploration = False # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!

        return new_states, is_terminals
            
            
                    
                    
                    
        
# FA: Actually the code block below is not necessary, because the Unity Environment already contains 20 robot arms... \
# FA: ...and takes care of synchronizing them. Keeping in the source, nevertheless, until the big cleanup!

# Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/v-14/95    
# A multi process class is needed to synchronize the acivities of all 20 workers.    

import torch.multiprocessing as mp

class MultiProcEnv(object):
    #def __init__(self, make_env_fn, make_env_kargs, seed, numworkers):
    def __init__(self, env, seed, numworkers):
        
        assert numworkers > 1
        assert  'unityagents.environment.UnityEnvironment' in str( type(env) )        
        
        #self.make_env_fn = make_env_fn
        #self.make_env_kargs
        self.seed = seed
        self.numworkers = numworkers
        
        self.pipes = [mp.Pipe() for rank in range(self.numworkers)]
        
        myargs = [(rank, self.pipes[rank][1]) for rank in range(self.numworkers)]
        self.workers = [mp.Process( target = self.work, args=myargs )]
        
        [w.start() for w in self.workers]
        
    #def work(self, rank, worker_end):
    def work(*args):    # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #for a in args:  # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
        #    print(a)    # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
    
        #env.self.make_env_fn( **self.make_env_kargs, seed = self.seed + rank)
        #while True:
        #    cmd, kwargs = worker_end.recv()
        #    if cmd == 'reset':
        #        worker_end.send( env.reset(**kwargs) )
        #    if cmd == 'step':
        #        worker_end.send( env_step(**kwargs) )
        #    if cmd == '_past_limit':
        #        worker_end.send( env._elapsed_steps >= env._max_episode_steps )
                
        #    env.close( **kwargs ) 
        #    del env
        #    worker_end.close()
        #    break
        pass # Debug! Debug! Debug! Debug! Debug! Debug! Debug! Debug!
            
    def step(self, actions):
        assert len(actions) == self.n_workers
        
        [self.send_msg( ('step', {'action':actions[rank]}), rank ) \
         for rank in range(self.n_workers)]
        
        results = []
        
        for rank in range(self.n_workers):
            parent, end, _ = self.pipes(rank)
            o, r, d, _ = parent_end.recv()
            if d:
                self.send_msg( ('reset', {}), rank )
                o = paren_end.recv()
                
            results.append( (o, np.array(r, dtype=np.float), np.array(d, dtype=np.float), _) )
            
        return [np.vstack(block) for block in np.array(results).T]
    
    
        
                                                                      
                                           
        
        