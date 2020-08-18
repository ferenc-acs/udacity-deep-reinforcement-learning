#Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
#Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb

import torch
import numpy as np
import mya2cnet as mynet

class a2cagent():
    def __init__(self, numworkers, env, brain, max_steps = 10000):
        assert numworkers > 1
        assert env == 'unityagents.environment.UnityEnvironment'
        assert brain == 'unityagents.brain.BrainParameters'
        
        self.numworkers = numworkers
        self.env = env
        self.brain = brain
        self.max_steps = max_steps
        
        def train(states):
            self.a2c_net = mynet.A2CNetwork(self.brain.vector_observation_space_size, self.brain.vector_action_space_size)
            self.a2c_opt = torch.optim.Adam(self.a2c_net.parameters())
            
            #states = self.env.reset()
            states = self.env.vector_observations
            
            for step in count(start=1):
                states, is_terminals = self.interaction_step(self.env.vector_observations)
                
                if ( is_terminals.sum() > 0 ) or ( step - n_start >= self.max_steps ):
                    next_values = self.a2c_net.evaluate_state(state).detach().numpy() * ( 1 - ( is_terminals.sum() > 0 ) )
                    
                    self.rewards.append(next_values)
                    self.values.append( torch.Tensor(next_values) )
                    self.optimize_model()
                    
                    self.logpas = list()
                    self.entropies = list()
                    self.rewards = list()
                    self.values = list()
                    
                    n_start = step
        
        def optimize_model(self):
            T = len(self.rewards)
            discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint = False)
            returns = np.array( [[np.sum(discounts[:T-t] * rewards[t:, w]) for t in range(T)] for w in range(self.n_workers) ] )
            
            np.values = values.data.numpy()
            tau_discounts = np.logspace(0, T-1, num=T-1, base=self.gamma*self.tau, endpoint = False)
            advs = rewards[:-1] + self.gamma * mp_values[1:] - np_values[:-1]
            
            gaes = np.array(
                [[np.sum(tau_discounts[:T-1-t] * advs[t:, w]) for t in range(T-1)] for w in range(self.n_workers)])
            
            discount_gaes = discounts[:-1] * gaes
            
            #np_values = values.data.numpy()
            value_error = returns - values
            value_loss = value_error.pow(2).mul(0.5).mean()
            policy_loss = -(discount_gaes.detach() * logpas).mean()
            entropy_loss = -entropies.mean()
            
            loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss + self.entropy_loss_weight * entropy_loss
            
            self.a2c_opt.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_( self.a2c_net.parameters(), self.a2c_net.max_grad_norm )
            
            self.a2c_opt.step()
            
            
                    
                    
                    
        
    
# Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/v-14/95    
# A multi process class is needed to synchronize the acivities of all 20 workers.    
class MultiProcEnv(object):
    def __init__(self, make_env_fn, make_env_kargs, seed, numworkers):
        self.make_env_fn = make_env_fn
        self.make_env_kargs
        self.seed
        self.numworkers
        
        self.pipes = [mp.Pipe() for rank in range(self.numworkers)]
        
        self.workers = [mp.Process( target = self.work, 
                                   args=(rank, self.pipes(rank)[1])) for rank in range(self.numworkers)]
        
        [w.start() for w in self.workers]
        
    def work(self, rank, worker_end):
        env.self.make_env_fn( **self.make_env_kargs, seed = self.seed + rank)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send( env.reset(**kwargs) )
            if cmd == 'step':
                worker_end.send( env_step(**kwargs) )
            if cmd == '_past_limit':
                worker_end.send( env._elapsed_steps >= env._max_episode_steps )
                
            env.close( **kwargs ) 
            del env
            worker_end.close()
            break
            
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
    
    
        
                                                                      
                                           
        
        