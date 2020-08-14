#Thx2: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11/
#Thx2: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb

class a2cagent():
    def __init__(self, numworkers):
        assert numworkers > 1
        
        self.numworkers = numworkers
        
    
    
    
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
        
                                                                      
                                           
        
        