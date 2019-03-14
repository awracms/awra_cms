'''
Estimators used are based on the following paper:

Andrea Saltelli, Paola Annoni, Ivano Azzini, Francesca Campolongo, Marco Ratto, 
and Stefano Tarantola. 
Variance based sensitivity analysis of model output. Design and estimator for 
the total sensitivity index. 
Computer Physics Communications, 181 (2010) 259-270

'''

import numpy as np
#from sobol import i4_sobol as sobol
from SALib.sample import sobol_sequence
from awrams.calibration.optimizer import Optimizer
from awrams.utils.messaging import Message
from collections import OrderedDict
import h5py
import pandas as pd


class SobolOptimizer(Optimizer):
    '''
    Sobol sampler for AWRAMS calibration system.
    '''

    meta = [('sample_space',str),('ss_iter',int)]

    def __init__(self,pspace,eval_fac,threshold,max_eval,converge_n=5,block_size=5):
        super().__init__()
        self.pspace = pspace
        self.eval_fac = eval_fac
        self.threshold = threshold
        self.max_eval = max_eval
        self.converge_n = converge_n
        self.block_size = block_size
        
    def run(self):
        self.evaluator = self.eval_fac.new()

        self.cur_delta = np.array((np.inf))
        self.n_eval = 0
        
        ndim = len(self.pspace)
        
        seed = 1
            
        fA = np.empty((0,))
        fB = np.empty((0,))
        #fAb = [np.empty((0,)) for i in range(ndim)]
        fAb = np.empty((ndim,0,))# for i in range(ndim)]
        
        prev_sens = [1e-10 for i in range(self.converge_n+1)]
        
        N = 0

        convergence = None

        cur_sens = None

        #+++ Have to generate whole population in advance due to limitations of SALib sobol_sequence,
        # or alternatively regenerate previous sequence every block..

        max_n = int(np.ceil(self.max_eval/(ndim+2))) + self.block_size

        spop = generate_sobol_population(max_n,2*ndim)
        
        while convergence is None:
            self.status = Message('running',n_eval = self.n_eval, cur_sens=cur_sens)
            self.check_status()
            A,B,Ab = build_sampling_matrices(self.block_size,self.pspace,spop,N)

            meta_A = [('A',N+n) for n in range(self.block_size)]
            meta_B = [('B',N+n) for n in range(self.block_size)]

            
   
            fA = np.concatenate([fA,self.evaluator.evaluate_population(A,meta_A)])
            fB = np.concatenate([fB,self.evaluator.evaluate_population(B,meta_B)])

            self.n_eval += 2 * self.block_size
            
            fAb.resize((ndim,fAb.shape[1]+len(Ab[0])))
 
            for i in range(ndim):
                meta_Ab = [('Ab%s'%i,N+n) for n in range(self.block_size)]
                fAb[i,-len(Ab[0]):] = self.evaluator.evaluate_population(Ab[i],meta_Ab)
                self.n_eval += len(Ab[i])

            N+= self.block_size
            
            for i in np.arange(self.block_size)[::-1]:
                cur_sens = sum([estimate_sens_total(fA[:N-i],fB[:N-i],fAb[:,:N-i],x,N-i) for x in range(ndim)])           
                prev_sens.pop()
                prev_sens.insert(0,cur_sens)
            self.cur_delta = np.abs((cur_sens-np.array(prev_sens[1:]))/np.array(prev_sens[1:])) 

            convergence = self.check_converged()
        
        self.status = Message('complete',n_eval = self.n_eval, cur_sens=cur_sens, condition=convergence)
        self.update_status()

        return cur_sens
            
            
    def check_converged(self):
        if (self.cur_delta < self.threshold).all():
            return 'Convergence threshold reached'
        if self.n_eval >= self.max_eval:
            return 'Maximum evaluations completed'

def build_sampling_matrices(N,pspace,spop,skip=0):
    d = len(pspace)
    #spop,seed = generate_sobol_population(N,2*d,seed)
    A = spop[skip:skip+N,:d] * pspace.range + pspace.min
    B = spop[skip:skip+N,d:] * pspace.range + pspace.min
    Ab = []
    for i in range(d):
        new = A.copy()
        new[:,i] = B[:,i]
        Ab.append(new)
    return A,B,Ab#,seed

def _generate_sobol_population(pop_size,ndim,seed=1):
    spop = np.array([sobol(ndim,x)[0] for x in range(seed,pop_size+seed)])
    return spop,pop_size+seed

def generate_sobol_population(pop_size,ndim):
    spop = sobol_sequence.sample(pop_size,ndim)
    return spop

def estimate_sens(fA,fB,fAb,i,N):
    '''
    Estimate the sensitivity of parameter i
    '''
    sens = (fB *(fAb[i] - fA)).sum()
    return 1/N * sens

def estimate_sens_total(fA,fB,fAb,i,N):
    '''
    Estimate the total sensitivity of parameter i
    '''
    sens = ((fA - fAb[i]) ** 2.0).sum()
    return 1/(2*N) * sens

def estimate_sens_pair(fAb,i,j,N):
    '''
    Estimate the paired sensitivity of parameters i and j
    '''
    sens = ((fAb[i] - fAb[j]) ** 2.0).sum()
    return (1/(2*N)) * sens

class SensitivityResults:
    '''
    Wrap the results of a SobolOptimizer calibration run
    '''
    def __init__(self,filename):
        self.ds = h5py.File(filename,'r')
        self.sample_space = self.ds['task_meta']['sample_space'][...]
        self.ndim = len(self.ds['parameter_name'])
        self.parameters = self.ds['parameter_name'][...]
        self.N = self.ds['task_meta']['ss_iter'][...].max() + 1
        self.catchment_ids = list(self.ds['local_scores/local_id'])
        
    @property
    def global_keys(self):
        return list(self.ds['global_scores'])
    
    def _get_si(self,scores,total=False):
        scores_std = (scores - scores.mean())/scores.std()
        fA,fB,fAb = get_sens_matrices(self.sample_space,scores_std,self.ndim)
        
        psens = OrderedDict()
        for i,p in enumerate(self.parameters):
            if total:
                sens = estimate_sens_total(fA,fB,fAb,i,self.N)
            else:
                sens = estimate_sens(fA,fB,fAb,i,self.N)
            psens[p] = sens
        
        return pd.Series(psens)       
    
    def get_global_si(self,k,total=False):
        '''
        Get the sensitivity index for global variable k
        total :: Return Si (False) or STi (True)
        '''
        scores = self.ds['global_scores'][k][...]
        return self._get_si(scores,total)
    
    def get_catchment_si(self,k,catch_id,total=False):
        '''
        Get the sensitivity index for variable k of catchment catch_id
        '''
        cidx = self.catchment_ids.index(catch_id)
        scores = self.ds['local_scores'][k][:,cidx]
        return self._get_si(scores,total)
    
    def get_all_catchment_si(self,k,total=False):
        '''
        Return a DataFrame of Si (or STi) for variable k, over all catchments
        '''
        df = pd.DataFrame(columns=self.parameters,index=self.catchment_ids)
        for cid in self.catchment_ids:
            df.loc[cid] = self.get_catchment_si(k,cid,total)
        return df

def get_sens_matrices(sample_space,scores,ndim):
    fa_idx = sample_space == 'A'
    fb_idx = sample_space == 'B'
    fab_idx = []
    for i in range(ndim):
        fab_idx.append(sample_space == 'Ab%s' % i)
    fA = scores[fa_idx]
    fB = scores[fb_idx]
    fAb = []
    for abidx in fab_idx:
        fAb.append(scores[abidx])
    
    return fA,fB,fAb
    