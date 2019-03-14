from numpy import concatenate as concat
import numpy as np
from collections import OrderedDict

class ParameterDesc:
    def __init__(self,pspace,idx,min,max):
        self.pspace = pspace
        self.idx = idx
        self['min'] = min
        self['max'] = max
        
    def __setitem__(self,k,v):
        target = getattr(self.pspace,k)
        target[self.idx] = v
        
    def __getitem__(self,k):
        return getattr(self.pspace,k)[self.idx]
        
    def __repr__(self):
        return 'min: %s, max: %s' % (self['min'],self['max'])
        
class ParameterSpace:
    def __init__(self):
        self.min = np.empty(shape=(0,))
        self.max = np.empty(shape=(0,))
        self.range = np.empty(shape=(0,))
    
        self.params = OrderedDict()
        
    def __setitem__(self,k,v):
        nparams = len(self.params)+1
        self.min.resize((nparams,))
        self.max.resize((nparams,))
        self.params[k] = ParameterDesc(self,nparams-1,v[0],v[1])
        
        self.range = self.max - self.min

    def __len__(self):
        return len(self.params)


class ParameterPoint:
    def __init__(self,score,parameters,meta=None):
        self.score = score
        self.params = parameters
        self.meta = meta

    def __repr__(self):
        return "Score: %s, Params: %s, Meta: %s" % (self.score,self.params,self.meta)

class ParameterPopulation:
    def __init__(self,score,parameters,meta=None):
        self.score = score
        self.params = parameters
        if meta is None:
            meta = np.array([None for i in range(len(score))])
        self.meta = meta
        
    def sort(self,ascending=True,inplace = True):
        ssorted = sorted(zip(self.score,self.params,self.meta), \
            key=lambda x: x[0],reverse=not ascending)
        score = np.array([k[0] for k in ssorted])
        params = concat([p[1] for p in ssorted]).reshape(self.params.shape)
        meta = np.array([k[2] for k in ssorted])
        
        if inplace:
            self.score = score
            self.params = params
            self.meta = meta
            return self
        else:       
            return ParameterPopulation(score,params,meta)

    def shuffle(self,rng=None,inplace=True):
        if rng is None:
            rng = np.random

        sidx = np.arange(len(self))
        rng.shuffle(sidx)

        score = self.score[sidx]
        params = self.params[sidx]
        meta = self.meta[sidx]

        if inplace:
            self.score = score
            self.params = params
            self.meta = meta
            return self
        else:       
            return ParameterPopulation(score,params,meta)

    @classmethod
    def from_points(self,points):
        score = np.array([p.score for p in points])
        params = np.array([p.params for p in points])
        meta = np.array([p.meta for p in points])
        return ParameterPopulation(score,params,meta)

    @classmethod
    def join_populations(self,populations):
        score = concat([p.score for p in populations])
        params = concat([p.params for p in populations])
        meta = concat([p.meta for p in populations])
        return ParameterPopulation(score,params,meta)


    def __len__(self):
        return len(self.score)
    
    def __iter__(self):
        for i in range(len(self.score)):
            yield self[i]
    
    def __getitem__(self,k):
        if isinstance(k,int):
            return ParameterPoint(self.score[k],self.params[k],self.meta[k])
        return ParameterPopulation(self.score[k],self.params[k],self.meta[k])
    
    def __setitem__(self,k,v):
        self.score[k] = v.score
        self.params[k] = v.params
        self.meta[k] = v.meta
        
def concatenate_populations(param_pops):
    score = concat([p.score for p in param_pops])
    params = concat([p.params for p in param_pops])
    meta = concat([p.meta for p in param_pops])
    return ParameterPopulation(score,params,meta)