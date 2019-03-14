from collections import OrderedDict
from awrams.utils.nodegraph.nodes import callable_to_funcspec,funcspec_to_callable
import numpy as np
import h5py
import pandas as pd

from awrams.utils.parameters import h5_to_dataframe

class ChildException(Exception):
    '''
    Should be used in any multiprocessing parent loop where a child process has thrown an exception
    The child should pass its own (original) exception through a messaging queue, and the parent process should report
    the exception, then raise ChildException into its own loop, and clean up
    '''
    def __init__(self):
        Exception.__init__(self)

def aggregate(in_data,weights,total_weight=None):
    out = (in_data * weights).sum(axis=-1)
    if total_weight is not None:
        out = out/total_weight
    return out

def get_params_from_mapping(mapping): # +++ Add default PSpace to models?
    from .parameters import ParameterSpace

    pspace = ParameterSpace()

    pdict = OrderedDict(sorted([(k,p.args) for k,p in mapping.items() \
        if p.node_type == 'parameter' and p.args['fixed'] == False]))
    for k,v in pdict.items():
        pspace[k] = v['min_val'],v['max_val']
    return pspace

def flat_params_to_dict(params,pspace):
    out_p = {}
    for i, k in enumerate(pspace.params):
        out_p[k] = params[i]
    return out_p

def flattened_areas(extent): #+++ Move to extents?
    return np.array(extent.areas[~extent.mask]).flatten()

def input_group(names,agg_method='mean'):
    '''
    Convenience function for building InputDef dict where all variables have same agg_method
    '''
    return OrderedDict([(n,InputDef(n,agg_method)) for n in names])

class InputDef:
    '''
    Define an (aggregated) mapping for input_var
    agg_method can (currently) be 'mean','volume',or 'flat'
    '''
    def __init__(self,input_var,agg_method='mean'):
        self.input_var = input_var
        self.agg_method = agg_method

class OptimizerSpec:
    def __init__(self,opt_cls,**opt_args):
        self.opt_cls = opt_cls
        self.opt_args = opt_args
        if hasattr(opt_cls,'meta'):
            self.opt_meta = opt_cls.meta
        else:
            self.opt_meta = None

class ObjectiveFunctionSpec:
    '''
    Function Spec for local objective function (should be generalised for local and global..)
    '''
    def __init__(self,objf_class,init_args=None):
        self.funcspec = callable_to_funcspec(objf_class)
        if init_args is None: init_args = {}
        self.init_args = init_args
        self.inputs_from_model = objf_class.input_schema
        self.output_schema = objf_class.output_schema


class ObjectiveSpec:
    '''
    Container for local/global objectives and observations +++ Refactor?
    '''
    def __init__(self,global_objf,catchment_objf,observations,eval_period):
        #+++ Possibly require separate obs for catchment, local...
        self._global_objf = callable_to_funcspec(global_objf)
        self.catchment_objf = catchment_objf
        self.observations = observations #+++ Not all objectives require obs...
        self.eval_period = eval_period

    @property
    def global_objf(self):
        return funcspec_to_callable(self._global_objf)

class EvolverSpec:
    def __init__(self,evolver_cls,evolver_init_args=None,evolver_run_args=None):
        self.evolver_cls = evolver_cls
        if evolver_init_args is None:
            evolver_init_args = {}
        if evolver_run_args is None:
            evolver_run_args = {}
        self.evolver_init_args = evolver_init_args
        self.evolver_run_args = evolver_run_args

# Raijin/PBS specific helpers

def prerun_raijin(cal_spec):
    import os
    jobfs = os.environ['PBS_JOBFS']
    original_log_path,logfn = os.path.split(cal_spec['logfile'])
    cal_spec['original_logfile'] = cal_spec['logfile']
    cal_spec['logfile'] = os.path.join(jobfs,logfn)
    return cal_spec
    
def postrun_raijin(cal_spec,server_locals):
    from shutil import copyfile
    copyfile(cal_spec['logfile'],cal_spec['original_logfile'])

class CalibrationResults:
    def __init__(self,filename):
        self.ds = h5py.File(filename,'r')
        self.parameter_names = list(self.ds['parameter_name'][...])
        self.catchment_ids = list(self.ds['local_scores/local_id'])
        
    @property
    def global_keys(self):
        return list(self.ds['global_scores'])
    
    @property
    def local_keys(self):
        return [k for k in self.ds['local_scores'] if k != 'local_id']

    def get_initial_parameters(self):
        '''
        Return a Series of all fixed parameter values used in this calibration run
        '''
        return h5_to_dataframe(self.ds['initial_parameters'])
    
    def get_parameter_values(self):
        '''
        Return a DataFrame of all parameter values over all iterations
        '''
        return pd.DataFrame(self.ds['parameter_values'][...],columns=self.parameter_names)
    
    def get_local_scores(self,local_key):
        '''
        Get all local scores for the specified output key
        '''
        return pd.DataFrame(self.ds['local_scores'][local_key][...],columns=self.catchment_ids)
    
    def get_global_scores(self):
        '''
        Get all local scores for the specified output key
        '''
        df = pd.DataFrame()
        for g in self.ds['global_scores']:
            df[g] = self.ds['global_scores'][g][...]
        return df
    
    def best_param_index(self,global_key=None,reverse=False):
        '''
        Find the iteration with the minimum score (or max if reverse is True)
        '''
        if global_key is None:
            global_key = self.global_keys[0]
        search_func = np.argmax if reverse else np.argmin 
        return search_func(self.ds['global_scores'][global_key])
    
    def get_best_paramset(self,global_key=None,reverse=False,include_fixed=True):
        '''
        Get the parameter set with the best (minimum) value in <global_key>
        '''
        bpi = self.best_param_index(global_key,reverse)
        best_pset = pd.Series(self.ds['parameter_values'][bpi],index=self.parameter_names)

        if include_fixed:
            out_df = self.get_initial_parameters()
            out_df.value.update(best_pset)
            
            best_pset = out_df

        return best_pset
