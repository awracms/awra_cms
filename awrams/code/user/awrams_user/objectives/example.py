from awrams.calibration.objectives import *
from awrams.calibration.support import input_group
import numpy as np

class LocalQTotal:
    '''
    Simple sum of run
    '''

    input_schema = input_group(['qtot','etot','dd'],'volume')
    output_schema = ['qtot_vol','etot_vol','dd_vol']

    def __init__(self,obs,eval_period):
        pass

    def evaluate(self,modelled):
        return np.array((np.sum(modelled['qtot']),np.sum(modelled['etot']),np.sum(modelled['dd'])))

class GlobalQTotal:

    output_schema = ['qtot_vol','etot_vol','dd_vol']
    objective_key = 'qtot_vol'

    def evaluate(self,l_results):
        out_d = dict( [(k, np.sum(l_results[k])) for k in self.output_schema] )
        return out_d

class TestLocalSingle:

    input_schema = input_group(['qtot'])
    output_schema = ['qtot_nse']

    def __init__(self,obs,eval_period,min_valid=15):

        self.valid_idx = {}
        self.nse = {}
        self.flow_variable = 'qtot'
        for k in [self.flow_variable]:

            data = obs[k]

            if np.isnan(data).any():
                nan_mask = np.isnan(data)
                self.valid_idx[k] = np.where(nan_mask == False)
            else:
                self.valid_idx[k] = slice(0,len(eval_period))

            self.nse[k] = NSE(data[self.valid_idx[k]])

    def evaluate(self,modelled):
        qtot_nse = self.nse[self.flow_variable](modelled[self.flow_variable][self.valid_idx[self.flow_variable]])
        return np.array(qtot_nse)

class TestGlobalSingle:

    output_schema = ['objf_val']
    objective_key = 'objf_val'

    def evaluate(self,l_results):
        objf_val = 1.0 - np.mean(l_results['qtot_nse'])
        return dict(objf_val = objf_val)

class TestLocalMulti:

    input_schema = input_group(['qtot','etot'])
    output_schema = ['qtot_nse','etot_nse']

    def __init__(self,obs,eval_period,min_valid=15,flow_variable='qtot_avg',et_variable='etot_avg'):

        self.valid_idx = {}
        self.nse = {}

        self.flow_variable = flow_variable
        self.et_variable = et_variable
        
        for k in [flow_variable,et_variable]:

            data = obs[k]

            if np.isnan(data).any():
                nan_mask = np.isnan(data)
                self.valid_idx[k] = np.where(nan_mask == False)
            else:
                self.valid_idx[k] = slice(0,len(eval_period))

            self.nse[k] = NSE(data[self.valid_idx[k]])

    def evaluate(self,modelled):
        qtot_nse = self.nse[self.flow_variable](modelled[self.flow_variable][self.valid_idx[self.flow_variable]])
        etot_nse = self.nse[self.et_variable](modelled[self.et_variable][self.valid_idx[self.et_variable]])
        return dict(qtot_nse=qtot_nse,etot_nse=etot_nse)

class TestGlobalMultiEval:

    output_schema = ['objf_val','qtot_nse','etot_nse']
    objective_key = 'objf_val'

    def evaluate(self,l_results):
        qtot_nse = np.mean(l_results['qtot_nse'])
        etot_nse = np.mean(l_results['etot_nse'])
        objf_val = 1.0 - (qtot_nse+etot_nse) * 0.5
        return dict(objf_val = objf_val, qtot_nse=qtot_nse, etot_nse=etot_nse)
