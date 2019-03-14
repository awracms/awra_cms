from awrams.calibration.objectives import *
from awrams.calibration.support import input_group
import numpy as np

obs = np.linspace(0.,1.,365)

def test_nse():
	nse = NSE(obs)
	assert( nse(obs) == 1.0 )
	assert( nse(np.repeat(obs.mean(),365)) == 0.0 )

def test_bias():
	bias = Bias(obs)
	assert( bias(obs) == 0.0 )
	assert( bias(obs*2.0) == 1.0 )

class LocalQTotal:
    '''
    Simple sum of run
    '''

    #input_schema = ['qtot','etot','dd']
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
        #return 1.0 - np.mean((l_results['qtot_nse'] + l_results['etot_nse']) * 0.5)
        return dict(objf_val = objf_val, qtot_nse=qtot_nse, etot_nse=etot_nse)

class LocalQtotRouting:

    input_schema = input_group(['qtot'],'flat')
    output_schema = ['qtot_nse']

    def __init__(self,obs,eval_period,routing_spec,min_valid=15):

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

        run_period = routing_spec['run_period']
        self.eval_index = np.s_[(eval_period[0]-run_period[0]).days:(eval_period[-1]-run_period[0]).days + 1]
        self.gauge_idx = routing_spec['gauge_idx']

        from awrams.models.awralrouting.routing_support import RoutingRunner
        self.routing_runner = RoutingRunner(**routing_spec)

    def evaluate(self,modelled):
        out_loss = self.routing_runner.do_routing(modelled['qtot'],modelled['params']['k_rout'])
        qtot = out_loss[self.eval_index,self.gauge_idx]
        qtot_nse = self.nse[self.flow_variable](qtot[self.valid_idx[self.flow_variable]])
        return np.array(qtot_nse)
