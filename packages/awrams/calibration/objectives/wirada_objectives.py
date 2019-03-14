from awrams.calibration.objectives import NSE,Bias,MonthlyResampler,FilteredMonthlyResampler
from awrams.calibration.support import input_group
#from DataProcessing.functions import boxcox
import numpy as np
import pandas as pd

'''
This is a naive schema which just consists of list of names
Needed for logging setup and dataframe construction
+++ Need to formalise this...
'''

class LocalEval:

    input_schema = input_group(['qtot'])
    output_schema = ['nse_daily','nse_monthly','bias']

    def __init__(self,obs,eval_period,min_valid=15,flow_variable='qtot'):
        '''
        obs : timeseries of observations
        obj_period : the period over which we will evaluate

        assume that observations and modelled data are
        already constrained to eval_period
        '''
        self.flow_variable = flow_variable
        
        # Create filters for any missing data
        qtot = obs[flow_variable]

        if np.isnan(qtot).any():
            nan_mask = np.isnan(qtot)
            self.valid_idx = np.where(nan_mask == False)
            self.to_monthly = FilteredMonthlyResampler(eval_period,qtot,min_valid)
        else:
            self.to_monthly = MonthlyResampler(eval_period)
            self.valid_idx = slice(0,len(eval_period))


        # Build the fast monthly resampler
        #self.to_monthly = MonthlyResampler(eval_period)

        # Build our caching evaluators

        self.bias = Bias(qtot[self.valid_idx])
        self.nse_d = NSE(qtot[self.valid_idx])
        self.nse_m = NSE(self.to_monthly(qtot))

    def evaluate(self,modelled):
        qtot = modelled[self.flow_variable][self.valid_idx]
        qtot_m = self.to_monthly(modelled[self.flow_variable])
        #return dict(nse_daily=self.nse_d((qtot)),nse_monthly=self.nse_m(qtot_m),bias=self.bias(qtot))
        return np.array((self.nse_d((qtot)),self.nse_m(qtot_m),self.bias(qtot)))

def base_global(l_results):
    return (l_results['nse_daily'] + l_results['nse_monthly'])/2.0 - 5.0*abs(np.log(1.0+l_results['bias']))**2.5


class GlobalMultiEval:
    output_schema = ['objf_val']
    objective_key = 'objf_val'

    def evaluate(self,l_results):
        #return dict(objf_val = 1.0 - np.mean(np.percentile(base_global(l_results),[25,50,75,100])))
        #return 1.0 - np.mean(base_global(l_results))
        #Below is original WIRADA implementation which takes mean of quartiles for catchments
        #def calculate(self,l_results):
        return dict(objf_val = 1.0 - np.mean(base_global(l_results)))
