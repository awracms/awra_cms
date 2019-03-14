from awrams.calibration.objectives import NSE,Bias,MonthlyResampler,FilteredMonthlyResampler, sampleCorrelPearson, makeNaN
import numpy as np

'''
This is a naive schema which just consists of list of names
Needed for logging setup and dataframe construction
+++ Need to formalise this...
'''

class LocalEval:

    schema = ['nse_daily','nse_monthly','bias']

    def __init__(self,obs,eval_period,min_valid=16, var1 = 'qtot_avg'):
        '''
        obs : timeseries of observations
        obj_period : the period over which we will evaluate

        assume that observations and modelled data are
        already constrained to eval_period
        '''

        self.valid_idx = {}
        self.nse = {}
        self.to_monthly = {}
        # Create filters for any missing data
        self.var1 = var1

        for k in [var1]:
            data = obs[k]

            if np.isnan(data).any():
                nan_mask = np.isnan(data)
                self.valid_idx[k] = np.where(nan_mask == False)
                self.to_monthly[k] = FilteredMonthlyResampler(eval_period,data,min_valid)
            else:
                self.to_monthly[k] = MonthlyResampler(eval_period)
                self.valid_idx[k] = slice(0,len(eval_period))

            # Build our caching evaluators
            self.bias[k] = Bias(data[self.valid_idx][k])
            self.nse_d[k] = NSE(data[self.valid_idx][k])
            self.nse_m[k] = NSE(self.to_monthly[k](data))

    def evaluate(self,modelled):
        qtot = modelled[self.var1][self.valid_idx][self.var1]
        qtot_m = self.to_monthly[self.var1](modelled[self.var1])
        return dict(nse_daily=self.nse_d((qtot)),nse_monthly=self.nse_m(qtot_m),bias=self.bias(qtot))

class TestGlobalSingle:
    def evaluate(self,l_results):
        return 1.0 - np.mean(l_results['qtot_nse'])

class TestLocalMulti:
    schema = ['qtot_viney', 'etot_correl_monthly', 'sm_correl_daily'] # for the logger
    def __init__(self,obs,eval_period,min_valid=16, var1='qtot', var2='etot', var3='w0'):
        self.valid_idx = {}
        self.to_monthly = {}
        self.nse_d = {}
        self.nse_m = {}
        self.bias = {}
        self.correl_m = {}
        self.correl = {}
        self.var1 = var1
        self.var2 = var2
        self.var3 = var3

        for k in [var1]:
            data = obs[k]
            if np.isnan(data).any():
                nan_mask = np.isnan(data)
                self.valid_idx[k] = np.where(nan_mask == False)
                self.to_monthly[k] = FilteredMonthlyResampler(eval_period,data,min_valid)
            else:
                self.to_monthly[k] = MonthlyResampler(eval_period)
                self.valid_idx[k] = slice(0,len(eval_period))

            # Build our caching evaluator
            if len(self.to_monthly[k](data))<=1:
                self.nse_m[k] = NSE(np.array(np.nan))
                self.nse_d[k] = NSE(np.array(np.nan))
                self.bias[k] = NSE(np.array(np.nan))
                
            else:
                self.nse_m[k] = NSE(self.to_monthly[k](data))
                self.bias[k] = Bias(data[self.valid_idx[k]])
                self.nse_d[k] = NSE(data[self.valid_idx[k]])

        for l in [var2]:
            data = obs[l]
            if np.isnan(data).any():
                self.to_monthly[l] = FilteredMonthlyResampler(eval_period,data,min_valid)
            else:
                self.to_monthly[l] = MonthlyResampler(eval_period)

            # Build our caching evaluators
            if len(self.to_monthly[l](data)) <=2:
                self.correl_m[l] = makeNaN(self.to_monthly[l](data))
            else:
                self.correl_m[l] = sampleCorrelPearson(self.to_monthly[l](data)) 
            

        for m in [var3]:
            data = obs[m]/100.
            if np.isnan(data).any():
                nan_mask = np.isnan(data)
                self.valid_idx[m] = np.where(nan_mask == False)
            else:
                self.valid_idx[m] = slice(0,len(eval_period))
            
            evaldata = data[self.valid_idx[m]]

            # Build our caching evaluators
            valid = len(evaldata) - len(evaldata[np.isnan(np.array(evaldata))])
            self.correl[m] = None
            if valid < 3: #threshold of valid values to calculate sample correlation
                self.correl[m] = makeNaN(data[self.valid_idx[m]])
            else:
                self.correl[m] = sampleCorrelPearson(data[self.valid_idx[m]])            
            
    def evaluate(self,modelled):
        qtot_nse_d = self.nse_d[self.var1](modelled[self.var1][self.valid_idx[self.var1]])
        qtot_nse_m = self.nse_m[self.var1](self.to_monthly[self.var1](modelled[self.var1]))
        qtot_bias = self.bias[self.var1](modelled[self.var1][self.valid_idx[self.var1]])
        qtot_viney_d=((qtot_nse_d + qtot_nse_m)/2 - 5.0*abs(np.log(1.0+qtot_bias))**2.5)

        sm_correl_d = self.correl['w0'](modelled['w0'][self.valid_idx['w0']])
        etot_correl_m = self.correl_m['etot'](self.to_monthly['etot'](modelled['etot']))

        return dict(qtot_viney=qtot_viney_d,etot_correl_monthly=etot_correl_m, sm_correl_daily=sm_correl_d)

def base_global_multi(l_results):
    #return 0.6*((l_results['qtot_nse_daily'] + l_results['qtot_nse_monthly'])/2.0 - 5.0*abs(np.log(1.0+l_results['qtot_bias']))**2.5) + 0.4*l_results['etot_nse_daily']
    return 0.7*l_results['qtot_viney'] + 0.15*l_results['etot_correl_monthly'] + 0.15*l_results['sm_correl_daily']

    
class GlobalMultiEval:
    def evaluate(self,l_results):
        # This method only conserves catchments where all the calibrated variates are present
        return 1.0 - np.mean(np.percentile([x for x in base_global_multi(l_results).values if not np.isnan(x)],[25,50,75,100]))
        
class AltGlobalMultiEval:
    def evaluate(self,l_results):
        # This method evaluates the fits to the various variables separately and conserves all catchments with data
        interm_result = 0.7 *np.mean(np.percentile([x for x in l_results['qtot_viney'].values if not np.isnan(x)],[25,50,75,100])) + 0.15*np.mean(np.percentile([x for x in l_results['etot_correl_monthly'].values if not np.isnan(x)],[25,50,75,100])) +  0.15*np.mean(np.percentile([x for x in l_results['sm_correl_daily'].values if not np.isnan(x)],[25,50,75,100]))
        return 1.0 - interm_result
