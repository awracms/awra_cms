from awrams.utils.ts import processing
import numpy as np

class Bias:
	'''
	Precomputed relative bias evaluator
	'''
	def __init__(self,obs):
		self.obs_sum = np.sum(obs)

	def __call__(self,modelled):
		mod_t = np.sum(modelled)
		return (mod_t - self.obs_sum) / self.obs_sum

class NSE:
	'''
	Precomputed NSE evaluator
	'''
	def __init__(self,obs):
		self.obs = obs
		self.obs_moment = np.sum((np.mean(self.obs)-self.obs)**2)

	def __call__(self,modelled):
		err_moment = np.sum((modelled-self.obs)**2)
		return 1.0 - err_moment/self.obs_moment
        
class CorrelPearson:
	'''
	For population
	'''
	def __init__(self,obs):
		self.obs = obs
		self.obs_normalised = self.obs.mean()
		self.obs_standardised = self.obs.std()

	def __call__(self,modelled):
		modelled_normalised = modelled.mean()
		modelled_standardised = modelled.std()
		return ((self.obs*modelled).mean()-self.obs_normalised*modelled_normalised)/(self.obs_standardised*modelled_standardised)
        
class sampleCorrelPearson:
	'''
	For sample
	'''
	def __init__(self,obs):
		self.obs = obs
		self.obs_normalised = self.obs.mean()
		self.obs_standardised = self.obs.std(ddof=1)
		self.n = len(self.obs)
	def __call__(self,modelled):
		modelled_normalised = modelled.mean()
		modelled_standardised = modelled.std(ddof=1)
		if self.n <=2:
			return (np.nan)
		else:
			return (np.sum(self.obs*modelled)-self.n*self.obs_normalised*modelled_normalised)/((self.n-1)*self.obs_standardised*modelled_standardised)
        
class MonthlyResampler:
	'''
	Precomputed daily->monthly resampler
	'''
	def __init__(self,eval_period):
		self.m_idx = processing.build_resample_index(eval_period,'m')

	def __call__(self,modelled):
		return processing.resample_with_index_monthly(modelled,self.m_idx)

class FilteredMonthlyResampler:
	'''
	Precomputed daily->monthly resampler with nan filter
	'''
	def __init__(self,eval_period,obs,min_valid):
		self.s_idx,self.s_mask = processing.build_masksample_indices(eval_period,'m',np.isnan(obs),min_valid)

	def __call__(self,modelled):
		return processing.resample_with_mask_index_monthly(modelled,self.s_idx,self.s_mask)

class makeNaN:
	'''
	For sample
	'''
	def __init__(self,obs):
		self.obs = obs
	def __call__(self,modelled):
		return np.nan