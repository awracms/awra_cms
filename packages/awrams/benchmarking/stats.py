# import scipy.stats as stats
import numpy as np
import pandas as pd

from .utils import valid_only

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('stats')


def nse(observed,modeled):
    '''
    Return Nash-Suttcliffe efficiency of observed/modelled series
    '''

    obs_mean = np.mean(observed)
    n = sum((observed - modeled)**2.0)
    d = sum((observed - obs_mean)**2.0)

    try:
        return 1 - (n/d)
    except ZeroDivisionError:
        return None


class StatsPair:
    def __init__(self,observed,predicted,drop_nan=True,o_name='Observed',p_name='Predicted',v_name=''):
        if drop_nan:
            self.observed = valid_only(observed)
            self.predicted = valid_only(predicted)
        else:
            self.observed = observed
            self.predicted = predicted

        if len(self.observed) != len(self.predicted):
            raise Exception("Cannot compare series of differing lengths")

        self.o_name = o_name
        self.p_name = p_name
        self.v_name = v_name

        self.p_mean = np.ma.mean(self.predicted)
        self.o_mean = np.ma.mean(self.observed)

        self.n = len(self.observed)

    def regress(self):
        return Regression(self.observed,self.predicted)

    def bias(self,relative=True,factor=100.,signed= True):
        bias = self.p_mean - self.o_mean
        if signed==False:
            bias = abs(bias)
        return bias if not relative else bias / self.o_mean * factor

    def nse(self):
        return nse(self.observed,self.predicted)

    def rmse(self):
        error = self.predicted - self.observed
        return np.sqrt(np.mean(error**2.))

    def pearsons_r(self):
        # return stats.pearsonr(self.observed,self.predicted)[0]
        return pearsonr(self.observed,self.predicted)

    def kge(self):
        return kge(self.observed,self.predicted)

    def fobj(self):
        nse = self.nse()
        if nse is None:
            return None
        bias = self.bias(factor=1.)
        return nse - 5*(np.abs(np.log(1+bias)))**2.5


class Regression:
    def __init__(self,observed,predicted,drop_nan=True):
        if drop_nan:
            self.observed = valid_only(observed)
            self.predicted = valid_only(predicted)
        else:
            self.observed = observed
            self.predicted = predicted

        if len(self.observed) != len(self.predicted):
            raise Exception("Cannot regress series of differing lengths")

        self.n = len(self.observed)

        # self.slope,self.intercept,self.r_value,self.p_value,self.std_err = stats.linregress(observed,predicted)
        self.slope,self.intercept,self.r_value,self.std_err = linregress(observed,predicted)
        self.r_squared = self.r_value ** 2.

    def best_fit(self):
        return (self.observed * self.slope) + self.intercept

    def rmse(self):
        error = self.best_fit() - self.observed
        return np.sqrt(np.mean(error ** 2.))


def standard_percentiles(series, pctiles=None):
    if pctiles is None:
        pctiles = [0,5,25,50,75,95,100]
    return pd.Series(index=[str(pc)+'%' for pc in pctiles],data=np.percentile(valid_only(series),pctiles))

def build_stats_df(ref_dct,mod_dct,comparison_sites,freq=None, aggr_how=None):
    '''
    Return dataframe of per-site comparison statistics for the reference and model supplied
    '''

    stats_idx = ['nse','bias','bias_relative','bias_absrel','rmse','pearsons_r','r_slope','r_intercept','kge','kge_alpha','kge_beta','mean','obs_mean','fobj']

    stats_df = pd.DataFrame(index=stats_idx)

    _obs = None # will be array to hold all data for calculating stats on all
    _mod = None
    for site in comparison_sites:
        try:
            obs = ref_dct[site] # observations for site
            mod = mod_dct[site] # results for model
        except KeyError: # for monthly and annual stats site may have been rejected
            continue
        valid_isect_idx = valid_only(obs).index.intersection(valid_only(mod).index)
        if len(valid_isect_idx) == 0:
            continue

        obs = obs.loc[valid_isect_idx]
        mod = mod.loc[valid_isect_idx]

        # accumulate for 'all' stats
        if _obs is None:
            _obs = obs.as_matrix().ravel()
            _mod = mod.as_matrix().ravel()
        else:
            _obs = np.r_[_obs, obs.as_matrix().ravel()]
            _mod = np.r_[_mod, mod.as_matrix().ravel()]

        s = StatsPair(obs,mod)
        try:
            r = s.regress()
        except ValueError: # data for site does not exist
            continue

        s_kge, _, kge_alpha, kge_beta = s.kge()

        site_stats = [s.nse(), s.bias(False), s.bias(True,factor=1.), s.bias(True,factor=1.,signed=False), s.rmse(),
                      s.pearsons_r(), r.slope, r.intercept, s_kge, kge_alpha, kge_beta, mod.mean(), obs.mean(), np.nan]
        stats_df[site] = site_stats

    if _obs is not None and _mod is not None:
        s = StatsPair(_obs,_mod)
        try:
            r = s.regress()
            s_kge, _, kge_alpha, kge_beta = s.kge()
            site_stats = [s.nse(),s.bias(False),s.bias(True,factor=1.),s.bias(True,signed=False), s.rmse(),s.pearsons_r(),r.slope,r.intercept, s_kge, kge_alpha, kge_beta,_mod.mean(),_obs.mean(),np.nan]
            stats_df['all'] = site_stats
        except ValueError: # data fro site does not exist
            stats_df['all'] = None

    return stats_df

def pearsonr(x, y):
    ### taken from scipy.stats.stats
    """
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)

    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    # r_den = np.sqrt(ss(xm) * ss(ym))
    r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym*ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    df = n-2
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        #prob = betai(0.5*df, 0.5, df / (df + t_squared))
    return r #, prob

def linregress(x, y=None):
    ### taken from scipy.stats.stats
    """
    Calculate a regression line

    This computes a least-squares regression for two sets of measurements.

    Parameters
    ----------
    x, y : array_like
        two sets of measurements.  Both arrays should have the same length.
        If only x is given (and y=None), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension.

    Returns
    -------
    slope : float
        slope of the regression line
    intercept : float
        intercept of the regression line
    r-value : float
        correlation coefficient
    p-value : float
        two-sided p-value for a hypothesis test whose null hypothesis is
        that the slope is zero.
    stderr : float
        Standard error of the estimate


    Examples
    --------
    >>> from scipy import stats
    >>> import numpy as np
    >>> x = np.random.random(10)
    >>> y = np.random.random(10)
    >>> slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    # To get coefficient of determination (r_squared)

    >>> print "r-squared:", r_value**2
    r-squared: 0.15286643777

    """
    TINY = 1.0e-20
    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            msg = "If only `x` is given as input, it has to be of shape (2, N) \
            or (N, 2), provided shape was %s" % str(x.shape)
            raise ValueError(msg)
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    n = len(x)
    xmean = np.mean(x,None)
    ymean = np.mean(y,None)

    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm*ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if (r > 1.0):
            r = 1.0
        elif (r < -1.0):
            r = -1.0

    df = n-2
    t = r*np.sqrt(df/((1.0-r+TINY)*(1.0+r+TINY)))
    #prob = distributions.t.sf(np.abs(t),df)*2
    slope = r_num / ssxm
    intercept = ymean - slope*xmean
    sterrest = np.sqrt((1-r*r)*ssym / ssxm / df)
    # return slope, intercept, r, prob, sterrest
    return slope, intercept, r, sterrest

def kge(x, y, return_all=True):
    """
    Kling-Gupta Efficiency
    Corresponding paper: 
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling
    output:
        kge: Kling-Gupta Efficiency
    optional_output:
        cc: correlation 
        alpha: ratio of the standard deviation
        beta: ratio of the mean

    Code based on spotpy
    https://github.com/thouska/spotpy

    The MIT License (MIT)

    Copyright (c) 2015 Tobias Houska

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    if len(x) == len(y):
        cc = np.corrcoef(x, y)[0, 1]
        alpha = np.std(x) / np.std(y)
        beta = np.sum(x) / np.sum(y)
        kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        raise ValueError("x,y input lengths mismatch")