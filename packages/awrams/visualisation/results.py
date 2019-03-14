"""
Notebook/IPython interface to model results, including saving results summary to disk

"""
from collections import Iterable
from glob import glob
import json
import os
import re

import numpy as np
import pandas as pd

from awrams.utils import extents, frequency
import awrams.utils.datetools as dt
from awrams.utils.metatypes import ObjectDict, ObjectContainer, DataDict
from awrams.utils.io.data_mapping import SplitFileManager, managed_dataset
from awrams.utils.io.netcdf_wrapper import start_date, end_date, \
    dataset_frequency
from .vis import timeseries, spatial, get_range, _sanitize_inputs, \
    _sanitize_period
from .support import VariableInstance,VariableGroup


def isiterable(obj):
    return isinstance(obj,Iterable)

def list_results(path='./'):
    '''
    list the folders in 'path' which contain AWRAMS results
    '''
    result_dirs = []
    for folder in os.listdir(path):
        cur_sub = os.path.join(path,folder)
        if os.path.isdir(cur_sub):
            if len(_identify_results_files(cur_sub))>0:
                result_dirs.append(cur_sub)
    return result_dirs

class ResultsEncoder(json.JSONEncoder):
    '''
    JSON Encoder for AWRA Results - make sure our types are serializable
    '''
    def default(self,obj): #pylint: disable=method-hidden
        if isinstance(obj,ObjectDict):
            return obj.__dict__
        elif isinstance(obj,np.core.ndarray):
            #naively dumping to list, no metadata regarding
            #turning back into array
            return list(obj)
        elif isinstance(obj,np.float32):
            #json won't encode float32; needs to be native python
            #+++ precision needs to be recorded otherwise we alias data!
            return float(obj)
        else:
            return json.JSONEncoder.default(self,obj)

def dict_to_od(d):
    '''
    Simple JSON hook to make sure we get tab completeable dict objects
    '''
    return ObjectDict(d)

def _load_json(filename):
    in_file = open(filename,'r')
    return json.load(in_file,object_hook=dict_to_od)

def _identify_results_files(results_folder):
    regexp = re.compile('^.+.nc$') #pylint: disable=anomalous-backslash-in-string
    return sorted([filename for filename in os.listdir(results_folder) if regexp.match(filename)])

def _variable_name(filename):
    regexp = re.compile('^(.+?)(_\d\d\d\d)?.nc$') #pylint: disable=anomalous-backslash-in-string
    g = regexp.match(filename).groups()
    return g[0]

def _identify_variables(filenames):
    return set([_variable_name(filename) for filename in filenames])

def _infer_time_period(filename,results_folder):
    filenames = sorted(glob(os.path.join(results_folder,_variable_name(filename)+'*.nc')))

    start = start_date(managed_dataset(filenames[0],'r'))
    end = end_date(managed_dataset(filenames[-1],'r'))
    freq = dataset_frequency(managed_dataset(filenames[0],'r'))

    return ObjectDict(type=freq,start=start.strftime('%Y-%m-%d'),
                      end=end.strftime('%Y-%m-%d'),representation='YYYY-MM-DD')

def _infer_extent(filename,results_folder):
    return extents.Extent.from_file(os.path.join(results_folder,filename))

def _infer_model_version(filename,results_folder):
    return 'unknown'

def _index_results(results_folder):
    filenames = _identify_results_files(results_folder)
    expected_variables = _identify_variables(filenames)
    result = ObjectDict()

    result.metadata_from = filenames[0]
    result.name = None
    result.extent = _infer_extent(result.metadata_from,results_folder)
    result.period = _infer_time_period(result.metadata_from,results_folder)
    result.variables = expected_variables
    result.model_version = _infer_model_version(result.metadata_from,results_folder)

    return result

def load_results(results_folder,results_name=None):
    """
    Retrieve a results set from a previously run simulation.

    results_folder - path to a set of AWRAMS results, including netcdf outputs and a results
                     index file (currently 'results.json')
    """
    results_dict = _index_results(results_folder)

    live_results = Results(model_version=results_dict.model_version,results_name=results_name)
    live_results.path = results_folder
    live_results.extent = results_dict.extent
    if results_dict.period.representation == "YYYY-MM-DD":
        results_dict.period.start = results_dict.period.start.replace('-','/')
        results_dict.period.end = results_dict.period.end.replace('-','/')
    live_results.period = dt.dates(results_dict.period.start,results_dict.period.end,freq=results_dict.period.type)

    for v in results_dict.variables:
        live_results._add_variable(v)

    return live_results


class Results(object):
    """
    Provides access to simulation results.
    """
    def __init__(self,model_version,results_name=None):
        self._variables = VariableGroup(self)
        self.extent = None
        self.period = None
        self.name = results_name
        self._path = None
        self._model_version = model_version
        self.parameters = ObjectDict()
        self.parameters.spatial = None
        self.parameters.landscape = None

    def _get_path(self):
        return self._path

    def _set_path(self,path):
        self._path = os.path.abspath(path)

    path = property(_get_path,_set_path)

    def __getitem__(self, item):
        '''
        :param item: hyper slice [variables,period,extent]
        :return:
        '''

        if item[0] == slice(None):
            ### all variables
            inputs = self._variables
        else:
            inputs = _sanitize_inputs(item[0])

        if item[1] == slice(None):
            period = self.period
        else:
            period = item[1]
        period = _sanitize_period(period,inputs.source.period.freq)

        if item[2] == slice(None):
            extent = self.extent
        else:
            extent = item[2]

        class Query:
            def __init__(self,variables,period,extent):
                self.variables = variables
                self.period = period
                self.extent = extent
                self.mpl = None # allows persistance of matplotlib thingy

            def timeseries(self,**kwds):
                '''
                Display data selections as timeseries plots
                plot object persists as <instance>.mpl

                :param kwds: standard matplotlib keyword arguments for manipulating plots
                :return:
                '''
                self.mpl = timeseries(self.variables,self.period,self.extent,**kwds)

            def spatial(self,**kwds):
                '''
                Display data selections as spatial images
                plot object persists as <instance>.mpl

                :param kwds: standard matplotlib keyword arguments for manipulating plots
                :return: None
                '''
                self.mpl = spatial(self.variables,self.period,self.extent,**kwds)

            def get_data_limits(self,colapse_dimension='time'):
                '''
                Get the data range for the selection

                :param colapse_dimension: 'time' or 'spatial'
                :return: minimum,maximum
                '''
                return get_range(self.variables,self.extent,self.period,colapse_dimension=colapse_dimension)


        q = Query(inputs,period,extent)
        for v in inputs:
            v.data = v.get_data(period,extent)

        return q

    def _set_model_version(self,new_version):
        self._model_version = new_version

    def _build_plotvars(self,variables):
        if variables is None:
            return self.variables

        if isinstance(variables,str):
            variables = [variables]

        return [self.variables[v] for v in variables]

    def plot_spatial(self,variables=None,period=None,extent=None):
        inputs = self._build_plotvars(variables)

        if extent is None:
            extent = self.extent

        if period is None:
            period = self.period

        spatial(VariableGroup(self,inputs),period,extent)

    def plot_timeseries(self,variables=None,extent=None,period=None):
        inputs = self._build_plotvars(variables)

        if period is None:
            period = self.period

        if extent is None:
            extent = self.extent

        timeseries(inputs,period,extent)

    def _get_variables(self):
        return self._variables

    def _add_variable(self,name,timesteps=None):
        if timesteps is None:
            timesteps = [frequency.DAILY]
        if frequency.DAILY in timesteps:
            path = self.path
        else:
            path = os.path.join(self.path, timesteps[0].lower())

        new_var = ResultVariable(self,name,path)
        self._variables[new_var.name] = new_var

    def _close_all(self):
        for result in self._variables:
            if result._reader is not None:
                result.reader.close_all()

    variables = property(_get_variables,fset=None,fdel=None,doc='Dictionary of variable available from results')


class ResultVariable(VariableInstance):
    def __init__(self,source,filename,path):
        self.filename = filename
        self.path = path

        pattern = os.path.join(os.path.abspath(path),filename+'*.nc')
        files = sorted(glob(pattern))
        ref_ds = managed_dataset(files[0])
        name = ref_ds.awra_var.name
        if name.startswith('/'):
            name = name[1:]
        units = ref_ds.awra_var.units
        ref_ds.close()

        VariableInstance.__init__(self,source,name,units)

        self._pattern = pattern
        self._reader = None
        self._dataset = ref_ds
        self.agg_method = 'mean' ### temporal aggregate method

    def _get_reader(self):
        if self._reader is None:
            self._reader = ResultsDataSet(self.name,self._pattern)
        return self._reader

    def get_data(self,period,extent):
        self.period = period
        data = self.reader.get_data(period,extent)
        data.mask = extent.mask
        return data

    reader = property(_get_reader)


class ResultsDataSet:
    """
    Essentially a TimeSeriesDataSet but nc files are closed after initial
    scanning to avoid "RuntimeError: too many files open"

    Support retrieval of point time series, single-day grids or data-cubes.
    """
    def __init__(self, variable, search_pattern, period=None, day_exist_chn_name=None):
        self.sfm = SplitFileManager.open_existing('',search_pattern,variable)

    def units(self):
        """
        :return: the units of the data, as stored in the units attribute of main variable
        """
        return self._open_data(self.files[0]).variables[self.variable].units

    def index_for_date(self,filename,date):
        """
        get offset in the time dimension for given date
        :param filename:
        :param date:
        :return: offset
        """
        time_index = self._open_data(filename)['time'][:]
        return np.argwhere(time_index == (date.toordinal() - self.epoch.toordinal()))[0]

    def locate_day(self,date):
        """
        :param date:
        :return: filename and offset in the time dimension for given date
        """

        for filename,(start,end) in list(self.file_map.items()):
            if date >= start and date <= end:
                return filename, self.index_for_date(filename,date)

        raise BaseException("Date out of range in dataset %s: %s"%(self.variable,date))

    def _locate_period(self,time_period):
        # Possible that a single timestamp may get passed in (if we're only asking for one day's data)
        if not isiterable(time_period):
            time_period = [time_period]

        start_index = self.locate_day(time_period[0])
        end_index = self.locate_day(time_period[-1])

        if len(time_period) == 1:
            required_file = self._open_data(start_index[0])
            time_slices = [[required_file,slice(start_index[1],start_index[1]+1)]]
        else:
            required_files = [self._open_data(f) for f in self.files[self.files.index(start_index[0]):self.files.index(end_index[0])+1]]
            time_slices = [[f,slice(None,None)] for f in required_files]
            time_slices[0][1] = slice(start_index[1],time_slices[0][1].stop,time_slices[0][1].step)
            time_slices[-1][1] = slice(time_slices[-1][1].start,end_index[1]+1,time_slices[-1][1].step)

        return time_slices

    def get_data(self,period,extent):
        """
        :param period:
        :param extent:
        :return: datacube for period and extent
        """
        if type(period) == pd.Timestamp:
            period = [period]

        data = self.sfm.get_data(period,extent)#[extent.indices]

        if not hasattr(data,'mask'):
            data = np.ma.MaskedArray(data,mask=False)

        data.mask = np.logical_or(data.mask,extent.mask)

        return simplify(data)


def simplify(data):
    if data.shape[0] == 1:
        return data[0]
    return data

class EnsembleResults:
    '''
    Wrapper to load a set of ensemble results; base_path contains subdirectories each containing a run
    '''
    def __init__(self,base_path):
        subfolders = os.listdir(base_path)
        self.results = {}
        
        self.period = None
        
        for k in subfolders:
            self.results[k] = load_results(os.path.join(base_path,k))
            if self.period is None:
                self.period = self.results[k].period
                self.extent = self.results[k].extent
                self.variables = list(self.results[k].variables._container.keys())
            
    def get_ens_results(self,period,extent,variable):
        data = {}
        for k,v in self.results.items():
            data[k] = v.variables[variable].get_data(period,extent)

        return data
    