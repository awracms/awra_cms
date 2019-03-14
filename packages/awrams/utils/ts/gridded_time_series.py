"""
Climate Data access module for the AWRA Modelling System.

Provides API consistency for access meteorological data from different forms (NetCDF, THREDDS, Flat File),
in a variety of ways (time series, grid)
"""
import netCDF4 as nc
import numpy as np
import awrams.utils.datetools as dt
from awrams.utils.fs import FileMatcher
from awrams.utils.io.netcdf_wrapper import geospatial_reference_from_nc, set_chunk_cache, start_date, end_date, epoch_from_nc
from awrams.utils.ts.time_series_infilling import FailOnDataGaps, FillWithZeros
from awrams.utils.awrams_log import get_module_logger
from awrams.utils.io import db_open_with
import pandas as pd
from awrams.utils.helpers import aquantize
from awrams.utils.settings import VAR_CHUNK_CACHE_SIZE, VAR_CHUNK_CACHE_NELEMS, VAR_CHUNK_CACHE_PREEMPTION, DB_OPEN_WITH #pylint: disable=no-name-in-module
from awrams.utils.io.input_buffer import InputReader
#from db_helper import mdc, _h5py
import re
from awrams.utils.messaging.general import NULL_CHUNK
from collections import Iterable

logger = get_module_logger('climate_data')

def isiterable(obj):
    return isinstance(obj,Iterable)

class BridgedDataSet:
    def connect_reader_bridge(self,bridge):
        self.bridge = bridge
        self.cur_chunk = NULL_CHUNK
        self.cur_chunk_idx = -1
        self.cur_period_idx = -1

    def set_active_period(self,period_idx):
        self.cur_period_idx += 1
        self.cur_chunk_idx = -1
        self.cur_chunk = NULL_CHUNK

    def read_data(self,period,location):
        if not self.cur_chunk.contains(location):
            self.cur_chunk_idx += 1
            self.cur_chunk = self.bridge.chunk_map[self.cur_chunk_idx]
            self.cur_data = self.bridge.get_chunk(self.variable,self.cur_chunk_idx,self.cur_period_idx)

        return self.cur_data[:,self.cur_chunk.idx(location)].astype('d')

    def get_year_map(self):
        year_map = {}
        for f in self._all_files:
            file_year = re.match('.*([0-9]{4})',f).groups()[0]
            year = int(file_year)
            year_map[year] = f
        return year_map

    def _open_data(self,fn):
        try:
            return self.db_open_with(fn,'r') #nc.Dataset(fn,'r')
        except RuntimeError as e:
            e.args += (fn,)
            raise


class TimeSeriesDataSet(object):
    """
    Represent a gridded time series stored in NetCDF using a schema understood
    by the AWRA Modelling System. (eg NetCDFs created by AWRA or the pre-processing
    tools).

    Support retrieval of point time series, single-day grids or data-cubes.
    """
    def __init__(self, variable, search_pattern, period=None, day_exist_chn_name=None):
        self.variable = variable
        self._reader = InputReader(variable)
        self._pattern = search_pattern
        self.db_open_with = db_open_with()

        if not search_pattern:
            search_pattern = variable+"*.nc"

        self.matcher = FileMatcher(search_pattern)
        self._all_files = sorted(self.matcher.locate())

        if not len(self._all_files):
            raise NoMatchingFilesException("No matching files for variable %s using search pattern %s"%(variable,search_pattern))

        if period is not None:
            years = dt.years_for_period(period)
            f_files = []
            for in_file in self._all_files:
                file_year = re.match('.*([0-9]{4})',in_file).groups()[0]
                if int(file_year) in years:
                    f_files.append(in_file)
            self.files = f_files
        else:
            self.files = self._all_files

        if len(self.files) == 0:
            # +++ quick fix for solar if whole period is less than 1990
            self.files = self._all_files[:1]

        self.open_files = [self._open_data(fn) for fn in self.files]

        ref_file = self.open_files[0]
        end_ref_file = self.open_files[-1]

        self.epoch_offset = int(ref_file.variables['time'][0])
        self.epoch = epoch_from_nc(ref_file)
        self.start_date = start_date(ref_file)
        self.end_date = end_date(end_ref_file,day_exist_chn_name)

        self.file_map = {}
        self._map_files()

        self.length = sum([ds.variables[self.variable].shape[0] for ds in self.open_files])
        self.shape = np.ma.concatenate([[self.length],ref_file.variables[self.variable].shape[1:]])
        self.geospatial_reference = geospatial_reference_from_nc(ref_file)

        self.latitudes = ref_file.variables['latitude'][:]
        self.longitudes = ref_file.variables['longitude'][:]


        #+++
        # We probably want to set the MDC in simulation run scenarios, but currently have a lot of tests
        # using netCDF directly;  reincorporate once h5py is used for everything
        #if DB_OPEN_WITH == '_h5py':
        #    self._set_mdc(mdc['K32'])

    def units(self):
        """
        Returns the units of the data, as stored in the units attribute of main variable
        """
        return self.open_files[0].variables[self.variable].units

    def __len__(self):
        return self.length

    def close_all(self):
        """
        Close/Release all NetCDF datasets held by this timeseries dataset
        """
        logger.debug("%s.close_all",self.__class__.__name__)
        for f in self.open_files:
            f.close()
        self.open_files = []

    def _map_files(self):
        for filename,dataset in zip(self.files,self.open_files):
            try:
                self.file_map[filename] = (
                 self.date_for_timestep(dataset.variables['time'][0]),
                 self.date_for_timestep(dataset.variables['time'][-1]))
            except np.ma.core.MaskError:
                self.file_map[filename] = (
                 self.date_for_timestep(dataset.variables['time'][0]),
                 self.date_for_timestep(dataset.variables['time'][:].compressed()[-1]))

    def date_for_timestep(self,timestep):
        """
        Return the date corresponding to a particular timestep,
        where timestep is an index from 0, from the start of
        the climate record for this variable
        """
        return dt.datetime.fromordinal(
           self.start_date.toordinal()+int(timestep-self.epoch_offset))

    def timestep_for_date(self,date):
        """
        Return the timestep corresponding to this date object.

        timestep is an index where 0 refers to the start of
        the climate record for this variable.

        Will return a negative timestep for dates prior to
        the start of the record. Similarly, may return a timestep
        beyond the end of the record
        """
        return date.toordinal() - self.start_date.toordinal()

    def _set_mdc(self,mdc):
        '''
        set metadata cache of all open files
        '''
        for fh in self.open_files:
            h5f = fh.file_id
            mdc_cache_config = h5f.get_mdc_config()
            mdc_cache_config.set_initial_size = mdc[0] #True
            mdc_cache_config.initial_size = mdc[1] #1024
            mdc_cache_config.max_size = mdc[2] #1024
            mdc_cache_config.min_size = mdc[3] #1024
            h5f.set_mdc_config(mdc_cache_config)

    def _open_data(self,fn):
        try:
            return self.db_open_with(fn,'r') #nc.Dataset(fn,'r')
        except RuntimeError as e:
            e.args += (fn,)
            raise

    def cell_for_location(self,location):
        """
        Return the cell index, within this spatial time series, of a given geographic location
        """
        return self.geospatial_reference.cell_for_geo(*location)

    def location_for_cell(self,cell):
        """
        Returns the geographic coordinates [lat,lon] for a given cell in this dataset
        """
        return [self.latitude_for_cell(cell),self.longitude_for_cell(cell)]

    def longitude_for_cell(self,cell):
        """
        Return the longitude for a given cell in this dataset,
        as required by routines that calculate the astronomical
        parameters (eg fday)
        """
        #return self.open_files[0].variables['longitude'][cell[1]]
        return self.longitudes[cell[1]]

    def latitude_for_cell(self,cell):
        """
        Return the latitude for a given cell in this dataset,
        as required by routines that calculate the astronomical
        parameters (eg fday)
        """
        #return self.open_files[0].variables['latitude'][cell[0]]
        return self.latitudes[cell[0]]

    def locate_day(self,date):
        """
        Returns NetCDF dataset and offset in the time dimension for given day
        """
        for filename,(start,end) in list(self.file_map.items()):
            if date >= start and date <= end:
                return (self.open_files[self.files.index(filename)],
                    self.timestep_for_date(date)-self.timestep_for_date(start))

        raise BaseException("Date out of range in dataset %s: %s"%(self.variable,date))

    def _locate_period(self,time_period):
        # Possible that a single timestamp may get passed in (if we're only asking for one day's data)
        if not isiterable(time_period):
            time_period = [time_period]

        start_index = self.locate_day(time_period[0])
        end_index = self.locate_day(time_period[-1])
        all_files = self.open_files
        required_files = all_files[all_files.index(start_index[0]):all_files.index(end_index[0])+1]
        time_slices = [[f,slice(None,None)] for f in required_files]
        time_slices[0][1] = slice(start_index[1],time_slices[0][1].stop,time_slices[0][1].step)
        time_slices[-1][1] = slice(time_slices[-1][1].start,end_index[1]+1,time_slices[-1][1].step)

        return time_slices

    def retrieve_time_series(self,cell,start=None,end=None):
        """
        Retrieve timeseries of the variable at a given cell (row,col)
        """

        if (not start is None) or (not end is None):
            return self.retrieve([self.start_date if start is None else start,
                             self.end_date if end is None else end],cell)
        return np.ma.concatenate(
                  [ds.variables[self.variable][:,cell[0],cell[1]]
                  for ds in self.open_files])

    def retrieve_grid(self,date):
        """
        Retrieve a grid of the variable for a given day
        """
        ncd_file,offset = self.locate_day(date)
        return ncd_file.variables[self.variable][offset,:,:]

    def fast_retrieve_grid(self,date,extent):
        """
        Retrieve a grid for the given extent; extent must _pretranslated_
        to our origin, no bounds checking is done
        (Called by stream methods who repeatedly use the same extent)
        Additionally, no extra mask information is applied
        """
        ncd_file,offset = self.locate_day(date)
        return ncd_file.variables[self.variable][offset,extent.x_index,extent.y_index]

    def _generate_padding(self,from_date,until_date,spatial_shape=None):
        pad_len = (until_date-from_date).days
        if spatial_shape:
            our_arr = np.empty(np.concatenate([[pad_len],spatial_shape]))
        else:
            out_arr = np.empty((pad_len))
        out_arr.fill(np.nan)
        return out_arr

    def _simplify(self,data):
        if data.shape[0] == 1:
            return data[0]
        return data

    def retrieve(self,time_period,spatial_slice):
        """
        Retrieve a block of data, 0-3 dimensions, covering the time_period
        and the spatial_slice

        time_period: An object that supports simple indexing to retrieve the
                     start and end of the desired time period (eg time_period[0] or [-1])

        spatial_space: Two objects that can be used to spatially slice the data.
                       Simple indices (eg [50]) will work, as will slice objects


        Currently only used for time slices, ++++ But, padding does not work for spatial regions


        """
        pad = False
        time_period = [time_period[0],time_period[-1]]

        pre_padding = []
        if time_period[0] < self.start_date:
            pad_until = min(time_period[-1]+dt.timedelta(1),self.start_date)
            pre_padding = self._generate_padding(time_period[0],pad_until)
            if pad_until < self.start_date:
                return self._simplify(pre_padding)

            time_period[0] = self.start_date
            pad = True

        post_padding = []
        if time_period[-1] > self.end_date:
            pad_from = max(time_period[0],self.end_date+dt.timedelta(1))
            post_padding = self._generate_padding(pad_from,time_period[-1]+dt.timedelta(1))

            if pad_from > self.end_date:
                return self._simplify(post_padding)

            time_period[-1] = self.end_date
            pad = True

        time_slices = self._locate_period(time_period)
        try:
            self._reader.ncd_time_slices = time_slices
            data = self._reader[spatial_slice]
            #try:
            #    self.chunk_reader.ncd_time_slices = time_slices
            #    data = self.chunk_reader[spatial_slice]
            #except AttributeError:
            #    data = np.ma.concatenate(
            #            [_slice[0].variables[self.variable][_slice[1],spatial_slice[0],spatial_slice[1]]
            #            for _slice in time_slices])
        except IndexError as ie:
            msg = """Attempt to retrieve data beyond extent. Requested (%s,%s) on spatial timeseries with
            spatial shape (%d,%d)"""
            available_shape = self.open_files[0].variables[self.variable].shape
            logger.critical(msg,str(spatial_slice[0]),str(spatial_slice[1]),self.shape[1],self.shape[2])
            raise

        if pad:
            data = np.ma.concatenate([pre_padding,data,post_padding])

        return self._simplify(data)

    def get_data(self,period,extent):
        '''
        Standard 'datacube' return method
        '''
        if type(period) == pd.Timestamp:
            period = [period]


        #++++
        #While we currently translate the extent, we don't apply any extra
        #mask information; possibly need an inherited 'get_data' behaviour
        #from a master class that knows about extents

        extent = extent.translate_to_origin(self.geospatial_reference)

        self._reader.ncd_time_slices = self._locate_period([period[0],period[-1]])
        data = self._reader[extent.indices]

        data.mask = np.logical_or(data.mask,extent.mask)

        return self._simplify(data)

    def earliest_start(self):
        """
        What is the earliest start date availble from this timeseries data?

        Typically this is just the start of the record, but may be modified by subclasses
        (eg with gap-filling or series extension)
        """
        return self.start_date

class ClimateDataSet(TimeSeriesDataSet):
    """
    Represent a single variable of meteorlogical inputs for AWRAMS.

    Assumptions:
     * Data stored in NetCDF files with 1 year of daily data per file.
     * Year (yyyy) should appear in filenames
     * All relevant files in one directory
    """
    def __init__(self,variable,search_pattern=None,period=None): #, _reader=None):
        """
        Create a ClimateDataSet for a given variable

        By default, looks for files matching <variable>*.nc in the current directory. Override with search_pattern
        If no period is supplied, will use all available data
        """

        super(ClimateDataSet,self).__init__(variable,search_pattern, period,day_exist_chn_name='exist')
        self.gap_filler = FailOnDataGaps()
        #self.gap_filler = FillWithZeros()

    def set_reader_mode(self, reader):
        """either InputReader (default) or InputChunkReader"""
        self._reader = reader(self.variable)

    def retrieve_time_series(self,location,start=None,end=None):
        series = super(ClimateDataSet,self).retrieve_time_series(location,start,end)
        if self.gap_filler.has_gaps(series,location):
            series = self.gap_filler.fill(series,location,self,series_start=start)

        return series

    def earliest_start(self):
        if type(self.gap_filler) == FailOnDataGaps:
            return self.start_date
        else:
            return self.epoch #dt.dates("1911-01-01")


class SplitClimateDataSet(BridgedDataSet,TimeSeriesDataSet):
    def __init__(self,variable,search_pattern=None,period=None):
        super(SplitClimateDataSet,self).__init__(variable,search_pattern, period,day_exist_chn_name='exist')
        self.gap_filler = FailOnDataGaps()
        #self.gap_filler = FillWithZeros()
        self.close_all()

    def earliest_start(self):
        if type(self.gap_filler) == FailOnDataGaps:
            return self.start_date
        else:
            return self.epoch #dt.dates("1911-01-01")

    def set_reader_mode(self, reader):
        pass

    def retrieve_time_series(self,location,start=None,end=None):
        '''
        Read valid data if available, otherwise return pure climatology
        '''

        if start < self.start_date:
            if end > self.start_date:
                raise Exception("Only whole years of missing data valid in SplitClimateDataSet")
            period = dt.dates(start,end)
            out_series = self.gap_filler.create_for_period(period,location)
            return out_series

        series = self.read_data(None,location)
        if self.gap_filler.has_gaps(series,location):
            series = self.gap_filler.fill(series,location,self,series_start=start)

        return series


class SplitDataSet(BridgedDataSet):
    '''
    Ultra-minimal dataset class; no gap filling etc, just maps files
    for the BufferedReadProcess class
    '''
    def __init__(self,variable,search_pattern=None,period=None):
        self.variable = variable
        self.db_open_with = db_open_with()

        if not search_pattern:
            search_pattern = variable+"*.nc"

        self.matcher = FileMatcher(search_pattern)
        try:
            self._all_files = sorted(self.matcher.locate())
        except:
            from awrams.utils.io.messages import MSG_CANT_MATCH_VARIABLE_WITH_PATTERN
            logger.critical(MSG_CANT_MATCH_VARIABLE_WITH_PATTERN,variable,search_pattern)
            raise

        if not len(self._all_files):
            raise BaseException("No matching files for variable %s using search pattern %s"%(variable,search_pattern))

        if period is not None:
            years = dt.years_for_period(period)
            f_files = []
            for in_file in self._all_files:
                file_year = re.match('.*([0-9]{4})',in_file).groups()[0]
                if int(file_year) in years:
                    f_files.append(in_file)
            self.files = f_files
        else:
            self.files = self._all_files

        if len(self.files) == 0:
            # +++ quick fix for solar if whole period is less than 1990
            self.files = self._all_files[:1]

        ref_file = self._open_data(self.files[0])
        end_ref_file = self._open_data(self.files[-1]) if len(self.files)>1 else ref_file

        self.epoch_offset = int(ref_file.variables['time'][0])
        end_offset = int(end_ref_file.variables['time'][-1])
        self.epoch = epoch_from_nc(ref_file)
        self.start_date = dt.datetime.fromordinal(self.epoch.toordinal()+self.epoch_offset)
        self.end_date = dt.datetime.fromordinal(self.epoch.toordinal()+end_offset)

        self.geospatial_reference = geospatial_reference_from_nc(ref_file)

        self.latitudes = aquantize(np.float64(ref_file.variables['latitude'][:]),0.05)
        self.longitudes = aquantize(np.float64(ref_file.variables['longitude'][:]),0.05)

        if 'var_name' in ref_file.ncattrs():
            self.variable = ref_file.getncattr('var_name')

        try:
            self.units = ref_file.variables[self.variable].units
        except AttributeError:
            pass

        ref_file.close()
        if len(self.files)>1:
            end_ref_file.close()

def dataset_for_inputs(inputs,ds_class=SplitClimateDataSet,gf_class=FillWithZeros):
    '''
    Helper function to build datasets from MET_INPUT_VARIABLES setting
    '''
    out = {}
    for k,v in list(inputs.items()):
        out[k] = ds_class(k,v)
        out[k].gap_filler = gf_class()
    return out


class NoMatchingFilesException(Exception):
    def __init__(self,msg):
        self.message = msg
