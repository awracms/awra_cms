from awrams.utils import extents
import pandas as pd
import awrams.utils.datetools as dt
import multiprocessing as mp
import numpy as np
#from awrams.utils.io.input_receiver import ClimateInputBridge
from awrams.utils.mapping_types import *
import awrams.utils.awrams_log

logger = awrams.utils.awrams_log.get_module_logger('processing')
'''

'''

class ProcessRunner:
    '''
    Connects parralel IO to processing modules
    '''
    def __init__(self,inputs,processors,period,extent,existing_inputs=False):
        '''
        Inputs: {var_name: input_reader}
        Processors: {var_name: [p0, p1...]}
        '''
        self.inputs = inputs
        self.processor_map = processors
        self.period = period
        self.split_periods = dt.split_period(self.period,'a')
        
        self.control = mp.Queue()
        self.input_mapper = ClimateInputBridge(inputs,self.split_periods,extent,existing_inputs)

        #self.extent = extent.translate_to_origin(self.input_mapper.input_bridge.geo_ref)

        self.extent = extent

    def run(self):

        logger.info("Running")

        progress_cells = 2000

        try:
            #gc.disable()
            #gc.enable()
            

            total_cell_days = float(len(self.period) * self.extent.cell_count)
            cell_days_done = 0

            cells = self.extent.cell_list()

            #for procs in self.processor_map.values():
            #    for processor in procs:
            #        processor.set_extent(self.extent)
            #processor.set_extent(self.input_mapper.input_bridge.geo_ref)

            for cur_period in self.split_periods:

                logger.info("Running period: %s: ",dt.pretty_print_period(cur_period))
                pct_done = (cell_days_done / total_cell_days) * 100.0
                logger.info("%.2f%% complete",pct_done)

                self.input_mapper.set_active_period(cur_period)
                for procs in list(self.processor_map.values()):
                    for processor in procs:
                        processor.set_active_period(cur_period)

                for cell in cells:
                    try:
                        msg = self.control.get_nowait()
                        if (msg):
                            if msg['message'] == 'error':
                                logger.error("Error running cell (%d,%d): %s",cell[0],cell[1],msg)
                                raise Exception("%s", msg)
                    except mp.queues.Empty:
                        pass
            
                    cell_days_done += len(cur_period)

                    data = self.input_mapper.get_cell_dict(cell)

                    for k, procs in list(self.processor_map.items()):
                        for processor in procs:
                            # +++ assuming mask value is -999.
                            processor.process_cell(np.ma.masked_values(data[k],-999.),cell)

            for procs in list(self.processor_map.values()):
                for processor in procs:
                    processor.finalise()
        except:
            raise
        finally:
            self.input_mapper.terminate()

            logger.info("Finished")

            while not self.control.empty():
                msg = self.control.get()
                if msg['message'] == 'occupancy':
                    logger.info("%s: %.2f%% occupancy" % (msg['process'], msg['value']*100.))
                else:
                    logger.info("%s" % msg)

class ProcessStatesRunner:
    '''
    Connects parralel IO to processing modules
    '''
    def __init__(self,inputs,processors,extent, period):
        '''
        Inputs: {var_name: input_reader}
        Processors: {var_name: [p0, p1...]}
        '''

        self.processor_map = processors

        self.input_mapper = StatesInputReader(inputs,period)

        self.extent = extent
        self.period = period
        self.inputs = inputs

    def run(self):

        logger.info("processing... %s",self.inputs)

        progress_cells = 2000

        try:
            total_cells = self.extent.cell_count
            cells_done = 0

            cells = self.extent.cell_list()


            for procs in list(self.processor_map.values()):
                for processor in procs:
                    processor.set_active_period(self.period)

            for cell in cells:

                pct_done = (cells_done / total_cells) * 100.0
                if not cells_done % progress_cells:
                    logger.info("%.2f%% complete",pct_done)
                cells_done += 1

                data = self.input_mapper.get_cell(cell)

                for k, procs in list(self.processor_map.items()):
                    for processor in procs:
                        # +++ assuming mask value is -999.
                        processor.process_cell(np.ma.masked_values(data,-999.),cell)

            for procs in list(self.processor_map.values()):
                for processor in procs:
                    processor.finalise()
        except:
            raise


class StatesInputReader:
    def __init__(self, input_map,period):
        import netCDF4 as nc
        var_name = list(input_map.keys())[0]
        self.ncd = nc.Dataset(input_map[var_name],'r')
        self.variable = self.ncd.variables[self.ncd.var_name]
        self._current_chunk_idx = (None,None)
        self.period = period
        self._period_idx = self._get_period_idx()

        from awrams.utils.settings import DEFAULT_CHUNKSIZE
        self.cs = DEFAULT_CHUNKSIZE

    def get_cell(self,cell):
        ci = self._get_chunk_idx(cell)

        if ci != self._current_chunk_idx:
            self._read_chunk(ci)
            self._current_chunk_idx = ci

        return self._data[:,cell[0]%self.cs[1],cell[1]%self.cs[2]]

    def _get_period_idx(self):
        from awrams.utils.io.netcdf_wrapper import epoch_from_nc

        epoch = epoch_from_nc(self.ncd)
        dti = pd.DatetimeIndex([(epoch + dt.days(int(ts))) for ts in self.ncd.variables['time']],freq='d')
        return dti.searchsorted(self.period)

    def _get_chunk_idx(self,cell):
        return int(cell[0] / self.cs[1]), int(cell[1] / self.cs[2])


    def _read_chunk(self, chunk_idx):
        lat_idx = np.s_[chunk_idx[0] * self.cs[1] : (chunk_idx[0]+1) * self.cs[1]]
        lon_idx = np.s_[chunk_idx[1] * self.cs[2] : (chunk_idx[1]+1) * self.cs[2]]
        self._data = self.variable[self._period_idx,lat_idx,lon_idx]

class Processor:
    def set_extent(self,extent):
        pass

class CellMapProcessor: 
    '''
    Passes cell processing on to mapped targets
    '''
    def __init__(self):
        self.cell_map = {}
        self.targets = []
        self.extent_map = {}

    def _map_cell(self,cell,target):
        cell_key = (cell[0],cell[1])
        if cell_key not in self.cell_map:
            self.cell_map[cell_key] = []
        self.cell_map[cell_key].append(target)

    def add_target(self,extent,target):
        '''
        Map a target processor to an extent
        '''
        if extent not in self.extent_map:
            self.extent_map[extent] = []

        self.extent_map[extent].append(target)

        if target not in self.targets:
            self.targets.append(target)

        for cell in extent:
            self._map_cell(cell,target)

    def set_extent(self,extent):
        self.cell_map = {}
        for e, targets in list(self.extent_map.items()):
            e = e.translate_to_origin(extent.geospatial_reference())
            #for target in targets:
            #    target.set_extent(e)
            for cell in e:
                for target in targets:
                    self._map_cell(cell,target)
        #for target in self.targets:
        #    target.set_extent(extent)


    def process_cell(self,data,cell):
        '''
        Process a single cell (AWRA-L), propogate through the chain
        '''
        cell_key = (cell[0],cell[1])
        if cell_key in self.cell_map:
            targets = self.cell_map[cell_key]
            for processor in targets:
                processor.process_cell(data,cell)

    def set_active_period(self,period):
        '''
        Set the current active period; propogate changes through the chain
        '''
        for processor in self.targets:
            processor.set_active_period(period)

    def finalise(self):
        '''
        Finalise all processing
        '''
        for processor in self.targets:
            processor.finalise()

class SpatialAggregateCellProcessor(Processor):
    '''
    Produce an area-correct mean aggregation for a given extent
    '''
    def __init__(self,period,extent,mode='mean',nan_check=False):
        if not hasattr(extent,'areas'):
            extent.compute_areas()

        self.extent = extent
        self.period = period
        self.mode = mode
        self.nan_check = nan_check

    def set_extent(self,extent):
        if not hasattr(extent,'areas'):
            extent.compute_areas()
        self.extent = extent

    def set_active_period(self,period):
        self.cells_done = 0
        self._data = np.ma.zeros(len(period))
        # cell-timeseries with missing data should not have their area contribute to the extent area, attempt to allow for this
        self._area = np.zeros(len(period))
        self._area_mask = np.ma.ones(len(period))

    def process_cell(self,data,cell):
        lcell = self.extent.localise_cell(cell)
        self._data += self.extent.areas[lcell[0],lcell[1]] * np.ma.filled(data, fill_value=0.) #data
        try:
            self._area_mask.mask = data.mask
        except AttributeError:
            pass
        if self.nan_check:
            self._area += self.extent.areas[lcell[0],lcell[1]] * np.ma.filled(self._area_mask, fill_value=0.)
        else:
            self._area += self.extent.areas[lcell[0],lcell[1]]
        self.cells_done += 1

        if self.cells_done == self.extent.cell_count:
            #if self.nan_check:
            self._area = np.where(self._area == 0., 1., self._area)
            if self.mode == 'mean':
                #return self._data / self.extent.area
                # area corrected for missing data
                return self._data / self._area
            elif self.mode == 'sum':
                return self._data

    def finalise(self):
        pass

class TimeSeriesCollector(Processor):
    '''
    Collect a single cell from a run into a TimeSeries
    ;Ignores the 'cell' argument and collects any data it receives
    '''
    def __init__(self,period):
        self.period = period
        self.pmap = dt.period_map(period)
        self._data = np.ma.zeros(len(period))

    def set_active_period(self,period):
        self.cur_idx = self.pmap[period].values

    def process_cell(self,data,cell):
        self._data[self.cur_idx] = data
        
    def finalise(self):
        self.data = pd.TimeSeries(index=self.period,data=self._data)

'''
Time filtering and resampling
'''

def eom_filter(in_date):
    return in_date == dt.end_of_month(in_date)

def eoy_filter(in_date):
    return in_date == dt.end_of_year(in_date)

def filter_index(filter_fn,data):
    '''
    Return (index,data) for members of sequence 'data' where 'filter_fn' evaluates True
    '''
    index = []
    out_data = []
    for i,v in enumerate(data):
        if filter_fn(v):
            index.append(i)
            out_data.append(v)
    return index,pd.DatetimeIndex(out_data)

class TimeFilter:

    def __init__(self,filter_fn,period):
        self.filter_fn = filter_fn
        self.valid_idx,self.valid_period = filter_index(self.filter_fn, period)

    def filter_data(self,data):
        '''
        Filter a DatetimeIndex by filter_fn, return data indexed by the results,
        along with the filtered DTI
        Right now this is hardcoded to always assume in the period is the the same
        as the full_period supplied as an init argument (which it always is when
        doing AWRAL_runs, currently)
        '''
        valid_data = data[[self.valid_idx]]
        return valid_data

    def filter_period(self,period):
        '''
        Filter a datetimeindex by filter_fn
        '''
        valid_idx,valid_period = filter_index(self.filter_fn, period)
        return pd.DatetimeIndex(valid_period)


class TimeFilteredProcessor:
    def __init__(self,filter_fn,in_coords):
        self.filter_fn = filter_fn
        out_period = TimeFilter(filter_fn,in_coords.time.index).valid_period
        self.out_coords = CoordinateSet([TimeCoordinates(awrams_time,out_period),in_coords.latitude,in_coords.longitude])

    def set_active_period(self,period):
        self.cur_period = period
        self.filter = TimeFilter(self.filter_fn,period)
        return self.filter.valid_period

    def process_cell(self,data,cell):
        out_data = self.filter.filter_data(data)
        if len(out_data) > 0:
            return out_data

    def finalise(self):
        pass

class TimeResampleProcessor:
    def __init__(self,freq,method,coordinates):
        self.in_coords = coordinates
        self.freq = dt.validate_timeframe(freq)
        if method == 'sum':
            self.method = np.sum
        elif method == 'mean':
            self.method = np.mean
        out_period = dt.resample_dti(self.in_coords.time.index,freq)
        self.out_coords = CoordinateSet([TimeCoordinates(awrams_time,out_period),coordinates.latitude,coordinates.longitude])

    def process(self,data,coords):
        ts = pd.TimeSeries(index = coords.time.index,data=data)
        ts_out = ts.resample(self.freq,self.method)
        out_cs = CoordinateSet([TimeCoordinates(awrams_time,ts_out.index),coords.latitude,coords.longitude])
        return np.array(ts_out), out_cs

    def process_cell(self,data,cell):
        '''
        Process a single cell (AWRA-L)
        '''
        return resample_with_index(data,self.res_idx,self.method)

    def set_active_period(self,period):
        self.cur_period = period
        self.res_idx = build_resample_index(period,self.freq)
        res_dti = dt.resample_dti(period,self.freq,as_period=True)
        return res_dti

    def finalise(self):
        pass

def resample_with_index(data,index,method=np.sum):
    '''
    Resample a series from the given indices using ufunc: method
    '''
    outshape = [len(index)]+[s for s in data.shape[1:]]
    out = np.empty(outshape)
    for i,idx in enumerate(index):
        out[i] = method(data[idx],axis=0)
    return out

def resample_with_weighted_mean(data,index,weights):
    '''
    Resample a series from the given indices using ufunc: method
    Scale the outputs by <weights>, where weights is the same
    shape as index
    '''
    outshape = [len(index)]+[s for s in data.shape[1:]]
    out = np.empty(outshape)
    for i,idx in enumerate(index):
        cur_data = data[idx]
        out[i] = np.sum([cur_data[j]*weights[i][j] for j in range(len(weights[i]))],axis=0)/sum(weights[i])
        #out[i] = method(data[idx]*weights[i],axis=0)/sum(weights[i])
    return out


def resample_with_index_monthly(data,index):
    '''
    Fast monthly resampler +++ only works for summation
    No nan_check - should 0-out any cells that should not be included in the input
    '''
    interim = np.zeros((31,len(index)))
    for i,idx in enumerate(index):
        interim[:idx.stop-idx.start,i] = data[idx]
    return np.sum(interim,0)

def resample_with_mask_index_monthly(data,sample_idx,mask_idx):
    '''
    Fast monthly resampler; masks out precomputed invalid data (mask_idx)
    '''
    interim = np.zeros((31,len(sample_idx)))
    vdata = data[mask_idx]
    for i,idx in enumerate(sample_idx):
        interim[:idx.stop-idx.start,i] = vdata[idx]
    return np.sum(interim,0)

def build_resample_index(period,timeframe,window=None):
    '''
    Return (slice/integer) indices matching the boundaries of a resampled period
    Optionally supply a window period (ie only produces indices within the window)
    '''
    tc = period_to_tc(period)
    tf = dt.validate_timeframe(timeframe)

    if window is None:
        window = period

    new_p = dt.resample_dti(window,tf,as_period=True)

    _,end_of_p,_ = dt.boundary_funcs(tf)

    if isinstance(period,pd.PeriodIndex):
        def enforce_freq(ts):
            return ts.to_period(period.freq)
    else:
        def enforce_freq(ts):
            return ts

    indices = []
    for p in new_p:
        s = enforce_freq(p.start_time)
        e = enforce_freq(end_of_p(p.start_time))
        indices.append(tc.get_index(slice(s,e)))

    return indices

def build_weighted_resample_index(period,timeframe,ref_frame='d',window=None):
    index = build_resample_index(period,timeframe,window)
    weights = []
    for i,idx in enumerate(index):
        sub_idx = period[idx]
        subweight = []
        for p in sub_idx:
            start = p.start_time
            end = p.end_time
            plen = 1 + (end.to_period(ref_frame) - start.to_period(ref_frame))
            subweight.append(plen)
        weights.append(subweight)
    return index,weights

def build_masksample_indices(period,timeframe,mask,min_valid=1):
    '''
    Build resample indices for the supplied timeframe with a validity threshold
    '''
    tc = period_to_tc(period)
    tf = dt.validate_timeframe(timeframe)

    new_p = dt.resample_dti(period,tf,as_period=True)

    _,end_of_p,_ = dt.boundary_funcs(tf)

    s_mask = mask.copy()
    sample_indices = []

    cur_s = 0
    cur_e = 0

    for p in new_p:
        cur_idx = tc.get_index(slice(p.start_time,end_of_p(p.start_time)))
        v_count = (~mask[cur_idx]).sum()
        if ( v_count >= min_valid ):
            cur_e += v_count
            sample_indices.append(slice(cur_s,cur_e))
            cur_s = cur_e
        else:
            s_mask[cur_idx] = True

    return sample_indices,np.where(s_mask==False)[0]


'''
Classes for connecting Processors
'''

class ProcessChain(Processor):
    '''
    Chain together a set out of output processes
    '''
    def __init__(self,processors):
        self.chain = processors

    def process(self,data,coords):
        #+++ Currently deprecated
        for processor in self.chain:
            res = processor.process(data,coords)
            if res != None:
                data,coords = res[0],res[1]
            else:
                pass
        return data, coords

    def process_cell(self,data,cell):
        '''
        Process a single cell (AWRA-L), propogate through the chain
        '''
        for processor in self.chain:
            data = processor.process_cell(data,cell)
            if data is None:
                break
        return data

    def set_extent(self,extent):
        for processor in self.chain:
            processor.set_extent(extent)

    def set_active_period(self,period):
        '''
        Set the current active period; propogate changes through the chain
        '''
        for processor in self.chain:
            new_period = processor.set_active_period(period)
            if not new_period is None:
                period = new_period
        return period

    def finalise(self):
        '''
        Finalise all processing
        '''
        for processor in self.chain:
            processor.finalise()

class ProcessSplitter:
    '''
    Send cell processing to multiple processors
    '''
    def __init__(self, processors):
        self.processors = processors

    def process_cell(self,data,cell):
        '''
        Process a single cell (AWRA-L) with multiple processors
        '''
        for processor in self.processors:
            processor.process_cell(data,cell)

    def set_active_period(self,period):
        for processor in self.processors:
            processor.set_active_period(period)

    def finalise(self):
        for processor in self.processors:
            processor.finalise()

'''
Utility functions (likely to become deprecated)
'''

def sum_over_time(variable,period,extent):
    '''
    Extract daily data for whole period and provide summed output
    '''

    #out_data = extents.zeros_like(extent)

    #for ts in period:
    #    out_data += variable.get_data([ts],extent)

    data = variable.get_data(period,extent)
    variable.data = data

    if len(period) > 1:
        data = data.sum(0)
        
    variable.agg_data = data
    return data

def mean_over_time(variable,period,extent):
    '''
    Extract daily data for whole period and provide mean output
    '''

    if 'CONSTANT_IN_TIME' in list(variable.meta.keys()):
        return variable.get_data(period,extent)

    data = sum_over_time(variable,period,extent)
    data /= len(period)
    variable.agg_data = data
    return data


def spatial_aggregate(variable,period,extent):

    out_data = pd.TimeSeries(index=period,name=variable.name)

    data = variable.get_data(period,extent)

    if extent.cell_count == 1:
        if len(data.shape) == 3:
            out_data[:] = data[:,0,0]
        else:
            out_data[:] = data

    else:

        out_data[:] = (data * extent.areas).sum(axis=(1,2))  ### sum over spatial dimensions

        out_data = out_data / extent.area

    variable.agg_data = out_data
    return out_data
