'''
Provides functionality for filling gaps in gridded time series using, for example,
precomputed climatologies.
'''

import numpy as np
import awrams.utils.datetools as dt
from awrams.utils.fs import md5_file

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('climate_data_infilling')

FILL_VALUE = -999.0

class ClimateDataGapFiller(object):
    def any_nans(self,series):
        # Optimisation for rapid detection of nans
        # See http://stackoverflow.com/a/6736970
        return np.isnan(np.sum(series))

    def has_gaps(self,series,location):
        if self.any_nans(series):
             msg = "NaNs in Series at %s"
             logger.debug(msg,str(location))
             return msg%str(location)

        if (series == FILL_VALUE).any():
            msg = "Series masked at %s"
            logger.debug(msg,str(location))
            return msg%str(location)

        return False

    def gap_list(self,series):
        return np.where(np.logical_or(series==FILL_VALUE,np.isnan(series)))[0].tolist()

    def fill(self,series,cell,main_source,series_start):
        raise NotImplementedError('ClimateDataGapFiller:fill not implemented')

    def create_for_period(self,period,location):
        raise NotImplementedError('ClimateDataGapFiller:create_for_period not implemented')
        
    def provenance(self):
        return {"strategy":self.__class__.__name__}

class FailOnDataGaps(ClimateDataGapFiller):
    def fill(self,series,location,main_source,series_start):
        message = self.has_gaps(series,location)
        if message:
            logger.critical(message)
            logger.info("The following timesteps have missing values for %s in %s, %s",
                        main_source.variable,str(location),str(self.gap_list(series)))
            raise BaseException(message)

class FillWithZeros(ClimateDataGapFiller):
    def fill(self,series,location,main_source,series_start):
        gaps = self.gap_list(series)
        outseries = series.copy()
        outseries[gaps] = 0.0
        return outseries

class Climatology(object):
    def __init__(self, variable, filename):
        self.data = {}
        self.filename = filename
        self.variable = variable
        self._current_row = -1
        self._is_leap_year = True

        import netCDF4 as nc
        self.ncd = nc.Dataset(filename, 'r')
        if 'month' in self.ncd.variables:
            self.freq = 'monthly'
        else:
            self.freq = 'daily'
            if self.ncd.variables['day_of_year'].size == 365:
                self._is_leap_year = False

        logger.info("Loading %s climatology from %s",self.freq,filename)
        #self.data = ncd.variables[variable][:]
        #for m in range(12):
        #    self.data[m + 1] = d[m,:,:]

    def get(self,month,location):
        return self.data[month][location[0],location[1]]

    def get_for_location(self,location):
        if location[0] != self._current_row:
            self._read_row(location[0])
        return self.data[:,location[1]]

    def _read_row(self,row):
        self.data = self.ncd.variables[self.variable][:,row,:]
        self._current_row = row

class FillWithClimatology(ClimateDataGapFiller):
    def __init__(self,climatology):
        self.climatology = climatology

#       {month:gd.Open(filename).GetRasterBand(1).ReadAsArray() for (month,filename) in zip(months,filenames)}

    def get(self,month,location):
        return self.climatology.get(month,location)

    def fill(self,series,location,main_source,series_start):

        gaps = self.gap_list(series)

        outseries = series.copy()

        if len(gaps) > 0:
            if series_start is None:
                series_start = main_source.start_date
            series_start = series_start.toordinal()

            if self.climatology.freq == 'monthly':
                idx = [dt.datetime.fromordinal(series_start+gap).month - 1 for gap in gaps]
                #months = [dt.datetime.fromordinal(series_start+gap).month - 1 for gap in gaps]
            else: # is daily
                idx = gaps

            # TODO Possible optimise by computing a set of months and extracting those value then
            # broadcasting out to the full list of gaps...
            loc_data = self.climatology.get_for_location(location)
            #values = [self.get(m,location) for m in months]
            outseries[gaps] = loc_data[idx] #[months]

        return outseries

    def create_for_period(self,period,location):
        '''
        Fill a complete period's worth of data with climatology (ie assume all gaps)
        '''
        loc_data = self.climatology.get_for_location(location)

        if self.climatology.freq == 'monthly':
            out_values = loc_data[period.month-1]
        elif self.climatology.freq == 'daily':
            if not dt.is_leap_year(period.year[0]) and self.climatology._is_leap_year:
                idx = period.dayofyear - 1
                if len(idx) > 59:
                    idx[59:] += 1 # shift by 1 day to allow for climatology including feb 29
                return loc_data[idx]

            return loc_data[period.dayofyear - 1]
        else:
            raise Exception("Climatology not recognised as monthly or daily for %s",self.climatology.filename)

        return out_values

    def provenance(self):
        result = super(FillWithClimatology,self).provenance()
        result['filename'] = self.climatology.filename
        logger.info("Generating MD5 summaries for climatology file %s",self.climatology.filename)
        result['md5'] = md5_file(self.climatology.filename)
        return result
        
class FillWithLatest(ClimateDataGapFiller):
    def __init__(self, period):
        self.period = period
        self.last_value = None # this needed to carry value over from last year if need be
        
    def fail(self,series,location,main_source,series_start):
        fail = FailOnDataGaps()
        fail.fill(series,location,main_source,series_start)

    def fill(self,series,location,main_source,series_start):
        if series_start.year < self.period[-1].year:
            # this type of fill only used for padding end of sim period where climate inputs don't exist
            self.fail(series,location,main_source,series_start)
        
        first_masked_index = None
        outseries = np.ma.MaskedArray(series.copy())
        outseries.mask = np.logical_or(series==FILL_VALUE,np.isnan(series))
        gaps = np.ma.flatnotmasked_edges(outseries)
        if gaps is None:
            first_masked_index = 0
            if self.last_value is None:
                raise Exception("series is all null, no value to use for padding")
        elif gaps[1] == -1: # has gaps but not at end so fail
            self.fail(series,location,main_source,series_start)
        else:
            self.last_value = series[gaps[1]]
            first_masked_index = gaps[1] + 1

        if first_masked_index is not None:
            outseries[first_masked_index:] = self.last_value
        if self.has_gaps(outseries,location): # fail if gaps in the middle of timeseries
            self.fail(outseries,location,main_source,series_start)
        return outseries.data
