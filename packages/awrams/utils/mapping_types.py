'''

Defines the datatypes used for analysis and persistence

Variables represent the conceptual objects that may be
used in analysis and modelling

MappedVariable objects refer to the concrete realisation of data
associated with variable(s) within a dataset

e.g "temp_max_day" refers
to the abstract maximum daily temperature; there may be
many different MappedVariable objects associating temp_max_day
with (potentially differing) measured datasets, but they
all refer to the same conceptual variable

Dimensions represent the physical or abstract space
associated with variables

Coordinates represent the specific sections of dimensions
occupied by data; they are the dimensional equivalent of MappedVariable

'''

PY_INTMAX = 2**63 - 1

import numpy as np
from . import datetools as dt
from . import extents as extents
from .metatypes import ObjectContainer
from copy import deepcopy
from awrams.utils.helpers import print_error
from awrams.utils.general import Indexer

class CellMethod:
    '''
    Represent a calculation over some dimension(s)
    applying to a given data point
    '''
    def __init__(self,operation,dimension):
        self.operation = operation
        self.dimension = dimension

class Units:
    '''
    Object representing a specific unit of measurement
    '''
    def __init__(self,name):
        if isinstance(name,Units):
            name = name.name            
        self.name = name

    def __eq__(self,other):
        return self.name == other.name

    def __ne__(self,other):
        return self.name != other.name

    def __repr__(self):
        return self.name

class Dimension:
    def __init__(self,name,units,dtype):
        '''
        Represent a single dimension of some data
        name: str, units: Units object, dtype: datatype
        '''
        self.name = name
        self.units = Units(units)
        self.dtype = dtype

    def __repr__(self):
        return "%s (%s)" % (self.name, self.units)

    def __eq__(self,other):
        return (self.name == other.name) and (self.units == other.units)

    def __ne__(self,other):
        return not self==other

    def _dump_attrs(self):
        attrs = {}
        attrs['units'] = self.units.name
        attrs['name'] = self.name
        attrs['standard_name'] = self.name
        attrs['long_name'] = self.name
        return attrs

class Coordinates:
    '''
    Index is a container of coords in unit space
    e.g [100,102,105] maps the unit values 100,102,105 to the data in 0,1,2
    '''
    def __init__(self,dimension,index,unlimited = False):
        self.dimension = dimension
        self.index = index
        self._index_map = None
        self.unlimited = unlimited

        self.icoords = Indexer(self._get_subset_coords)
        self.iindex = Indexer(self._get_subset_index)

    def __repr__(self):
        return "%s :\n%s" % (self.dimension, self.index)

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return self.index.__iter__()

    def __getitem__(self,idx):
        return self.index[idx]

    def __eq__(self,other):
        return ((self.dimension == other.dimension) and (self.index == other.index).all())

    def __ne__(self,other):
        return not self==other

    def _persist_index(self):
        return self.index

    def _map_index(self):
        self._index_map = {}
        for i, i_val in enumerate(self.index):
            self._index_map[i_val] = i
        self._index_map[None] = None

    def _get_subset_coords(self,subcoords):
        return Coordinates(self.dimension,self.index[self.get_index(subcoords)],self.unlimited)

    def _get_subset_index(self,subindex):
        return Coordinates(self.dimension,self.index[subindex],self.unlimited)

    def get_index(self,coords):
        '''
        Return the indices in local dataspace for a given set of coords,
        expressed as either a slice, or a single member of the coord dtype
        +++
        Assumes that coords are contiguous with the same step value
        '''
        #+++
        # Some specific time indexing that should definitely be moved to TimeCoordinates
        # Also assumes daily freq only...
        if self._index_map is None:
            self._map_index()

        try:
            if type(coords) == slice:
                if coords.start in [None, 0]:
                    start_i = 0
                else:
                    try:
                        start_i = self._index_map[coords.start]
                    except KeyError:
                        start_i = self._index_map[coords.start.to_period(freq='D')]
                if coords.stop in [None,PY_INTMAX]:
                    stop_i = len(self.index)
                else:
                    try:
                        stop_i = self._index_map[coords.stop] + 1
                    except KeyError:
                        stop_i = self._index_map[coords.stop.to_period(freq='D')] + 1
                return slice(start_i,stop_i)

            if hasattr(coords,'__len__'):
                if len(coords) == 1:
                    return self._index_map[coords[0]]
                else:
                    if ((self._index_map[coords[-1]]-self._index_map[coords[0]]) + 1) == len(coords):
                        return slice(self._index_map[coords[0]],self._index_map[coords[-1]] + 1)
                    else:
                        return [self._index_map[c] for c in coords]
            else:
                return self._index_map[coords]
        except KeyError as e:
            print_error("%s out of range for %s" % (e,self.dimension))
            raise

class TimeCoordinates(Coordinates):
    '''
    Special case of coordinates that uses Python datetime objects internally,
    but persists to ordinals
    '''
    def __init__(self,dimension,index,unlimited=True):
        Coordinates.__init__(self,dimension,index,unlimited)

    def _infer_freq(self):
        if len(self.index) > 1:
            import pandas as pd
            self.index = pd.DatetimeIndex(self.index,freq=infer_freq(self.index))

    def _persist_index(self):
        dt_idx = self.index.to_datetime()
        self._infer_freq()
        _,eop,_ = dt.boundary_funcs(self.index.freqstr)

        out_idx_pit = []
        for ts in dt_idx:
            out_idx_pit.append(eop(ts))

        epoch_ordinal = self.dimension.epoch.toordinal()
        return [ts.toordinal() - epoch_ordinal for ts in out_idx_pit]


class BoundedCoordinates:
    '''
    Mixin for Coordinates objects with bounded ranges
    boundaries are expressed inclusively on the Python side, but persist
    to rhs-exclusive (CF conventions)
    '''
    def __init__(self,boundaries):
        self.boundaries = boundaries

    def _validate_bounds(self):
        i_c, b_c = len(self.index), len(self.boundaries)
        if i_c != b_c:
            raise Exception("Length mismatch for index and boundaries (%s, %s)" % (i_c,b_c))

    def __repr__(self):
        return "%s :\n%s (%s - %s) ... %s (%s - %s)" % (self.dimension, self.index[0],
            self.boundaries[0][0],self.boundaries[0][1], self.index[-1],
            self.boundaries[-1][0],self.boundaries[-1][1])

    def _persist_boundaries(self):
        raise Exception("Not implemented")

class BoundedTimeCoordinates(TimeCoordinates,BoundedCoordinates):
    '''
    Time coordinates representing ranges of time
    By convention, index is the _last day_ of the period being represented
    '''
    def __init__(self,dimension,index,boundaries):
        TimeCoordinates.__init__(self,dimension,index)
        BoundedCoordinates.__init__(self,boundaries)
        self._validate_bounds()

    def _persist_index(self):
        #self._infer_freq()
        out_idx = [p.end_time for p in self.index]
        epoch_ordinal = self.dimension.epoch.toordinal()
        return [ts.toordinal() - epoch_ordinal for ts in out_idx]

    def _persist_boundaries(self):
        epoch_ordinal = self.dimension.epoch.toordinal()
        bounds = [([b[0].toordinal() - epoch_ordinal, (b[1].toordinal() - epoch_ordinal)+1]) for b in self.boundaries]
        return bounds

class CoordinateSet(ObjectContainer):
    '''
    A grouped set of coordinates
    '''
    def __init__(self,coordinates):
        ObjectContainer.__init__(self)
        self.dimensions = ObjectContainer()

        for coord in coordinates:
            self[coord.dimension.name] = coord
            self.dimensions[coord.dimension.name] = coord.dimension

        self._set_shape()

    def _set_shape(self):
        shape = [(len(x)) for x in self]
        self.shape = tuple(shape)

    def update_coord(self,coord,in_place=True):
        if in_place:
            cs = self
        else:
            cs = CoordinateSet(self)
        cs[coord.dimension.name] = coord
        cs._set_shape()
        return cs

    def get_index(self,coords): #+++ Testing for missing coords
        if type(coords) == CoordinateSet:
            out = []
            for k,v in self.items():
                if k in coords:
                    out.append(v.get_index(coords[k].index))
                else:
                    out.append(v.get_index(slice(None)))
            return tuple(out)
        elif hasattr(coords,'__len__'):
            out = []
            for c,c_data in zip(self,coords):
                out.append(c.get_index(c_data))
            return tuple(out)
        else:
            return self.time.get_index(coords)

    def _DEP_get_index(self,coords): #+++ Replaced, wait for testing before deletion
        if type(coords) == CoordinateSet:
            out = []
            for c,c_data in zip(self,coords):
                out.append(c.get_index(c_data.index))
            return tuple(out)
        elif hasattr(coords,'__len__'):
            out = []
            for c,c_data in zip(self,coords):
                out.append(c.get_index(c_data))
            return tuple(out)
        else:
            return self.time.get_index(coords)

    def copy(self):
        return deepcopy(self)

    def validate_chunksizes(self,chunksizes,fixed_first=True):
        chunksizes_out = []
        for i, dim_sz in enumerate(self.shape):
            chunksizes_out.append(min(dim_sz,chunksizes[i]))
        if fixed_first:
            chunksizes_out[0] = chunksizes[0]
        return chunksizes_out

class TimeDimension(Dimension):
    def __init__(self,epoch,freq='d',calendar='gregorian'):
        units = gen_time_units_for_epoch(epoch,freq)
        Dimension.__init__(self,'time',units,np.int32)
        self.epoch = epoch
        self.freq = freq
        self.calendar = calendar

    def _dump_attrs(self):
        attrs = Dimension._dump_attrs(self)
        attrs['calendar'] = self.calendar
        return attrs

# Dict of attrs that should that are part of the netCDF format, but should not be considered metadata
NC_RESERVED_ATTRS = ['DIMENSION_LIST','CLASS','NAME','REFERENCE_LIST']

class Variable:
    '''
    Generalised notion of a variable; it has units
    and dimensions
    '''

    def __init__(self,name,units,meta=None):
        self.name = name
        self.units = units if isinstance(units,Units) else Units(units)
        self.meta = {} if meta is None else meta
        if 'Description' in self.meta:
            self.meta['long_name'] = self.meta['Description']

    def __repr__(self):
        return "%s (%s)" % (self.name, self.units)

    def _dump_attrs(self):
        attrs = {}

        avail = list(self.meta.keys())

        attrs['name'] = self.name
        attrs['units'] = self.units.name

        for k in ['name','units']:
            if k in avail:
                avail.remove(k)

        for k in ['long_name','standard_name']:
            v = self.meta.get(k)
            if v:
                attrs[k] = v
                avail.remove(k)
            else:
                attrs[k] = self.name
        
        # +++ Certain netCDF4 versions fail to store doubles correctly here
        # Force all to strings at this point...
        for k in avail:
            #attrs[k] = str(self.meta[k]) #+++
            attrs[k] = self.meta[k]
        return attrs

    @classmethod
    def from_ncvar(self,ncvar):
        attrs = dict([[k,ncvar.getncattr(k)] for k in ncvar.ncattrs() if k not in NC_RESERVED_ATTRS])
        try:
            return Variable(attrs['name'],attrs['units'],attrs)
        except KeyError:
            return Variable(ncvar.name,attrs['units'],attrs)


class GeoTemporalVariable(Variable):
    def __init__(self,name,units,meta=None):
        Variable.__init__(self,name,units,meta)

class MappedVariable:
    '''
    Represents a concrete block of data for a Variable;
    ie it has known coordinates, and a specific datatype
    '''
    def __init__(self,variable,coordinates,dtype,mode='r'):
        self.variable = variable
        self.coordinates = self._validate_coordinates(coordinates)
        self.dtype = dtype

    def _validate_coordinates(self,coordinates):
        if type(coordinates) != CoordinateSet:
            if isinstance(coordinates,Coordinates):
                coordinates = [coordinates]
            coordinates = CoordinateSet(coordinates)
        return coordinates

    def __validate_coordinates(self,coordinates):
        if len(coordinates) != len(self.variable.dimensions):
            raise Exception("%s coordinate sets passed for %s dimensions" % (len(coordinates), len(self.variable.dimensions)))
        for i, coord in enumerate(coordinates):
            if coord.dimension != self.variable.dimensions[i]:
                raise Exception("Coordinates for %s being applied to %s" % (coord.dimension, self.variable.dimensions[i]))
        return coordinates

    def __repr__(self):
        return "%s (%s)" % (self.variable, self.coordinates)

def gen_time_units_for_epoch(epoch,freq='DAILY'):
    tf = dt.validate_timeframe(freq)
    units = dt.units_for_tf[tf]
    return Units("%s since %s" % (units, epoch))

DIMENSIONS = {}

deg_north = Units("degrees_north")
deg_east = Units("degrees_east")
DIMENSIONS['latitude'] = latitude = Dimension('latitude',deg_north,np.float64)
DIMENSIONS['longitude'] = longitude = Dimension('longitude',deg_east,np.float64)
DIMENSIONS['time'] = awrams_time = TimeDimension(dt.date(1900,1,1))
DIMENSIONS['hypsometric_percentile'] = Dimension('hypsometric_percentile',Units('percentile'),np.float64)
mm = Units('mm')


def get_dimension(name,units=None,dtype=None):
    '''
    Helper function for supplying pre-existing dimension objects
    '''
    if name not in DIMENSIONS:
        DIMENSIONS[name] = Dimension(name,units,dtype)
    
    return DIMENSIONS[name]

def latlon_to_coords(lats,lons):
    return CoordinateSet([Coordinates(latitude,lats), Coordinates(longitude,lons)])

def gen_coordset(period,extent):
    '''
    Generate a set of GeoTemporal (time,lat,lon) coordinates for a period/extent
    '''
    tc = period_to_tc(period)
    sc = extent.to_coords()
    return CoordinateSet([tc,sc[0],sc[1]])

def period_to_tc(p_idx):
    '''
    Return a set of (bounded) coordinates for a pandas period_idx
    '''
    import pandas as pd # +++ Localise import
    if isinstance(p_idx,pd.PeriodIndex):
        boundaries = []
        for p in p_idx:
            boundaries.append([p.start_time,p.end_time])
        return BoundedTimeCoordinates(awrams_time,p_idx,boundaries)
    else:
        #+++ Just hope this captures other cases...
        return TimeCoordinates(awrams_time,p_idx)

def infer_freq(index):
    '''
    Attempt to infer a frequency from a pair of dates
    (pandas requires 3 dates, we often have only 2!)
    '''
    if index[1]-index[0] == dt.days(1):
        return('d')
    import pandas as pd # +++ Localise import
    if type(index) is pd.DatetimeIndex:
        if index[1]-index[0] < dt.days(32):
            return('M')
        elif index[1]-index[0] > dt.days(364):
            return('A')

    for freq in ['M','A']:
        sop,eop,_ = dt.boundary_funcs(freq)
        if (index[0]==eop(index[0]) and index[1]==eop(index[1])):
            return freq
        if (index[0]==sop(index[0]) and index[1]==sop(index[1])):
            return freq
    raise Exception("Unable to infer frequency")

def infer_period(boundaries):
    '''
    Given a set of (inclusive) boundaries, infer the frequency
    of the underlying period
    '''
    b = boundaries[0]
    td = b[1]-b[0]
    days = td.days + 1

    if days == 1:
        return 'd'
    elif days in [28,29,30,31,90,91,92]:
        return 'm'
    elif days in [365,366]:
        return 'A-'+dt.name_of_month[b[1].month]

    raise Exception("Unable to infer frequency")


