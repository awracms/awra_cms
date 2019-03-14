import json
import numpy as np
import datetime
import dateutil
import pandas as pd
from awrams.utils import mapping_types as mt
from awrams.utils.messaging.buffer_group import DataSpec
from awrams.utils.io.data_mapping import SplitFileManager,AnnualSplitSchema,FlatFileSchema,managed_dataset
from awrams.utils.mapping_types import CoordinateSet, Coordinates,get_dimension,TimeCoordinates
from awrams.utils.extents import get_default_extent, from_latlons
from awrams.utils.awrams_log import get_module_logger
from awrams.utils.metatypes import ObjectDict as odict

logger = get_module_logger('nodes')

def callable_to_funcspec(c):
    if str(c.__class__) == "<class 'module'>":
        func_spec = dict(objtype='module',module=c.__name__)
    else:    
        func_spec = dict(name=c.__name__,objtype='class')
        if hasattr(c,'__module__'):
            func_spec['module'] = c.__module__
        else:
            func_spec['module'] = c.__class__.__module__
    return func_spec

def funcspec_to_callable(func_spec):
    import importlib
    m = importlib.import_module(func_spec['module'])
    if func_spec['objtype'] == 'class':
        return getattr(m,func_spec['name'])
    else:
        return m

class GraphNode:
    def __init__(self,node_type,executor,out_type,inputs=None,properties=None,args=None):
        self.node_type = node_type
        
        if inputs is None:
            inputs = []
        if properties is None:
            properties = {}
        if args is None:
            args = {}
            
        self.inputs = inputs
        self.out_type = out_type
        self.properties = properties
        self.args = args

        if callable(executor):
            executor = callable_to_funcspec(executor)

        self.executor = executor
    
    def get_executor(self):
        return funcspec_to_callable(self.executor)(**self.args)

    def to_dict(self):
        return {
            'node_type': self.node_type,
            'properties': self.properties,
            'inputs': self.inputs,
            'out_type': self.out_type,
            'args': self.args,
            'executor': self.executor
        }
        
    def __repr__(self):
        return '%s(%s):%s' % (self.node_type,self.inputs,self.args)

class GraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GraphNode):
            return dict(obj_type='GraphNode', data = {
                'node_type': obj.node_type,
                'properties': obj.properties,
                'inputs': obj.inputs,
                'out_type': obj.out_type,
                'args': obj.args,
                'executor': obj.executor
            })
        elif isinstance(obj,np.ndarray):
            return dict(obj_type='ndarray', data = {
                    'data': obj.tolist(),
                    'dtype': obj.dtype.str
            })
        elif isinstance(obj,datetime.datetime):
            return dict(obj_type='datetime', date = {
                    'data': obj.isoformat()
                })
        else:
            return json.JSONEncoder.default(self, obj)
        

class GraphDecoder(json.JSONDecoder):   
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        
    def object_hook(self,obj):
        if 'obj_type' in obj:
            obj_type = obj['obj_type']
            if obj_type=='GraphNode':
                return GraphNode(**obj['data'])
            elif obj_type=='ndarray':
                return np.ndarray(obj['data']['data'],dtype=obj['data']['dtype'])
            elif obj_type=='datetime':
                return dateutil.parser.parse(obj['data'])
            else:
                raise Exception("Unknown type", obj_type)
        else:
            return obj
            
def dump_mapping(mapping,filename):
    with open(filename,'w') as fh:
        json.dump(mapping,fh,cls=GraphEncoder)
        
def load_mapping(filename):
    with open(filename,'r') as fh:
        return json.load(fh,cls=GraphDecoder)
    
'''
Predefined node types and convenience definitions
+++ TODO   flattened/expanded are utility functions - move into support?
'''

def get_flattened(data,mask):
    if mask.any():
        return data[...,~mask]
    else:
        return data.reshape(list(data.shape[:data.ndim-2])+[mask.shape[0]*mask.shape[1]])

def get_expanded(data,mask,fill_value=np.nan,as_ma=True):
    '''
    Expand a flattened arrays into a spatial
    data : input array
    mask : mask for output array
    '''
    outshape = list(data.shape[:data.ndim-1])+[mask.shape[0],mask.shape[1]]

    all_false = False

    if data.shape[-1] == mask.shape[0]*mask.shape[1]:
        out = data.reshape(outshape)
        all_false = True
    else:    
        out = np.empty(outshape,dtype=data.dtype)
        if fill_value is not None:
            out.fill(fill_value)
        out[...,~mask] = data

    if as_ma:
        out = np.ma.MaskedArray(out,fill_value=fill_value)
        if all_false:
            out_mask = False
        else:
            out.mask = mask        
    
    return out

def expand_dict(data_dict,mask,fill_value=np.nan,as_ma=True):
    '''
    Expand a dictionary of flattened arrays into spatial arrays
    <see get_expanded>
    '''
    out_dict = odict()
    for k,v in data_dict.items():
        try:
            if isinstance(v,np.ndarray):
                out_dict[k] = get_expanded(v,mask,fill_value,as_ma)
            elif isinstance(v,dict):
                out_dict[k] = expand_dict(v,mask,fill_value,as_ma)
        except Exception as e:
            logger.critical('Failed expansion of %s' % k)
            raise e

    return out_dict

class InputNode:
    def get_data(self,coords):
        raise NotImplementedError

    def get_coords(self):
        raise NotImplementedError

    def get_extent(self):
        raise NotImplementedError

    def get_data_flat(self,coords,mask):
        return get_flattened(self.get_data(coords),mask)

    def get_data_multiflat(self,coords,masks):
        '''
        ::coords:: iterable of coords
        ::masks:: iterable of masks
        Return a flattened array of the joined outputs of each set of coords and mask supplied
        '''

        data = []

        for i in range(len(coords)):
            c,m = coords[i],masks[i]
            data.append(self.get_data_flat(c,m))

        return np.concatenate(data,axis=-1)

    def get_dataspec(self):
        raise NotImplementedError

    def get_dataspec_flat(self):
        dspec = self.get_dataspec()
        if dspec.valtype == 'array':
            if list(dspec.dims[-2:]) == ['latitude','longitude']:
                dspec.dims = list(dspec.dims[:-2]) + ['cell']
        return dspec

class ForcingNode(InputNode):
    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        return DataSpec('array',dims,self.dtype)

class ProcessNode:
    def process(self,inputs):
        raise NotImplementedError

    def get_dataspec(self,in_specs):
        return promote_dataspecs(in_specs)

class _ExecutionNode:
    def __init__(self):
        pass
    
    def get_data(self,coords,inputs):
        raise NotImplementedError

    def get_data_flat(self,coords,inputs,mask):
        return get_flattened(self.get_data(coords,inputs),mask)

def promote_dataspecs(in_specs):
    curvtype = 'scalar'
    curdtype = None
    dims = []
    for ispec in in_specs:
        if ispec.valtype == 'array':
            curvtype = 'array'
            comp = np.ndarray(shape=(1,),dtype=ispec.dtype)

            #+++ Doesn't actually check if compatible, just assumes
            # we can broadcast to larger arrays....
            if len(ispec.dims) > len(dims):
                dims = ispec.dims
        else:
            comp = np.ndarray(shape=(1),dtype=ispec.dtype)[0]
        if curdtype is None:
            curdtype = np.result_type(comp)
        else:
            curdtype = np.result_type(curdtype,comp)

    return DataSpec(curvtype,dims,curdtype)

class TransformNode(ProcessNode):
    def __init__(self,tfunc,func_args):
        self.tfunc = funcspec_to_callable(tfunc)
        self.func_args = func_args
                 
    def process(self,inputs):
        return self.tfunc(*inputs,**self.func_args)


class ConstNode(InputNode):
    def __init__(self,value):
        self.value = value
        
    def get_data(self,coords):
        return self.value

    def get_data_flat(self,coords,mask):
        return self.value

    def get_data_multiflat(self,coords,masks):
        return self.value

    def get_dataspec(self):
        return DataSpec('scalar',[],type(self.value))

class ConstArray(InputNode):
    def __init__(self,value,dims):
        self.value = value
        self.dims = dims
        
    def get_data(self,coords):
        return self.value

    def get_data_flat(self,coords,mask):
        return self.value

    def get_data_multiflat(self,coords,masks):
        return self.value

    def get_dataspec(self):
        return DataSpec('array',self.dims,self.value.dtype)

class ParameterNode(ConstNode):
    def __init__(self, value, min_val, max_val, fixed=False, description=''):
        """Parameter defined by user supplied constants

        Args:
            value (double): (Default) value of parameter
            min (double): Minimum value
            max (double): Max value
            fixed (bool, optional): If false, parameter will be included in calibration
            description (str, optional): Full description of parameter
        """
        self.value = value
        self.min_val = min_val
        self.max_val = max_val      
        self.fixed = fixed
        self.description = description


class WiradaJSONParameterNode(ParameterNode):
    def __init__(self, filename, key, min_val, max_val, fixed):
        """Parameter supplied from WIRADA format JSON file

        Args:
            filename (str): Full path to input JSON file
            key (str): Parameter key
            fixed (bool, optional): If supplied, overrides value of fixed in JSON 

        Raises:
            Exception: Raises if entry cannot be located in supplied filename
        """
        with open(filename, 'r') as fh:
            # +++ Consider implementing a caching mechanism to avoid loading data multiple times
            dparams = json.load(fh)
            found = False

            while not found:
                for entry in dparams:
                    param_lower = entry['MemberName'].lower()
                    if param_lower == key:
                        super().__init__(entry['Value'], min_val, max_val,
                                         fixed, entry['DisplayName'])
                        found = True
            if not found:
                raise Exception("No entry found for %s in %s" % (key, filename))

class CalibrationResultsParameterNode(ParameterNode):
    def __init__(self, filename, key):
        from awrams.calibration.support import CalibrationResults
        res = CalibrationResults(filename)
        bestp = res.get_best_paramset()
        param = bestp.loc[key]
        super().__init__(**param)

class AssignNode(ProcessNode):

    def process(self,inputs):
        return inputs[0]

    def get_dataspec(self,in_specs):
        return in_specs[0]

class StaticDataNode(InputNode):
    '''
    Used primarily for caching results of a previous graph run
    <flat> should be set if data is preflattened
    '''
    def __init__(self,data,dataspec,flat=True):
        self.data = data
        self.dspec = dataspec
        self.flat = flat

    def get_data(self,coords):
        if self.flat:
            raise Exception("Attempting to return unflattened data from flat static node")
        else:
            return self.data

    def get_data_flat(self,coords,mask):
        if self.flat:
            return self.data
        else:
            return get_flattened(self.data,mask)
    
    def get_dataspec(self):
        return self.dspec

class StaticForcing(InputNode):
    def __init__(self,data_map,nc_var,dims=['time','latitude','longitude']):
        self.data_map = data_map
        self.nc_var = nc_var
        self.dims = dims

    def get_data_prepack(self,cid):
        return self.data_map[cid]

    def get_dataspec(self):
        return DataSpec('array',self.dims,np.float64)

class SplitFileReaderNode(InputNode):
    def __init__(self,path,pattern,nc_var,cache=False,map_func=None):
        from awrams.utils.io.data_mapping import SplitFileManager

        if map_func is not None:
            map_func = funcspec_to_callable(map_func)

        self.sfm = SplitFileManager.open_existing(path,pattern,nc_var,map_func=map_func)

        self.cache = cache
        self.data = None

    def get_data(self,coords):
        if self.cache:
            if self.data is None:
                d = self.sfm.get_padded_by_coords(self.get_coords())
                self.data = np.where(d==self.sfm.fillvalue,np.nan,d)

            cs = self.get_coords()
            sub_idx = cs.get_index(coords)
            return self.data[sub_idx]

        d = self.sfm.get_padded_by_coords(coords)
        return np.where(d==self.sfm.fillvalue,np.nan,d)

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = self.sfm.mapped_var.dtype
        return DataSpec('array',dims,dtype)

    def get_coords(self):
        return self.sfm.cs

    def get_extent(self):
        return self.sfm.get_extent()

class StateFileReaderNode(SplitFileReaderNode):
    def __init__(self,path,pattern,nc_var,date=None):
        super().__init__(path,pattern,nc_var)
        self.date = date
        if date is not None:
            p_idx = pd.DatetimeIndex([date])
            self._tc = mt.TimeCoordinates(mt.awrams_time,p_idx)
    
    def get_data(self,coords):
        from copy import deepcopy

        if self.date is None:
            tc = deepcopy(coords['time'])
            tc.index = [tc.index[0] - 1]
        else:
            tc = self._tc
        coords = CoordinateSet([tc,coords['latitude'],coords['longitude']])
        d = self.sfm.get_padded_by_coords(coords)[0,:]
        return np.where(d==self.sfm.fillvalue,np.nan,d)

class InitStateFromArray(InputNode):
    def __init__(self,data,extent):
        self.data = data

        self.extent = extent
        self.coords = extent.to_coords()

        if (len(self.coords[0]),len(self.coords[1])) != data.shape:
            Exception("data array shape does not match extent shape")

    def get_data(self,coords):
        lat_idx = self.coords[0].get_index(coords[1])
        lon_idx = self.coords[1].get_index(coords[2])
        if len(self.data.shape) == 1:
            return self.data
        return self.data[lat_idx,lon_idx]

    def get_dataspec(self):
        return DataSpec('array',['latitude','longitude'],np.float64)

    def get_coords(self):
        return self.coords

    def get_extent(self):
        return self.extent

class GapFiller(ProcessNode):
    def process(self,inputs):
        base_data = inputs[0]
        fill_data = inputs[1]
        gaps = np.isnan(base_data)
        if gaps.any(): ### there are gaps
            if gaps.all():
                base_data = fill_data
            else:
                # d[gaps] = fill[gaps]
                base_data = np.where(gaps,fill_data,base_data)
        return base_data      

class ForcingGapFiller(InputNode):
    """
    # +++
    # Absolutely should be refactored to separate reading from filling
    # What if the source were not files? (eg DAP etc)
    # At the very least, allow class specs as options rather than hardcoding
    # See: ClimatologyNode for discussion of other solutions
    """
    
    def __init__(self,path,pattern,nc_var,filler_fn):
        self.nc_var = nc_var
        self.force = SplitFileReaderNode(path,pattern,nc_var)
        self.filler = ClimatologyNode(filler_fn,nc_var)

    def get_data_flat(self,coords,mask):
        d = self.force.get_data_flat(coords,mask)
        gaps = np.isnan(d)
        if gaps.any(): ### there are gaps
            fill = self.filler.get_data_flat(coords,mask)
            if gaps.all():
                d = fill
            else:
                # d[gaps] = fill[gaps]
                d = np.where(gaps,fill,d)
        return d

    def get_data(self,coords):
        d = self.force.get_data(coords)
        gaps = np.isnan(d)
        if gaps.any(): ### there are gaps
            fill = self.filler.get_data(coords)
            if gaps.all():
                d = fill
            else:
                # d[gaps] = fill[gaps]
                d = np.where(gaps,fill,d)
        return d

    def get_dataspec(self):
        return self.force.get_dataspec()

    def get_coords(self):
        return self.force.get_coords()

    def get_extent(self):
        return self.force.get_extent()

class ClimatologyNode(InputNode):
    """
    +++ Currently always loads all data during get_data(), even if target input contains no gaps
        Solution 1 : Provide a map of known gaps (even if it's as simple as 'no data for these years')
        Solution 2 : Allow InputNodes to receive inputs (ie function like ProcessNodes), then do 
                     conditional reads based on gappiness
        Solution 3 : Allow the gapfiller node to wrap a reader class.  Problematic in that it hides which nodes are performing IO

        S1 is definitely the lower hanging fruit (ie doesn't require refactoring anything outside this class)

        Some combination of all of the above would be 'ideal' but will need to see whether it's worth the effort...
    """
    def __init__(self,filename,variable,preload=False):
        import netCDF4 as nc
        self.fh = nc.Dataset(filename,'r')
        lats,lons = self.fh.variables['latitude'][...], self.fh.variables['longitude'][...]
        self.cs = mt.CoordinateSet(mt.latlon_to_coords(lats,lons))
        self.extent = from_latlons(lats,lons)
        if preload:
            self.v = self.fh.variables[variable][...]
        else:
            self.v = self.fh.variables[variable]

    def get_data(self,coords):
        ### convert time to doy (dont forget 29 feb)
        ### accumulate yearwise
        data = []
        for y in np.unique(coords[0].index.year):
            t = [datetime.datetime(2000,ts.month,ts.day).timetuple().tm_yday - 1 for ts in coords[0][coords[0].index.year == y]]
            idx = self.cs.get_index([coords.latitude,coords.longitude])
            idx = (t,idx[0],idx[1])
            data.append(self.v[idx])

        data = np.concatenate(data).reshape(coords.shape)
        return data

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = self.v.dtype
        return DataSpec('array',dims,dtype)

    def get_coords(self):
        return self.cs

    def get_extent(self):
        return self.extent

class ModisForcingNode(InputNode):
    '''
    expects 8day ncf for modis data, picks the day closest to requested day
    '''
    def __init__(self,path,pattern,nc_var,preload=False):
        import netCDF4 as nc
        self.fh = nc.Dataset(path+pattern,'r')
        self.v = self.fh.variables[nc_var]
        self.time_idx = self.fh.variables['time'][:]
        lats,lons = self.fh.variables['latitude'][...], self.fh.variables['longitude'][...]
        self.cs = mt.CoordinateSet(mt.latlon_to_coords(lats,lons))

    def get_data(self,coords):
        from datetime import datetime

        c_idx = self.cs.get_index([coords.latitude,coords.longitude])

        data = []
        for d in np.unique(coords[0]):

            d_idx = d.to_pydatetime().toordinal() - datetime(1900,1,1).toordinal()

            t_idx = (np.abs(self.time_idx - d_idx)).argmin()

            idx = (t_idx,c_idx[0],c_idx[1])
            data.append(self.v[idx])

        data = np.concatenate(data).reshape(coords.shape)
        return data

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = np.float32 #self.v.dtype
        return DataSpec('array',dims,dtype)

    def get_coords(self):
        return self.cs

    def get_extent(self):
        return self.extent

class MonthlyClimatologyNode(InputNode):
    def __init__(self,filename,variable,preload=False):
        #import netCDF4 as nc
        self.fh = managed_dataset(filename,'r')
        self.extent = self.fh.get_extent()
        self.cs = self.extent.to_coords()
        if preload:
            self.v = self.fh.variables[variable][...]
        else:
            self.v = self.fh.variables[variable]

    def get_data(self,coords):
        ### convert time to doy (dont forget 29 feb)
        ### accumulate yearwise
        idx = self.cs.get_index([coords.latitude,coords.longitude])
        tdata = self.v[:,idx[0],idx[1]]

        full_m_idx = coords.time.index.month - 1

        return tdata[full_m_idx].reshape(coords.shape)

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = self.v.dtype
        return DataSpec('array',dims,dtype)

    def get_coords(self):
        return self.cs

    def get_extent(self):
        return self.extent


class _SpatialHDF5Node(InputNode):
    def __init__(self,filename,variable,preload=False):
        self.filename = filename
        self.variable = variable
        self.preload = preload

        import h5py
        self.fh = h5py.File(filename,'r')

        lats,lons = self.fh['dimensions/latitude'][...], self.fh['dimensions/longitude'][...]
        self.cs = mt.CoordinateSet(mt.latlon_to_coords(lats,lons))
        if preload:
            self.v = self.fh[variable][...]
        else:
            self.v = self.fh[variable]
        self.dtype = self.v.dtype

    def close(self):
        self.fh.close()

    def get_data(self,coords):
        idx = self.cs.get_index([coords.latitude,coords.longitude])
        return self.v[idx].reshape(coords.shape[1:])

    def get_dataspec(self):
        return DataSpec('array',['latitude','longitude'],self.dtype)

class _HypsoHDF5Node(_SpatialHDF5Node):
    def __init__(self,filename,variable,preload=False):
        SpatialHDF5Node.__init__(self,filename,variable,preload)

    def get_data(self,coords):
        idx = self.cs.get_index([coords.latitude,coords.longitude])
        data = self.v[:,idx[0],idx[1]]
        return data.reshape((data.shape[0],coords.shape[1],coords.shape[2]))

    def get_dataspec(self):
        return DataSpec('array',['hypsometric_percentile','latitude','longitude'],self.dtype)

class ConstHDF5Node(InputNode):
    def __init__(self,filename,variable,dims):
        import h5py
        fh = h5py.File(filename,'r')
        if len(dims) != len(fh[variable].shape):
            raise Exception("Dimension list does not match shape of data")
        self.value = fh[variable][...]
        self.dims = dims
        fh.close()

    def get_data(self,coords):
        return self.value

    def get_data_flat(self,coords,mask):
        return self.value

    def get_dataspec(self):
        return DataSpec('array',self.dims,self.value.dtype)

class OutputNode(InputNode):
    def __init__(self,var_name,dtype=np.float64):
        self.var_name = var_name
        self.dtype = dtype

    def set_data(self, data):
        self.data = data

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        return DataSpec('array',dims,self.dtype)

class WriterNode:
    def get_dataspec(self):
        raise NotImplementedError

    def get_dataspec_flat(self):
        dspec = self.get_dataspec()
        if dspec.valtype == 'array':
            if list(dspec.dims[-2:]) == ['latitude','longitude']:
                dspec.dims = list(dspec.dims[:-2]) + ['cell']
        return dspec

class FileWriterNode(WriterNode):
    def __init__(self,path,nc_var,mode='r+',dtype=np.float64,chunksizes=None,ncparams = None):
        self.fm = None #SplitFileManager.open_existing(path,pattern,nc_var)

        self.path = path
        self.nc_var = nc_var
        self._file_mode = mode
        self.dtype = dtype
        self.schema = AnnualSplitSchema
        self._init = True
        self._open = True
        self.ncparams = ncparams
        self.chunksizes = chunksizes
        self.dtype = dtype

    def close(self):
        self.fm.close_all()

    def open_files(self): #period,extent):
        if self._open:
            # self.fm = self.file_manager_type.open_existing(self.path,self.nc_var+'.nc',self.nc_var,mode='r+')
            # self.fm = SplitFileManager.open_existing(self.path,self.nc_var+'*.nc',self.nc_var,mode='r+')
            self.fm = SplitFileManager.open_existing(self.path,self.nc_var+'.nc',self.nc_var,mode='r+')
            self._open = False

    def init_files(self,period,extent):
        # if self._init:
        v = mt.Variable(self.nc_var,'mm')
        cs = mt.gen_coordset(period,extent)

        mvar = mt.MappedVariable(v,cs,self.dtype) #in_dtype)

        self.fm = SplitFileManager(self.path,mvar)

        clobber = self._file_mode == 'w'
        self.fm.create_files(self.schema,False,clobber,chunksize=self.chunksizes,ncparams=self.ncparams)#,chunksize=(1,16,16)) #None) #(64,32,32)) #(1,16,16))

        self._open = True

    def write_data(self,coords,data):
        data = data.filled(-999.)
        self.fm.set_by_coords(coords,data)

    def sync_all(self):
        for f in self.fm._open_datasets.values():
            f.ncd_group.sync()

    def process(self,coords,data):
        self.open_files()
        self.write_data(coords,data)

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = self.fm.mapped_var.dtype
        return DataSpec('array',dims,dtype)

class SplitFileWriterNode(FileWriterNode):

    def __init__(self,path,nc_var,mode='r+',dtype=np.float64,chunksizes=None,ncparams=None):
        super().__init__(path,nc_var,mode,dtype,chunksizes,ncparams)

    def open_files(self): #period,extent):
        if self._open:
            self.fm = SplitFileManager.open_existing(self.path,self.nc_var+'_[0-9][0-9][0-9][0-9].nc',self.nc_var,mode='r+')
            self._open = False

class FlatFileWriterNode(FileWriterNode):
    def __init__(self,path,nc_var,mode='w',freq='M',dtype=np.float64,chunksizes=None,ncparams=None):
        super().__init__(path,nc_var,mode,dtype,chunksizes,ncparams)
        self.freq = freq.upper()
        if self.freq not in ['D','M','A']:
            raise Exception("unrecognised period freq")
        self.schema = FlatFileSchema

    def init_files(self,period,extent):
        import pandas as pd
        if self.freq == 'D':
            p = period
        elif self.freq == 'M':   ### pull out the whole month ends
            p = pd.DatetimeIndex([d for d in period if d == d - pd.offsets.Day() + pd.offsets.MonthEnd()])
        elif self.freq == 'A':   ### pull out the whole year ends
            p = pd.date_range(period[0] - pd.offsets.YearBegin(),
                                 period[-1],
                                 freq='A')
        super().init_files(p,extent)

    def write_data(self,coords,data):
        if self.freq == 'D':
            new_coords = coords
            idx = np.ones(len(coords['time']),dtype=np.bool)
        elif self.freq == 'M':
            idx = coords['time'].index.is_month_end
            if not idx.any(): ### no month ends
                return
            t = Coordinates(coords['time'].dimension,coords['time'][idx])
            new_coords = CoordinateSet((t,coords['latitude'],coords['longitude']))
        elif self.freq == 'A':
            idx = coords['time'].index.is_year_end
            if not idx.any(): ### no month ends
                return
            t = Coordinates(coords['time'].dimension,coords['time'][idx])
            new_coords = CoordinateSet((t,coords['latitude'],coords['longitude']))
        else:
            raise Exception("unrecognised period freq")

        data = data[idx,:].filled(-999.)
        self.fm.set_by_coords(new_coords,data)

class TimeOffsetReaderNode(InputNode):
    def __init__(self,child,offset):
        self.child = GraphNode(**child).get_executor()
        self.offset = offset

    def get_data(self,coords):
        cs = coords.update_coord(mt.period_to_tc(coords.time.index + self.offset),False)
        return self.child.get_data(cs)

    def get_dataspec(self):
        return self.child.get_dataspec()


def transform(tfunc,inputs,func_args=None):
    if isinstance(inputs,str):
        inputs = [inputs]
    if func_args is None:
        func_args = {}
    if callable(tfunc):
        tfunc = callable_to_funcspec(tfunc)
    elif isinstance(tfunc,str):
        module,name = tfunc.rsplit('.',1)
        tfunc = dict(module=module,name=name)
    if not (isinstance(tfunc,dict) and 'name' in tfunc and 'module' in tfunc):
        raise Exception("Invalid transformation function")
    return GraphNode('transform',callable_to_funcspec(TransformNode),'from_inputs',inputs,args=odict({'tfunc': tfunc, 'func_args': func_args}))

def _arithmetic(lhs,rhs,op):
    return transform(np.__getattribute__(op),[lhs,rhs])    

def _mul3(a,b,c):
    return a*b*c

def _mul4(a,b,c,d):
    return a*b*c*d

def _add3(a,b,c):
    return a+b+c

def _add4(a,b,c,d):
    return a+b+c+d

def _mix(lhs,rhs,ratio):
    return (1.0-ratio)*lhs + ratio*rhs

_muls = {2:np.multiply,3:_mul3,4:_mul4}
_adds = {2:np.add,3:_add3,4:_add4}

def add(*args):
    return transform(_adds[len(args)],args)  

def sub(lhs,rhs):
    return transform(np.subtract,[lhs,rhs]) 

def mul(*args):
    return transform(_muls[len(args)],args)  

def div(lhs,rhs):
    return transform(np.divide,[lhs,rhs]) 

def mix(lhs,rhs,ratio):
    return transform(_mix,[lhs,rhs],odict(ratio=ratio))

def isnan(a):
    return transform(np.isnan,[a])

def _where(a,b):
    # return np.where(np.isnan(a),b,a)
    idx = np.isnan(a)
    a[idx] = b[idx]
    return a

def where(a,b):
    return transform(_where,[a,b])

def average(a,b): #+++ Only used by HRU adder, which should be weighted anyway
    return (a + b) / 2.

def forcing_from_dict(data_map,nc_var,dims):
    return GraphNode('forcing_from_dict',callable_to_funcspec(StaticForcing),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=odict(data_map=data_map,nc_var=nc_var,dims=dims))

def forcing_from_ncfiles(path,pattern,nc_var,cache=False,map_func=None):
    if map_func is not None:
        map_func = callable_to_funcspec(map_func)
    return GraphNode('forcing_from_ncfiles',callable_to_funcspec(SplitFileReaderNode),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=odict(path=path,pattern=pattern,nc_var=nc_var,cache=cache,map_func=map_func))

def forcing_from_modis_ncfiles(path,pattern,nc_var,cache=False):
    return GraphNode('forcing_from_modis_ncfiles',callable_to_funcspec(ModisForcingNode),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=odict(path=path,pattern=pattern,nc_var=nc_var,preload=cache))

#+++ Refactor (climatologies should be read in separate node and connected via filler node)
def forcing_gap_filler(nc_path,pattern,nc_var,climatology_file): 
    return GraphNode('forcing_with_climatology',callable_to_funcspec(ForcingGapFiller),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=odict(path=nc_path,pattern=pattern,nc_var=nc_var,filler_fn=climatology_file))

def gap_filler(base_node,fill_node):
    return GraphNode('gap_filler',GapFiller,out_type='from_inputs',inputs=[base_node,fill_node],properties={})

def monthly_climatology(filename,variable,preload=True):
    return GraphNode('monthly_climatology',callable_to_funcspec(MonthlyClimatologyNode),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=odict(filename=filename,variable=variable,preload=preload))

def init_state_from_ncfile(path,pattern,nc_var,date=None):
    return GraphNode('init_state_from_ncfile',callable_to_funcspec(StateFileReaderNode),out_type='init',inputs=None,properties={'io': 'r'},\
             args=odict(path=path,pattern=pattern,nc_var=nc_var,date=date))

def init_state_from_array(data,extent):
    return GraphNode('init_state_from_array',callable_to_funcspec(InitStateFromArray),out_type='init',inputs=None,properties={'io': 'r'},\
             args=odict(data=data,extent=extent))

def _spatial_from_hdf5(filename,variable,preload=False):
    return GraphNode('spatial_from_hdf5',_SpatialHDF5Node,out_type='spatial',inputs=None,properties={'io': 'r'},\
             args=odict(filename=filename,variable=variable,preload=preload))

def spatial_from_file(filename,variable=None,preload=False,force_grid=None):
    from .file_nodes import SpatialFileNode
    return GraphNode('spatial_from_file',SpatialFileNode,out_type='spatial',inputs=None,properties={'io': 'r'},\
             args=odict(filename=filename,variable=variable,preload=preload,force_grid=force_grid))

def spatial_from_multifile(pattern,dimension,dim_func,variable=None,preload=False):
    from .file_nodes import SpatialMultiFileNode
    return GraphNode('spatial_from_from_multifile',SpatialMultiFileNode,out_type='spatial',inputs=None,properties={'io': 'r'},\
             args=odict(pattern=pattern,dimension=dimension,dim_func=dim_func,variable=variable,preload=preload))

def static(data,dataspec,out_type,flat=True):
    return GraphNode('static',StaticDataNode,out_type=out_type,inputs=None,args=odict(data=data,dataspec=dataspec,flat=flat))

def hypsometric_from_multifile(pattern,variable=None,preload=False):
    return spatial_from_multifile(pattern,get_dimension('hypsometric_percentile'),_hyps_layer_func,variable,preload)

def _hypso_from_hdf5(filename,variable,preload=False):
    return GraphNode('hypso_from_hdf5',_HypsoHDF5Node,out_type='spatial',inputs=None,properties={'io': 'r'},\
             args=odict(filename=filename,variable=variable,preload=preload))

def climatology(filename,variable,preload=False):
    return GraphNode('climatology',ClimatologyNode,out_type='from_inputs',inputs=None,properties={'io': 'r'},\
             args=odict(filename=filename,variable=variable,preload=preload))

def const_from_hdf5(filename,variable,dims):
    return GraphNode('const_from_hdf5',ConstHDF5Node,out_type='const',inputs=None,properties={},\
         args=odict(filename=filename,variable=variable,dims=dims))

def const_array(value,dims):
    return GraphNode('constant',ConstArray,out_type='const',inputs=None,properties={},args=odict(value=value,dims=dims))

def const(value,**kwargs):
    return GraphNode('constant',ConstNode,out_type='const',inputs=None,properties=kwargs,args=odict(value=value))

def parameter(value,min_val,max_val,fixed=False,description='',**kwargs):
    return GraphNode('parameter',ParameterNode,out_type='const',inputs=None,properties=kwargs,args=odict(value=value,min_val=min_val,max_val=max_val,fixed=fixed,description=description))

def parameter_from_calibration_results(filename, key):
    return GraphNode('parameter',CalibrationResultsParameterNode,out_type='const', \
                     inputs=None, properties = {}, args = odict(filename=filename, key=key))

def parameter_from_json(filename,key,value=None,min_val=None,max_val=None,fixed=None):
    return GraphNode('parameter',JSONParameterNode,out_type='const',inputs=None,properties={},args=odict(filename=filename,key=key,min_val=min_val,max_val=max_val,fixed=fixed))

def assign(from_node):
    return GraphNode('assign',AssignNode,out_type='from_inputs',inputs=[from_node],properties={})

def write_to_ncfile(path,nc_var,mode='a',dtype=np.float64):
    '''
    save output to a netCDF4 file
    :param path: save path
    :param nc_var: name of variable
    :param mode: 'w' clobber or 'a' or 'r+' append
    :return: GraphNode
    '''
    return GraphNode('write_to_ncfile',callable_to_funcspec(FileWriterNode),out_type='ncfile',inputs=[nc_var],
                     properties={'io': 'w'},args=odict(path=path,nc_var=nc_var,mode=mode,dtype=dtype,chunksizes=chunksizes,ncparams=ncparams))


def write_to_annual_ncfile(path,nc_var,mode='a',dtype=np.float64,chunksizes=None,ncparams=None):
    '''
    save output to annually split netCDF4 files
    :param path: save path
    :param nc_var: name of variable
    :param mode: 'w' clobber or 'a' or 'r+' append
    :return: GraphNode
    '''
    return GraphNode('write_to_annual_ncfile',callable_to_funcspec(SplitFileWriterNode),out_type='ncfile',inputs=[nc_var],
                     properties={'io': 'w'},args=odict(path=path,nc_var=nc_var,mode=mode,dtype=dtype,chunksizes=chunksizes,ncparams=ncparams))

def write_to_ncfile_snapshot(path,nc_var,mode='a',freq='M',dtype=np.float64,chunksizes=None,ncparams=None):
    return GraphNode('write_to_ncfile_snapshot',callable_to_funcspec(FlatFileWriterNode),out_type='ncfile',inputs=[nc_var],
                     properties={'io': 'w'},args=odict(path=path,nc_var=nc_var,mode=mode,freq=freq,dtype=dtype,chunksizes=chunksizes,ncparams=ncparams))

def model_output(var_name):
    return GraphNode('output_variable',OutputNode,out_type=None,inputs=None,properties={'io': 'from_model'},
             args=odict(var_name=var_name))

def persist(var_name):
    return GraphNode('output_variable',OutputNode,out_type='save',inputs=[var_name],properties={'io': 'from_model'},
             args=odict(var_name=var_name))

def init_states_from_dict(imap,data_map,extent):
    '''
    replace existing initial state mappings with ndarrays in data_map
    :param mapping:
    :param data_map:
    :return:
    '''
    for k in data_map:
        imap[k] = init_state_from_array(data_map[k],extent=extent)


def _hyps_layer_func(f):
    '''
    Convenience function to map a group of 3 numbers in a filename to a floating point value
    e.g "Hypsometric_Grid_025.flt" -> 25.0
    '''

    import re
    return float(re.match('.*([0-9]{3})',f).group(1))

def time_offset_reader(existing,offset):
    return GraphNode('time_offset_reader',TimeOffsetReaderNode,out_type=existing.out_type,inputs=None,properties={'io': 'r'},\
         args=dict(child=existing.to_dict(),offset=offset))
