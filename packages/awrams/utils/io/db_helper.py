import h5py
import numpy as np
from collections import OrderedDict
import types

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('awrams.utils.io')

#from awrams.utils.settings import VAR_CHUNK_CACHE_SIZE, VAR_CHUNK_CACHE_NELEMS, VAR_CHUNK_CACHE_PREEMPTION#pylint: disable=no-name-in-module

from awrams.utils.config_manager import get_system_profile

sys_settings = get_system_profile().get_settings()
VAR_CHUNK_CACHE_SIZE = sys_settings['IO_SETTINGS']['VAR_CHUNK_CACHE_SIZE']
VAR_CHUNK_CACHE_NELEMS = sys_settings['IO_SETTINGS']['VAR_CHUNK_CACHE_NELEMS']
VAR_CHUNK_CACHE_PREEMPTION = sys_settings['IO_SETTINGS']['VAR_CHUNK_CACHE_PREEMPTION']

propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
settings = list(propfaid.get_cache())
#settings[1]        # size of hash table
settings[2] = 0     #2**17 # =131072 size of chunk cache in bytes
                    # which is big enough for 5x(75, 1, 50 chunks;
                    # default is 2**20 =1048576
settings[3] = 1.    # preemption 1 suited to whole chunk read/write
propfaid.set_cache(*settings)
propfaid.set_fapl_sec2()
propfaid.set_sieve_buf_size(0)
propfaid.set_fclose_degree(h5py.h5f.CLOSE_STRONG)
#propfaid.set_fapl_stdio()

mdc = dict(off  =(False,),
           K1   =(True, 2**10, 2**10, 2**10),
           K32  =(True, 2**15, 2**15, 2**15),
           K262 =(True, 2**18, 2**18, 2**18),
           M1   =(True, 2**20, 2**20, 2**20),
           )

def decode(str_like):
    '''
    some versions of h5py return strings or bytes in a seemingly arbitrary fashion... try to work around this
    '''
    if isinstance(str_like,str):
        return str
    elif isinstance(str_like,np.bytes_):
        return str_like.tostring().decode()
    elif isinstance(str_like,np.ndarray):
        return str_like[0].decode()
    else:
        return str_like.decode()

class h5pyOb:
    def __init__(self):
        pass

    def _init_dims(self):
        self.dimensions = OrderedDict()
        dmap = {}
        for k,v in self.items():
            try:
                v_class = decode(v.attrs['CLASS'])
                if v_class == "DIMENSION_SCALE":
                    if k in self.variables:
                        v_ref = v.attrs['REFERENCE_LIST'][0][1]
                        dim = h5pyDimension()
                        if v.maxshape[0] is None:
                            dim._isunlimited = True
                        dmap[v_ref] = [k,dim]
            except:
                pass

        #for i in range(len(dmap)):
        for k,v in dmap.items():
            #mapping = dmap[i]
            #self.dimensions[mapping[0]] = mapping[1]   
            self.dimensions[v[0]] = v[1]   

    def _init_attrs(self):
        for k in self.attrs.keys():
            try:
                #self.__dict__[k] = self.attrs[k].decode(errors='ignore')
                self.__dict__[k] = decode(self.attrs[k])
            except AttributeError:
                self.__dict__[k] = self.attrs[k]

    def _populate(self):
        self.variables = dict()
        self.groups = dict()

        for k in self.keys():
            if isinstance(self[k], h5py.Dataset):
                #if 'NAME' in self[k].attrs and self[k].attrs['NAME'].decode().startswith("This is a netCDF dimension but not a netCDF variable"):
                if 'NAME' in self[k].attrs and decode(self[k].attrs['NAME']).startswith("This is a netCDF dimension but not a netCDF variable"):
                    ### this for dimension 'nv' in monthly files
                    continue
                self.variables[k] = h5pyDataset(self[k])
                try:
                    self.variables[k].units = decode(self[k].attrs['units'])
                except KeyError:
                    self.variables[k].units = None

            elif isinstance(self[k], h5py.Group):
                self.groups[k] = h5pyGroup(self[k])

        self._init_dims()

    def ncattrs(self):
        return list(self.attrs.keys())

    def getncattr(self, key):
        try:
            if self.attrs[key].shape == (1,):
                return decode(self.attrs[key][0]) #np.asscalar(self.attrs[key])
        except:
            pass

        return self.__dict__[key]

    def filters(self):
        pars = {}
        pars['fletcher32'] = self.fletcher32
        pars['shuffle'] = self.shuffle
        pars['zlib'] = self.compression == 'gzip' and True or False
        pars['complevel'] = self.compression_opts
        return pars

    def chunking(self):
        return self.chunks


class h5pyDimension:
    def __init__(self):
        self._isunlimited = False

    def isunlimited(self):
        return self._isunlimited


class h5pyGroup(h5py.Group,h5pyOb):
    def __init__(self, group):
        h5py.Group.__init__(self, group.id)
        h5pyOb.__init__(self)
        self._populate()
        self._init_attrs()

    def createGroup(self, name):
        """
        to mimick netCDF4 createGroup method
        creates group with id=gid (becomes self.id of new group)
        """
        gid = h5py.h5g.create(self.id, name)
        self.groups[name] = h5pyGroup(gid)
        return self.groups[name]

    def setncattr(self, key, value):
        self.attrs[key] = value


class h5pyDataset(h5py.Dataset,h5pyOb):
    def __init__(self, dataset):
        h5py.Dataset.__init__(self, dataset.id)
        self._init_attrs()

        if 'DIMENSION_LIST' in self.attrs:
            self.dimensions = OrderedDict()
            for d in self.dims:
                for i in d.items():
                    self.dimensions[i[0]] = i[1]

        elif 'CLASS' in self.attrs:
                v_class = self.attrs['CLASS'].decode()
                if v_class == "DIMENSION_SCALE":
                    self.dimensions = OrderedDict()
                    self.dimensions[self.name] = h5pyDimension()

    #    self.name = super(h5pyDataset,self).__getattribute__('name')

    # @property
    # def name(self):
    #     return super(h5pyDataset,self).__getattribute__('name')

    # def __getitem__(self, slice):
    #     """
    #     always return masked array; much better than netCDF4 which will return
    #     numpy.ndarray if no masked_values present = PAINFUL!
    #     """
    #     #logger.info("h5pyDataset.__getitem__")
    #     data = super(h5pyDataset, self).__getitem__(slice)
    #     if data.size == 1: # a scalar #type(slice) == int:
    #         return np.ma.masked_values([data], self.fillvalue)[0]
    #     else:
    #         return np.ma.masked_values(data, self.fillvalue)

    # def __setitem__(self, slice, data):
    #     """
    #     handle resizing here
    #     """
    #     logger.info("h5pyDataset.__setitem__ %s %s %s %s %s %s",self.file,self.name,slice,self.shape,self[slice].shape,len(data))
    #     if self[slice].shape[0] < len(data):
    #         self.resize((len(data),))
    #     logger.info("h5pyDataset.__setitem__")
        #h5py.Dataset.__setitem__(self, slice, data)
    #    h5py.Dataset.write_direct(data, dest_sel=slice)

class _h5py(h5py.File,h5pyOb):
    """
    open database with h5py.File and return object that looks like netCDF4.Dataset
    """
    def __init__(self, file_name, mode='r'):
        self.file_name = file_name
        self._filepath = file_name # mimick netCDF4 attribute

        flags = h5py.h5f.ACC_RDWR
        if mode == 'r':
            flags = h5py.h5f.ACC_RDONLY
        fid = h5py.h5f.open(file_name.encode(), flags=flags, fapl=propfaid)

        self.file_id = fid
        self._id = fid

        h5py.File.__init__(self, fid) #, 'a', driver='sec2')
        h5pyOb.__init__(self)

        self.root = h5pyGroup(self)

        try:
            self.var_name = decode(self.attrs['var_name'])
        except KeyError:
            pass #self.var_name = None

        self._populate()
        self._init_attrs()

    def open(self,fid):
        h5py.File.__init__(self, fid) #, 'a', driver='sec2')
        h5pyOb.__init__(self)

        self.root = h5pyGroup(self)


        self.var_name = decode(self.attrs['var_name'])

        self._populate()
        self._init_attrs()

    def createGroup(self, name):
        self.groups[name] = self.root.createGroup(name)
        return self.groups[name]
        #return self.root.createGroup(name)

    def filepath(self):
        return self._filepath

    def write_direct(self, data, dest_sel):
        self[self.var_name].write_direct(data, dest_sel=dest_sel)

    def flush(self):
        h5py.h5f.flush(self.file_id, h5py.h5f.SCOPE_GLOBAL)

    def close(self):
        self.flush()
        h5py.File.close(self)
        #self.file_id.close()

    def sync(self):
        self.flush()

    def set_mdc(self,mdc):
        h5f = self.file_id
        mdc_cache_config = h5f.get_mdc_config()
        mdc_cache_config.set_initial_size = mdc[0] #True
        mdc_cache_config.initial_size = mdc[1] #1024
        mdc_cache_config.max_size = mdc[2] #1024
        mdc_cache_config.min_size = mdc[3] #1024
        h5f.set_mdc_config(mdc_cache_config)

try:
    import netCDF4 as nc

    class _nc(nc.Dataset):
        """
        open database and set chunk cache
        """
        def __init__(self, file_name, mode='r'):
            nc.Dataset.__init__(self, file_name, mode)
            for v in self.variables:
                self.variables[v] = _v(self.variables[v])

        def get_attr(self,key):
            return self.variables[self.var_name].getncattr(key)

        def set_chunk_cache(self, **params):
            p = dict(var_chunk_cache_size=VAR_CHUNK_CACHE_SIZE,
                     var_chunk_cache_nelems=VAR_CHUNK_CACHE_NELEMS,
                     var_chunk_cache_preemption=VAR_CHUNK_CACHE_PREEMPTION)
            p.update(**params)
            self.variables[self.var_name].set_var_chunk_cache(size=p['var_chunk_cache_size'],
                                                              nelems=p['var_chunk_cache_nelems'],
                                                              preemption=p['var_chunk_cache_preemption'])

        def flush(self):
            self.sync()

        def __getitem__(self,idx):
            return self.variables[idx]

except:
    pass

'''
Provide a DB_OPEN_WITH wrapper function that tries to open a file with _h5py, then attempts _nc
if it fails (useful to enforce h5py being the primary API but still support netCDF3 files)
'''
def _fallthrough(file_name,mode='r'):
    try:
        return _h5py(file_name,mode)
    except:
        return _nc(file_name,mode)

class _v:
    '''
    wrap existing Variable to add property attrs
    '''
    def __init__(self,v):
        self.v = v
        self.name = self.v.name
        self.attrs = {}
        for a in self.ncattrs():
            self.attrs[a] = [self.v.getncattr(a)]

        if hasattr(v,'units'):
            self.units = self.v.units
        if hasattr(v,'bounds'):
            self.bounds = self.v.bounds
        if hasattr(v,'fillvalue'):
            self.fillvalue = self.v.fillvalue
        elif '_FillValue' in self.attrs:
            self.fillvalue = self.attrs['_FillValue']
        if hasattr(v,'dimensions'):
            self.dimensions = self.v.dimensions
        self.dtype = self.v.dtype
        self.shape = self.v.shape
        

    def __getitem__(self, idx):
        return self.v[idx]

    def __setitem__(self, idx, data):
        self.v[idx] = data

    def getncattr(self,k):
        return self.v.getncattr(k)

    def ncattrs(self):
        return self.v.ncattrs()

    def __len__(self):
        return len(self.v)

# def set_mdc(h5_file,mdc):
#     h5f = h5_file.fid
#     mdc_cache_config = h5f.get_mdc_config()
#     mdc_cache_config.set_initial_size = mdc[0] #True
#     mdc_cache_config.initial_size = mdc[1] #1024
#     mdc_cache_config.max_size = mdc[2] #1024
#     mdc_cache_config.min_size = mdc[3] #1024
#     h5f.set_mdc_config(mdc_cache_config)

def _set_h5nc_attr_DIRECT(nc_obj,k,v):
    try:
        nc_obj.setncattr(k,v)
    except:
        print("Failed setting %s as %s on %s" % (k,v,nc_obj))
        raise

def _set_h5nc_attr_NEWSTYLE(nc_obj,k,v):
    if isinstance(k,str):
        nc_obj.setncattr_string(k,v)
    else:
        nc_obj.setncattr(k,v)

set_h5nc_attr = _set_h5nc_attr_DIRECT
