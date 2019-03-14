from awrams.utils.mapping_types import *
import awrams.utils.datetools as dt
import pandas as pd
import numpy as np
from awrams.utils.precision import aquantize
from awrams.utils.metatypes import ObjectDict as o
import os, shutil
import re

from awrams.utils.general import Indexer

#import netCDF4 as ncd
from collections import OrderedDict
from awrams.utils.io import open_append

from awrams.utils.messaging.buffers import shm_as_ndarray

# +++ Tidy namespace
from awrams.utils.messaging.general import *

from awrams.utils.io.netcdf_wrapper import NCVariableParameters

#from awrams.utils.settings import DB_OPEN_WITH, MAX_FILES_PER_SFM #pylint: disable=no-name-in-module

from awrams.utils.awrams_log import get_module_logger as _get_module_logger
logger = _get_module_logger('data_mapping')

import glob

from awrams.utils.io import db_open_with
db_opener = db_open_with()
import awrams.utils.io.db_helper as dbh
set_h5nc_attr = dbh.set_h5nc_attr

from awrams.utils.config_manager import get_system_profile

sys_settings = get_system_profile().get_settings()

DB_OPEN_WITH = sys_settings['IO_SETTINGS']['DB_OPEN_WITH']
MAX_FILES_PER_SFM = sys_settings['IO_SETTINGS']['MAX_FILES_PER_SFM']
DEFAULT_CHUNKSIZE = sys_settings['IO_SETTINGS']['DEFAULT_CHUNKSIZE']

def find_time_coord(coordinates):
    '''
    Locate the time dimension in a set of coordinates
    '''
    for i, coord in enumerate(coordinates):
        if isinstance(coord.dimension,TimeDimension):
            return i, coord
    raise Exception("No time dimension found in coordinate set")

def substitute_index(indices,index_to_sub,position):
    '''
    Replace the item in <position> with index_to_sub, return a tuple
    '''
    out_index = []
    for i in range(0,len(indices)):
        if i == position:
            out_index.append(index_to_sub)
        else:
            out_index.append(indices[i])
    return tuple(out_index)

def normalize_slice(s,dim_len):
    '''
    normalize a slice to start at 0, where dim_len is the max value
    '''
    if s.start == None:
        return s
    else:
        offset = s.start
        stop = dim_len - offset if s.stop == None else s.stop - offset
        return slice(0,stop,s.step)

def offset_slice(s,offset):
    return slice(s.start+offset,s.stop+offset)

class IndexGetter:
    '''
    Helper class for using index creation shorthand
    eg IndexGetter[10:,5] returns [slice(10,None),5]
    '''
    def __getitem__(self,indices):
        return indices

index = IndexGetter()

'''
Convenience functions from filtering/mapping dates in SplitFileManager.open_existing()
'''

def filter_years(period):
    from re import match

    years = np.unique(period.year)
    def ff(x):
        m = match('.*([0-9]{4})',x)
        try:
            return int(m.groups()[0]) in years
        except:
            return False
    return ff

def map_filename_annual(fn):
    file_year = re.match('.*([0-9]{4})',fn).groups()[0]
    return dt.dates(file_year)

'''
'''

def check_inverted_lat(ref_cs,comp_cs):
    inverted = False
    if len(comp_cs.latitude) > 1:
        inverted = (ref_cs.latitude[1] > ref_cs.latitude[0]) != (comp_cs.latitude[1] > comp_cs.latitude[0])

    if inverted:
        out_coords = comp_cs.update_coord(comp_cs.latitude.iindex[::-1],False)
    else:
        out_coords = comp_cs

    return inverted, out_coords

        #+++CMS

def split_padding(period,avail):
    try:
        avail_start = avail[0].to_timestamp()
        avail_end = avail[-1].to_timestamp()
    except AttributeError:
        avail_start = avail[0]
        avail_end = avail[-1]

    if period[0] < avail_start:
        prepad = (period[0], min(period[-1],avail_start-1))
    else:
        prepad = None
    if period[-1] > avail_end:
        postpad = (max(period[0],avail_end+1), period[-1])
    else:
        postpad = None
        
    actual = max(avail_start,period[0]),min(avail_end,period[-1])
        
    if period[-1] < avail_start:
        actual = None
    if period[0] > avail_end:
        actual = None
        
    return prepad,actual,postpad

def index_shape(idx,numpy_simplify=False):
    shape = []
    for i in idx:
        if isinstance(i,slice):
            shape.append(i.stop-i.start)
        else:
            if not numpy_simplify:
                shape.append(1)
    return tuple(shape)

def simple_shape(shape):
    return tuple([s for s in shape if s > 1])

def desimplify(target):
    '''
    Return indices returning all data for a target shape
    '''
    indices = []
    for i,s in enumerate(target):
        if s == 1:
            indices.append(0)
        else:
            indices.append(np.s_[:])
    return indices

def simplify_indices(indices):
    '''
    Reduce dimensionality if any of the indices are of size 1
    '''
    out_index = []
    for i in range(0,len(indices)):
        cur_index = indices[i]
        if type(cur_index) != int:
            out_index.append(cur_index)
    return tuple(out_index)

class SplitFileManager:
    def __init__(self,path,mapped_var,max_open_files=MAX_FILES_PER_SFM):
        '''
        Manage NetCDF data persistence for a given variable.
        '''

        self.mapped_var = mapped_var
        self.path = path

        self.time_file_map = {}
        self.file_time_map = {}
        self._open_datasets = OrderedDict()
        self._open_accessors = OrderedDict()
        self.max_open_files = max_open_files

    def locate_day(self,day):
        #+++ Duplicate?
        t_idx = self.splitter.coordinates.time.get_index(day)
        idx, seg = self.splitter._locate_segment(t_idx)
        return self.file_map[seg]

    def get_fn_ds_map(self):
        return dict(var_name=self.mapped_var.variable.name, splitter=self.splitter, file_map=self.file_map)

    def get_frequency(self):
        return self.splitter.coordinates.index.freq

    def get_period_map_multi(self,periods):
        '''
        Return the period mapping for a presplit set of periods
        '''
        pmap = {}

        def find_period(p,tfm):
            for k,v in tfm.items():
                try:
                    idx = v.get_index(p)
                    return k,idx
                except:
                    pass
            raise IndexError()

        for i,p in enumerate(periods):
            #+++
            #Don't set when file doesn't exist, also catches
            #0 length periods... still, might hide other exceptions?
            try:
                #s_idx = self.splitter.coordinates.time.get_index(p[0])
                #idx,seg = self.splitter._locate_segment(s_idx)
                #t_idx = seg.coordinates.time.get_index(p)
                #fname = self.file_map[seg]
                fname,t_idx = find_period(p,self.file_time_map)
                pmap[i] = {}
                pmap[i]['filename'] = fname
                pmap[i]['time_index'] = t_idx
            except IndexError:
                pass
        return pmap

    def get_period_map(self,period):
        '''
        Return the period mapping for the given period (splitting if needed)
        '''
        periods = dt.split_discontinuous_dti(period)
        all_periods = []
        for p in periods:
            all_periods += self.splitter.split_dti(p)
        return self.get_period_map_multi(all_periods), all_periods

    def get_chunked_periods(self,chunksize):
        '''
        Return (ordered) DatetimeIndices of all the HDF chunks contained in the dataset
        '''
        chunk_p = []
        for v in self.file_time_map.values():
            chunk_p += dt.split_period_chunks(v,chunksize)
        chunk_p.sort(key = lambda x: x[0])
        return chunk_p

    @classmethod
    def open_existing(self,path,pattern,variable,mode='r',ff=None,max_open_files=MAX_FILES_PER_SFM,map_func=None):
        '''
        classmethod replacement for open_files
        '''
        sfm = SplitFileManager(None,None)
        self.mode = mode
        sfm.map_files(path,pattern,variable,ff=ff,map_func=map_func)
        return sfm

    def map_files(self,path,pattern,variable,ff=None,max_open_files=MAX_FILES_PER_SFM,map_func=None):

        var_name = variable if isinstance(variable,str) else variable.name

        self.var_name = var_name

        search_pattern = os.path.join(path,pattern)
        files = glob.glob(search_pattern)
        files.sort()

        if ff is None:
            def ff(x):
                return True

        _files = []
        for f in files:
            if ff(f):
                _files.append(f)
        files=_files

        if len(files) == 0:
            raise Exception("No files found in %s matching %s" % (path,pattern))

        #import netCDF4 as ncd
        #db_opener_TEST = ncd.Dataset

        dsm_start = DatasetManager(db_opener(files[0],self.mode))
        #dsm_start = DatasetManager(open_append(db_opener_TEST,files[0],self.mode))

        self.ref_ds = dsm_start

        coords = dsm_start.get_coords()


        time = dsm_start.get_coord('time')
        time_idx = time.index

        self.file_time_map[files[0]] = time
        self.time_file_map[time.index[0]] = files[0]

        tsegs = [time.index]

        if len(files) > 1:
            for fn in files[1:]:
                if map_func is not None:
                    t = map_func(fn)
                else:
                    dsm = DatasetManager(db_opener(fn,self.mode))
                    t = dsm.get_coord('time')
                self.file_time_map[fn] = t
                self.time_file_map[t[0]] = fn
                #self.time_access_map[t[0]] = dsm.variables[var_name] #+++ as below...
                #self.datasetmanager_map[t[0]] = dsm #+++ deprecate? Only works if files are open...
                tsegs.append(t)
                #time_idx = time_idx.union(t.index)  

        new_segs = []

        for i in range(1,len(tsegs)):
            first_new, last_old = tsegs[i][0], tsegs[i-1][-1]
            tdelta = first_new - last_old

            day = dt.days(1)

            if tdelta > day:
                new_t = dt.dates(last_old+day,first_new-day)
                new_segs.append(new_t)
                self.time_file_map[new_t[0]] = None

        all_segs = sorted(tsegs+new_segs,key=lambda t: t[0])

        self.seg_time_map = dict([(i,t[0]) for i,t in enumerate(all_segs)])

        full_t = time.index.union_many(all_segs)
        full_tc = TimeCoordinates(time.dimension,full_t)

        self.cs = CoordinateSet((full_tc,coords.latitude,coords.longitude))

        self.splitter = Splitter(full_tc,all_segs)

        ncvar = dsm_start.variables[var_name]
        self.fillvalue = ncvar.attrs['_FillValue'][0]
        #self.fillvalue = 1.0
        v = Variable.from_ncvar(ncvar)

        self.mapped_var = MappedVariable(v,self.cs,ncvar.dtype)

    def create_files(self,schema,leave_open=True,clobber=False,chunksize=None,file_creator=None,file_appender=None,create_dirs=True,ncparams=None,**kwargs):
        '''
        kwargs are propagated to NCVariableParameters
        '''

        if ncparams is None:
            ncparams = {}
        
        if file_creator is None:
            def create_new_nc(fn):
                import netCDF4 as ncd
                #import awrams.utils.io.h5netcdf.legacyapi as ncd
                try:
                    return ncd.Dataset(fn,'w')
                except RuntimeError:
                    from awrams.utils.io.general import h5py_cleanup_nc_mess
                    h5py_cleanup_nc_mess(fn)
                    return ncd.Dataset(fn,'w')
                #return db_opener(fn,'w')
            file_creator = create_new_nc

        if file_appender is None:
            def append_nc(fn):
                # return ncd.Dataset(fn,'a')
                try:
                    #return db_opener(fn,'a')
                    return open_append(db_opener,fn,'a')
                except:
                    logger.critical("EXCEPTION: %s",fn)
                    raise
            file_appender = append_nc

        if create_dirs:
            os.makedirs(self.path,exist_ok=True)

        period = self.mapped_var.coordinates.time.index
        split_periods = schema.split_periods(period)

        self.var_name=self.mapped_var.variable.name

        # Examine first existing file to see if we need to extend the coordinates
        if clobber == False:
            p = split_periods[0]
            fn = os.path.join(self.path,schema.gen_file_name(self.mapped_var.variable,p)+'.nc')

            if os.path.exists(fn):
                ds = file_appender(fn)
                dsm = DatasetManager(ds)
#                logger.info("filename %s",fn)
                existing_coords = dsm.get_coords()

                if 'time' in existing_coords:
                    #+++
                    # Could definitely generalise this to autoexpand in
                    # other dimensions, hardcoding for time being the 'normal' case...

                    #seg_time = seg.coordinates.time

                    existing_time = dsm.get_coord('time').index
                    extension_time = p

                    new_seg_tc = period_to_tc(existing_time.union(extension_time))

                    global_extension = self.mapped_var.coordinates.time.index

                    new_global_tc = period_to_tc(existing_time.union(global_extension))

                    dsm.set_time_coords(new_seg_tc)

                    self.mapped_var.coordinates.update_coord(new_global_tc)

                    period = self.mapped_var.coordinates.time.index
                    split_periods = schema.split_periods(period)

                ds.close()
                #self.splitter.set_coordinates(self.mapped_var.coordinates)

        for p in split_periods:
            fn = os.path.join(self.path,schema.gen_file_name(self.mapped_var.variable,p)+'.nc')

            tc = period_to_tc(p)

            new_file = True

            if os.path.exists(fn):
                if not clobber:
                    ds = file_appender(fn)
                    dsm = DatasetManager(ds)

                    if len(p) > len(ds.variables['time']):
                        dsm.set_time_coords(tc, resize=True)

                    new_file = False
                else:
                    os.remove(fn)
           

            '''
            Separate into function ('createCoordinates'?)
            Possibly removing any (direct) reference to netCDF
            '''
            if new_file:
                ds = file_creator(fn)# ncd.Dataset(fn,'w')
                dsm = DatasetManager(ds)

                cur_cs = CoordinateSet((tc,self.mapped_var.coordinates.latitude,self.mapped_var.coordinates.longitude))

                for coord in cur_cs:
                    dsm.create_coordinates(coord)

                from awrams.utils.io.netcdf_wrapper import NCVariableParameters

                if chunksize is None:
                    chunksize = DEFAULT_CHUNKSIZE #pylint: disable=no-name-in-module

                chunksizes = cur_cs.validate_chunksizes(chunksize)

                ncd_params = NCVariableParameters(chunksizes=chunksizes,**kwargs)
                ncd_params.update(**ncparams)

                target_var = dsm.create_variable(self.mapped_var,ncd_params)
                dsm.awra_var = target_var

                set_h5nc_attr(dsm.ncd_group,'var_name',self.mapped_var.variable.name)
                #dsm.ncd_group.setncattr('var_name',self.mapped_var.variable.name)

                dsm.set_time_coords(cur_cs.time, resize=True)

            ds.close()

        if leave_open:
            self.mode = 'a'
            all_files = [schema.gen_file_name(self.mapped_var.variable,p) + '.nc' for p in schema.split_periods(period)]
            self.map_files(self.path,'*',self.mapped_var.variable.name,ff=lambda f: os.path.split(f)[1] in all_files)   

    def cell_for_location(self,location):
        lat_i = self.splitter.coordinates.latitude.get_index(lat)
        lon_i = self.splitter.coordinates.longitude.get_index(lon)
        return (lat_i,lon_i)



    def finalise(self):
        '''
        Flush all current writes to disk'
        '''
        self.accessor.finalise()

    def _DEP_open_all(self,mode='r'):
        # +++ 
        for seg in self.splitter.segments:
            ds = db_opener(self.file_map[seg],'a')

            if DB_OPEN_WITH == '_h5py':
                ds.set_mdc(dbh.mdc['K32'])

            dsm = DatasetManager(ds)
            self.datasetmanager_map[seg] = dsm

            target_var = ds.variables[self.mapped_var.variable.name]
            self.array_map[seg] = target_var

        self.accessor = SplitArrayOutputWriter(list(self.array_map.values()),self.splitter)

    def close_all(self):
        '''
        Close all open datasets
        '''
        for ds in list(self._open_datasets.values()):
            ds.close()
        self._open_datasets = OrderedDict()
        self._open_accessors = OrderedDict()

    def get_extent(self):
        return self.ref_ds.get_extent()

    def get_coords(self):
        return self.cs

    def _get_accessor(self,time_index):
        if time_index not in self._open_accessors:
            fn = self.time_file_map[time_index]
            if fn is None:
                return None

            try:
                self._open_datasets[time_index] = ds = DatasetManager(db_opener(fn,self.mode))
            except Exception as e:
                print(time_index,fn)
                raise(e)

            if DB_OPEN_WITH == '_h5py':
                ds.ncd_group.set_mdc(dbh.mdc['K32'])
            try:
                self._open_accessors[time_index] = ds.variables[self.var_name] # open_dataset(time_file_map[key])
            except:
                print(list(ds.variables),self.var_name)
                raise
        if len(self._open_datasets) > self.max_open_files:
            _,ds = self._open_datasets.popitem(False) # close_dataset(ofiles[key])
            ds.close()
            self._open_accessors.popitem(False)
        return self._open_accessors[time_index]

    def get_by_index(self,indices):

        out_shape = index_shape(indices)
        out_data = np.empty(out_shape,dtype=self.mapped_var.dtype) #+++ dtype
        out_data.fill(np.nan)

        seg_indices = self.splitter.split_indices(indices[0])

        g_offset = 0 - seg_indices[0].global_index.start

        if len(seg_indices) == 1:
            seg_i = seg_indices[0]
            tidx = self.seg_time_map[seg_i.seg_idx]
            accessor = self._get_accessor(tidx)#self.time_access_map[tidx]
            if accessor is not None:
                data = accessor[seg_i.local_index,indices[1],indices[2]]
                dshape = index_shape((seg_i.local_index,indices[1],indices[2]))
                data[data==accessor.fillvalue] = np.nan
                return data.reshape(dshape)
  
        out_data = np.empty(out_shape,dtype=self.mapped_var.dtype) #+++ dtype
        out_data.fill(np.nan)

        for seg_i in seg_indices:
            tidx = self.seg_time_map[seg_i.seg_idx]
            accessor = self._get_accessor(tidx)#self.time_access_map[tidx]
            if accessor is not None:
                data = accessor[seg_i.local_index,indices[1],indices[2]]
                dshape = index_shape((seg_i.local_index,indices[1],indices[2]))
                data[data==accessor.fillvalue] = np.nan
                out_data[offset_slice(seg_i.global_index,g_offset)] = data.reshape(dshape)
        
        return out_data

    def set_by_index(self,indices,data):

        set_shape = index_shape(indices)

        seg_indices = self.splitter.split_indices(indices[0])

        g_offset = 0 - seg_indices[0].global_index.start

        for seg_i in seg_indices:
            tidx = self.seg_time_map[seg_i.seg_idx]
            accessor = self._get_accessor(tidx)
            if accessor is not None:
                to_write = data[offset_slice(seg_i.global_index,g_offset)]
                to_write = to_write.reshape(simple_shape(to_write.shape))
                accessor[seg_i.local_index,indices[1],indices[2]] = to_write
            else:
                raise Exception("No file store available for this period")

    def set_by_coords(self,coords,data):
        #+++ Does not yet handle inverted latitudes
        full_idx = self.cs.get_index(coords)
        self.set_by_index(full_idx,data)

    def get_data(self,period,extent):
        '''
        Return a datacube as specified in time and space
        Doesn't support masking (yet)
        '''
        if not hasattr(period,'__len__'):
            #Assume we've been given a single date rather than a DTI
            period = pd.DatetimeIndex([period])
            single_day = True
        else:
            single_day = False

        data = self.get_by_coords(gen_coordset(period,extent))
        if single_day:
            data = data[0]

        return data

    def get_by_coords(self,coords):
        #+++ Does no masking
        inverted, coords = check_inverted_lat(self.cs,coords)

        full_idx = self.cs.get_index(coords)
        
        out_data = self.get_by_index(full_idx)

        if inverted:
            # +++ Maybe need smarter inversion for non time/lat/lon datasets
            out_data = out_data[:,::-1,:]

        return out_data

    def get_padded_by_coords(self,coords):
        actual = self.splitter.coordinates
        period = coords.time.index
        split = split_padding(period,actual.index)

        full_shape = coords.shape
        
        #full_data = np.ma.empty(full_shape,dtype=np.float32) #+++ Get from nc_var dtype
        #full_data.mask = True

        
        if split[1] is None:
            full_data = np.empty(full_shape,dtype=self.mapped_var.dtype)
            full_data[...] = np.nan
            return full_data

        inverted, coords = check_inverted_lat(self.cs,coords)

        spatial_idx = self.cs.get_index((None,coords.latitude,coords.longitude))

        actual_idx = actual.get_index(slice(split[1][0],split[1][1]))

        full_actual_idx = (actual_idx,spatial_idx[1],spatial_idx[2])

        actual_data = self.get_by_index(full_actual_idx)

        if hasattr(actual_data,'mask'):
            actual_data = actual_data.data

        actual_data[actual_data==self.fillvalue] = np.nan
        
        if split[0] is None and split[2] is None:
            full_data = actual_data.reshape(full_shape)
            if inverted:
                full_data = full_data[:,::-1,:]

            return full_data
        else:
            full_data = np.empty(full_shape,dtype=self.mapped_var.dtype)
            full_data[...] = np.nan

        
        actual_shape = index_shape(full_actual_idx)

        if split[0] is not None:
            dstart = (split[0][1]-split[0][0]).days + 1
        else:
            dstart = 0

        write_index = desimplify(actual_shape)
        write_index[0] = np.s_[dstart:(dstart+actual_shape[0])]

        full_data[write_index] = actual_data.reshape(index_shape(full_actual_idx,True))

        if inverted:
            full_data = full_data[:,::-1,:]
        
        return full_data

    def write_by_coords(self,coords,data):
        write_index = desimplify(data.shape)
        self.accessor.set_by_coords(coords,data[write_index])

from awrams.utils.general import invert_dict

#+++ Put this somewhere more appropriate - in config?
COORD_ALIAS_TO_REP = {'lat': 'latitude', 'lon': 'longitude'}
COORD_ALIAS_TO_NC = invert_dict(COORD_ALIAS_TO_REP)

CONFORM_TO_GRID = 0.001 # +++ Quantize lat/lon to this resolution.  Change to None for no quantization

class DatasetManager:
    '''
    Convenience wrapper for a NetCDF group or dataset
    The group must exist before instantiating the DatasetManager
    '''
    def __init__(self,ncd_group,conform=CONFORM_TO_GRID):
        self.ncd_group = ncd_group
        self._update_dicts()
        self.conform_to_grid = conform

    def close(self):
        self.ncd_group.close()

    def _update_dicts(self):
        self.variables = o()
        self.groups = o()
        self.attrs = o()
        for v in self.ncd_group.variables:
            self.variables[v] = self.ncd_group.variables[v]
        if 'var_name' in self.ncd_group.ncattrs():
            self.awra_var = self.ncd_group.variables[self.ncd_group.getncattr('var_name')]
        else:
            self.awra_var = self._imply_variable()
        for a in self.ncd_group.ncattrs():
            self.attrs[a] = self.ncd_group.getncattr(a)
        for g in self.ncd_group.groups:
            #if g != 'provenance':
            self.groups[g] = DatasetManager(self.ncd_group.groups[g])

    def _imply_variable(self):
        for k, v in self.variables.items():
            if hasattr(v,'dimensions'):
                if not k.endswith('_bounds') and list(v.dimensions) == list(self.ncd_group.dimensions):
                    return v

    def create_coordinates(self, coord):
        '''
        Coord is an AWRAMS Coordinates object;
        This function creates both the dimensions and the coordinate variables
        '''
        coord_len = 0 if coord.unlimited else len(coord)
        self.ncd_group.createDimension(coord.dimension.name,coord_len)
        coord_var = self.ncd_group.createVariable(coord.dimension.name,datatype=coord.dimension.dtype,dimensions=[coord.dimension.name])
        
        dim_attrs = coord.dimension._dump_attrs()

        for k,v in dim_attrs.items():
            set_h5nc_attr(coord_var,k,v)

        #coord_var.setncatts(coord.dimension._dump_attrs())
        # Indices may be native python types, use _persistent_index when writing
        coord_var[:] = coord._persist_index()
        if isinstance(coord,BoundedCoordinates):
            if 'nv' not in self.ncd_group.dimensions:
                self.ncd_group.createDimension('nv',2)
            set_h5nc_attr(coord_var,'bounds',coord.dimension.name+'_bounds')
            #coord_var.setncattr('bounds',coord.dimension.name+'_bounds')
            bounds_var = self.ncd_group.createVariable(coord.dimension.name+'_bounds',datatype='i',dimensions=[coord.dimension.name,'nv'])
            bounds_var[:] = coord._persist_boundaries()
        self._update_dicts()

    def create_variable(self,mapped_var,ncd_params=None,**kwargs):
        '''
        Takes a MappedVariable object, propogates to NetCDF
        Returns the NetCDF variable
        '''

        if ncd_params is None:
            from awrams.utils.io.netcdf_wrapper import NCVariableParameters
            ncd_params = NCVariableParameters()
        ncd_params.update(**kwargs)
        var_dims = [(dim.name) for dim in mapped_var.coordinates.dimensions]

        for dim in var_dims:
            if dim not in list(self.ncd_group.dimensions.keys()):
                self.create_coordinates(mapped_var.coordinates[dim])


        target_var = self.ncd_group.createVariable(mapped_var.variable.name,datatype=mapped_var.dtype,dimensions=var_dims,**ncd_params)
        
        v_attrs = mapped_var.variable._dump_attrs()

        for k,v in v_attrs.items():
            set_h5nc_attr(target_var,k,v)

        #target_var.setncatts(mapped_var.variable._dump_attrs())
        self._update_dicts()
        return target_var

    def create_group(self,group_name):
        '''
        Create a new group within the current group; return
        '''
        self.groups[group_name] = group = self.ncd_group.createGroup(group_name)
        return group

    def get_coords(self):

        coords = []
        for k in list(self.ncd_group.dimensions.keys()):
            if k in list(self.ncd_group.variables.keys()):
                coords.append(self.get_coord(k))
        #coords = [self.get_coord('time'),self.get_coord('latitude'),self.get_coord('longitude')]
        return CoordinateSet(coords)

    def get_coord(self,coord):
        '''
        Get a Coordinates object whose name matches 'coord'
        '''
        #from awrams.utils.io.netcdf_wrapper import epoch_from_nc
        from awrams.utils.geo import GeoArray
        from awrams.utils.datetools import timespecs_from_str
        from awrams.utils.mapping_types import infer_freq

        def from_epoch(epoch,ts,unit):
            return epoch + pd.Timedelta(ts,unit=unit)

        if coord in self.ncd_group.variables:
            coord_nc = coord
        else:
            try:
                coord_nc = COORD_ALIAS_TO_NC[coord]
            except:
                raise Exception("Coordinate not found", coord)

        coord_rep = COORD_ALIAS_TO_REP.get(coord)
        if coord_rep is None:
            coord_rep = coord

        if coord_rep == 'time':
            #epoch = epoch_from_nc(self.ncd_group)
            time_var = self.ncd_group.variables[coord_nc]
            epoch,unit = timespecs_from_str(time_var.units)
            #dti = pd.DatetimeIndex([(epoch + dt.days(int(ts))) for ts in self.ncd_group.variables['time'][:]],freq='d')
            dti = epoch + pd.TimedeltaIndex(time_var[...],unit=unit)
            freq = unit if len(dti) == 1 else infer_freq(dti)
            dti = pd.DatetimeIndex(dti,freq)

            if 'bounds' in time_var.ncattrs():
                bounds_var = self.ncd_group.variables[time_var.bounds]
                boundaries = []
                for b in bounds_var[:]:
                    boundaries.append([from_epoch(epoch,b[0],unit),from_epoch(epoch,b[1]-1,unit)])

                #Attempt to infer period frequency from boundaries...
                p = infer_period(boundaries)

                return BoundedTimeCoordinates(TimeDimension(epoch),dti,boundaries)
            else:
                return TimeCoordinates(TimeDimension(epoch),dti)
        elif coord_rep == 'latitude':
            ncvar = self.ncd_group.variables[coord_nc]
            lat = np.float64(ncvar[:])
            if not hasattr(lat,'__len__'):
                lat = np.array([lat])
            #return Coordinates(latitude,aquantize(lat,0.05)) +++ only here for badly behaved (float32) coords
            if self.conform_to_grid is not None:
                lat = aquantize(lat,self.conform_to_grid)
            lat = GeoArray.from_degrees(lat).to_degrees()
            return Coordinates(latitude,lat)
        elif coord_rep == 'longitude':
            lon = np.float64(self.ncd_group.variables[coord_nc][:])
            if not hasattr(lon,'__len__'):
                lon = np.array([lon])
            if self.conform_to_grid is not None:
                lon = aquantize(lon,self.conform_to_grid)
            lon = GeoArray.from_degrees(lon).to_degrees()
            return Coordinates(longitude,lon)
        else:
            ncvar = self.ncd_group.variables[coord_nc]
            if hasattr(ncvar,'units'):
                units = Units(ncvar.units)
            else:
                units = Units('unknown unit')
            #+++ Can we assume the dimension name is the coord name?
            dim = Dimension(coord_rep,units,ncvar.dtype)
            return Coordinates(dim,ncvar[:])

    def get_mapping_var(self,full_map=False):
        '''
        Return a mapping_types.Variable object from the netCDF information
        full_map will include coordinates and datatype
        '''
        
        ncvar = self.awra_var

        attrs = dict([[k,ncvar.getncattr(k)] for k in ncvar.ncattrs()])
        out_var = Variable(attrs['name'],attrs['units'],attrs)
        if full_map:
            cs = self.get_coords()
            return MappedVariable(out_var,cs,ncvar.dtype)
        else:
            return out_var

    def set_time_coords(self,time_coords,resize=False):
#        logger.info("set_time_coords %s %s",self.ncd_group.variables['time'].shape, len(time_coords))
        if resize or self.ncd_group.variables['time'].shape[0] < len(time_coords):
            if isinstance(self.ncd_group.variables['time'], dbh.h5pyDataset):
                self.ncd_group.variables['time'].resize((len(time_coords),))
                for var in self.variables.values():
                    if hasattr(var,'dimensions'):
                        ncdims = var.dimensions
                        if 'time' in ncdims:
                            dims = var.dims
                            tdim = None
                            for i, dim in enumerate(dims):
                                if 'time' in dim.keys():
                                    tdim = i
                            if tdim is not None:
                                new_shape = list(var.shape)
                                new_shape[tdim] = len(time_coords)
                                var.resize(new_shape)

                #self.awra_var.resize((len(time_coords),self.awra_var.shape[1],self.awra_var.shape[2]))
#                logger.info("set_time_coords (is h5pyDataset) %s",self.awra_var.shape)
            else:
                if 'time' in self.awra_var.dimensions:
                    self.awra_var[len(time_coords) - 1,0,0] = -999.
#                logger.info("set_time_coords (is nc.variable) %s",self.awra_var.shape)
        self.ncd_group.variables['time'][:] = time_coords._persist_index()

        if 'bounds' in self.ncd_group.variables['time'].ncattrs():
            bounds_var = self.ncd_group.variables[self.ncd_group.variables['time'].getncattr('bounds')]
            if isinstance(self.ncd_group.variables['time'], dbh.h5pyDataset):
                bounds_var.resize((len(time_coords),bounds_var.shape[1]))
            bounds_var[:] = time_coords._persist_boundaries()

    def get_dates(self):
        from awrams.utils.io.netcdf_wrapper import epoch_from_nc

        epoch = epoch_from_nc(self.ncd_group)
        dti = pd.DatetimeIndex([(epoch + dt.days(int(ts))) for ts in self.ncd_group.variables['time']],freq='d')
        return dti

    def get_daterange(self):
        from awrams.utils.io.netcdf_wrapper import epoch_from_nc

        epoch = epoch_from_nc(self.ncd_group)
        time = self.ncd_group.variables['time']
        return [(epoch + dt.days(int(ts))) for ts in (time[0],time[len(time)-1])]

    def get_extent(self,use_mask=True):
        #from awrams.utils.extents import from_boundary_coords
        from awrams.utils import geo
        from awrams.utils.extents import Extent
        lat,lon = self.get_coord('latitude'), self.get_coord('longitude')

        if use_mask is True:
            ref_data = self.awra_var[0]
            if not hasattr(ref_data,'mask'):
                try: ### maybe using h5py which doesn't return an ma
                    mask = np.ma.masked_values(ref_data, self.awra_var.attrs['_FillValue'][0])
                    mask = mask.mask
                except AttributeError:
                    mask = False
            else:
                mask = ref_data.mask
        else:
            mask = False

        origin = geo.GeoPoint.from_degrees(lat.index[0],lon.index[0])
        cell_size = geo.GeoUnit.from_dms(lat.index[-1] - lat.index[0]) / (len(lat.index)-1)
        nlats,nlons = len(lat),len(lon)

        lat_orient = 1 if cell_size > geo.GeoUnit(0) else -1
        cell_size = cell_size * lat_orient

        georef = geo.GeoReference(origin,nlats,nlons,cell_size,lat_orient=lat_orient)

        if not isinstance(mask,np.ndarray):
            mask = bool(mask)
 
        return Extent(georef,mask=mask)
        #return from_boundary_coords(lat[0],lon[0],lat[-1],lon[-1],compute_areas = False,mask=mask)

def flattened_cell_file(variable,period,extent,dtype=np.float64,filename=None,chunksize=None,leave_open=True):

    import netCDF4 as ncd

    cell_dim = Dimension('cell',Units('cell_idx'),np.int32)
    cell_c = Coordinates(cell_dim,list(range(extent.cell_count)))
    tc = period_to_tc(period)

    cs = CoordinateSet([cell_c,tc])

    m_var = MappedVariable(variable,cs,dtype)

    if filename is None:
        filename = variable.name + '.nc'

    ds = ncd.Dataset(filename,'w')
    dsm = DatasetManager(ds)

    if chunksize is None:
        chunksize = (1,len(period))

    chunksizes = cs.validate_chunksizes(chunksize)

    from awrams.utils.io.netcdf_wrapper import NCVariableParameters
    ncp = NCVariableParameters(chunksizes=chunksizes)

    dsm.create_variable(m_var,ncp)

    lats,lons = extent._flatten_fields()

    lat_v = MappedVariable(Variable('lats',deg_north),cell_c,np.float64)
    lon_v = MappedVariable(Variable('lons',deg_east),cell_c,np.float64)

    dsm.create_variable(lat_v)[:] = lats
    dsm.create_variable(lon_v)[:] = lons

    if not leave_open:
        dsm.close()
    else:
        return dsm

def build_cell_map(ds):
    locs = list(zip(ds.variables['lats'][:],ds.variables['lons'][:]))
    cell_map = {}
    for i, cell in enumerate(locs):
        cell_map[cell] = i
    return cell_map

def managed_dataset(fn,mode='r',conform=CONFORM_TO_GRID):
    '''
    Open a file with a DatasetManager object
    '''
    if mode == 'w':
        import netCDF4 as ncd
        #from awrams.utils.io.h5netcdf import legacyapi as ncd
        return DatasetManager(ncd.Dataset(fn,mode),conform=conform)
    else:
        return DatasetManager(db_opener(fn,mode),conform=conform)#ncd.Dataset(fn,mode))

def offset_slice(s,offset):
    return slice(s.start+offset,s.stop+offset)

class Segment:
    def __init__(self,coords,start,end):
        self.coords = coords
        self.start = start
        self.end = end
        
    def __len__(self):
        return self.end - self.start + 1

class SegSplit:
    def __init__(self,seg_idx,local_index,global_index):
        self.seg_idx = seg_idx
        self.local_index = local_index
        self.global_index = global_index
        
    def __repr__(self):
        return '%s local: %s global: %s' % (self.seg_idx,self.local_index,self.global_index)

class Splitter(object):
    '''
    Base class for splitting data access based
    on boundaries accross a single dimension of the input
    '''
    def __init__(self,coordinates,subcoords):
        '''
        coordinates: CoordinateSet to be split
        strategy: string identifying splitting strategy
        '''
        self.coordinates = coordinates
        self.dim_pos = None
        self.dim_len = 0
        self._build_segments(subcoords)

    def _build_segments(self,subcoords):
        '''
        Create the segments for a given CoordinateSet
        Each segment will map to a particular set of indices,
        and be uniquely identifiable.
        '''
        self.segments = []
        for sc in subcoords:
            seg_idx = self.coordinates.get_index(sc)
            if isinstance(seg_idx,int):
                seg_idx = slice(seg_idx,seg_idx+1)
            self.segments.append(Segment(sc,seg_idx.start,seg_idx.stop-1))

    def locate_segment(self,index):
        '''
        index is single integer, return the position in the segment array,
        and the segment
        '''
        for i, segment in enumerate(self.segments):
            if index >= segment.start and index <= segment.end:
                return i

        raise IndexError("Index %i out of range" % index)

    def split_coords_to_indices(self,coordinates):
        return self.split_indices(self.coordinates.get_index(coordinates))

    def split_indices(self,indices):
        if isinstance(indices,int):
            seg_start = seg_end = self.locate_segment(indices)
            indices = slice(indices,indices+1)
        else:
            seg_start = self.locate_segment(indices.start)
            seg_end = self.locate_segment(indices.stop-1)
        
        if seg_start == seg_end:
            seg = self.segments[seg_start]
            return [SegSplit(seg_start,slice(indices.start-seg.start,indices.stop-seg.start),indices)]

        out_indices = []
        
        seg = self.segments[seg_start]
        out_indices.append(SegSplit(seg_start,slice(indices.start-seg.start,len(seg)),slice(indices.start,seg.end+1)))
        
        for i in np.arange(seg_start+1,seg_end):
            cur_seg = self.segments[i]
            local_slice = slice(0,len(cur_seg))
            global_slice = slice(cur_seg.start,cur_seg.end+1)
            out_indices.append(SegSplit(i,local_slice,global_slice))
                           
        seg = self.segments[seg_end]
        out_indices.append(SegSplit(seg_end,slice(0,indices.stop-seg.start),slice(seg.start,indices.stop)))
                           
        return out_indices

    def split_data_coords(self,data,coords):
        '''
        For a given block of data whose shape matches the supplied
        coords, return a list of (segment,index,data) dicts, where index
        is localised to the segment
        '''
        return self.split_data(data, self.coordinates.get_index(coords))


    def split_data(self,data,indices):
        '''
        For a given block of data whose shape matches the supplied
        indices, return a list of (segment,index,data) dicts, where index
        is localised to the segment
        '''
        if isinstance(indices,Coordinates):
            indices = self.coordinates.get_index(indices)

        seg_indices = self.split_indices(indices)

        out_data = []

        for seg_i in seg_indices:
            seg_data = data[simplify_indices(seg_i.global_index)]
            out_split = o(segment=seg_i.segment,index=seg_i.local_index,data=seg_data)
            out_data.append(out_split)

        return out_data

class SplitSchema:
    @classmethod
    def gen_file_name(self,variable,period):
        '''
        Return a file name (without extension) for the supplied variable and period
        variable: awrams.utils.mapping_types.Variable
        period: pandas.DatetimeIndex
        '''
        raise NotImplementedError()
        
    @classmethod
    def split_periods(self,period):
        '''
        Return a list of (split) periods from the supplied period
        period: pandas.DatetimeIndex
        '''
        raise NotImplementedError()

class AnnualSplitSchema(SplitSchema):
    @classmethod
    def gen_file_name(self,variable,period):
        year = np.unique(period.year)
        if len(year) != 1:
            raise Exception("Invalid period for annual split")
        return variable.name+'_'+str(year[0])
        
    @classmethod
    def split_periods(self,period):
        return dt.split_period(period,'a')

class MonthlySplitSchema(SplitSchema):
    @classmethod
    def gen_file_name(self,variable,period):
        year = np.unique(period.year)
        month = np.unique(period.month)
        if len(year) != 1 or len(month) != 1:
            raise Exception("Invalid period for monthly split")
        
        mstr = str(month[0])
        mstr = mstr.zfill(2)
            
        return variable.name+'_'+str(year[0])+'_'+mstr
        
    @classmethod
    def split_periods(self,period):
        return dt.split_period(period,'m')
    
class FlatFileSchema(SplitSchema):
    @classmethod
    def gen_file_name(self,variable,period):
        return variable.name
        
    @classmethod
    def split_periods(self,period):
        return [period] 