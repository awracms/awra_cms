import numpy as np
from collections import OrderedDict
from awrams.utils.mapping_types import gen_coordset
from awrams.utils import datetools as dt
from awrams.utils import extents

class Domain:
    def __init__(self,coords):
        self.coords = coords
    
    @property
    def dims(self):
        return list(self.coords)
    
    def is_virtual(self):
        for k,v in self.coords.items():
            if v.is_virtual():
                return True
        return False
    
    def realise(self,**kwargs):
        new_coords = OrderedDict()
        for cname,coord in self.coords.items():
            c_kwargs = dict([(k,kwargs[k]) for k in kwargs if k in coord.aspects])
            new_coords[cname] = coord.realise(**c_kwargs)
        return Domain(new_coords)
    
    def subdivide(self,coord,aspect,method,*args,**kwargs):
        dm = self.coords[coord]
        new_dm = subdivide_domain_map(dm,aspect,method,*args,**kwargs)
        new_coords = self.coords.copy()
        new_coords[coord] = new_dm
        return Domain(new_coords)
    
    def shape(self,flat):
        out_shape = []
        for v in self.coords.values():
            out_shape += list(v.shape(flat))
        return tuple(out_shape)

class DomainMap:
    def __init__(self,mapping,aspects=None):
        if aspects is None:
            aspects = []
        self.aspects = aspects
        self.mapping = mapping
    
    def __getitem__(self,k):
        return self.mapping[k]
    
    def is_virtual(self):
        return len(self.aspects) > 0
    
    def realise(self,**kwargs):
        index = [slice(None,None) for i in self.mapping.shape]
        new_aspects = self.aspects.copy()
        for k,v in kwargs.items():
            index[self.aspects.index(k)] = v
            new_aspects.remove(k)
        new_mapping = self.mapping[tuple(index)]
        return DomainMap(new_mapping,new_aspects)
    
    def shape(self,flat=True):
        if self.is_virtual():
            raise Exception("Calling shape on virtual DomainMap")
        return self.mapping.shape(flat)            

class DomainMappedCoordTime:
    @property
    def value_obj(self):
        return self.dti
    
class DomainMappedDatetimeIndex(DomainMappedCoordTime):
    def __init__(self,dti):
        self.dti = dti
        
    def shape(self,flat=True):
        return (len(self.dti),)
        
    def __repr__(self):
        return self.dti.__repr__()
    
class DomainMappedCoordSpatial:
    @property
    def value_obj(self):
        return self.extent
    
class DomainMappedExtent(DomainMappedCoordSpatial):
    def __init__(self,extent):
        self.extent = extent
        
    def __repr__(self):
        return self.extent.__repr__()
    
    def shape(self,flat=True):
        if flat:
            return (self.extent.cell_count,)
        else:
            return self.extent.shape
        
class DomainMappedFlatIndex(DomainMappedCoordSpatial):
    def __init__(self,findex):
        self.findex = findex
        
    @property
    def extent(self):
        return self.findex.as_extent()
    
    def __repr__(self):
        return self.findex.__repr__()
    
    def shape(self,flat=True):
        if flat:
            return self.findex.shape
        else:
            return self.extent.shape


class FlatIndex:
    def __init__(self,ref,start,end):
        self.ref = ref
        self.start = start
        self.end = end
        
    @property
    def shape(self):
        return (self.end - self.start + 1,)
    
    def get_relative_index(self,other):
        return slice(self.start,self.end + 1)
    
    def _as_extent(self):
        cell_map = np.arange(np.prod(self.ref.shape))
        #cell_map.shape = self.ref.shape
        #cell_map_masked = np.ma.MaskedArray(cell_map,self.ref.mask)
        cell_map_flat = cell_map[~self.ref.mask.flatten()]
        ncols = self.ref.shape[1]
        
        start_cell,end_cell = cell_map_flat[self.start],cell_map_flat[self.end]
        
        start_row,start_ml = start_cell//ncols,start_cell%ncols
        end_row,end_ml = end_cell//ncols,end_cell%ncols
        
        subex = self.ref.ioffset[start_row:(end_row+1),:].copy()
        subex.mask[0,:start_ml] = True
        subex.mask[-1,end_ml+1:] = True
        
        return subex
    
    def as_extent(self):
        cell_map = np.arange(np.prod(self.ref.shape))
        #cell_map.shape = self.ref.shape
        #cell_map_masked = np.ma.MaskedArray(cell_map,self.ref.mask)
        cell_map_flat = cell_map[~self.ref.mask.flatten()]
        ncols = self.ref.shape[1]

        start_cell,end_cell = cell_map_flat[self.start],cell_map_flat[self.end]

        start_row,start_ml = start_cell//ncols,start_cell%ncols
        end_row,end_ml = end_cell//ncols,end_cell%ncols

        outex = self.ref.copy()

        outex.mask[0:start_row,:] = True
        outex.mask[start_row,:start_ml] = True
        outex.mask[end_row+1:,:] = True
        outex.mask[end_row,end_ml+1:] = True

        return outex
    
    def __repr__(self):
        return 'SubIdx: [%s:%s]\nRef: %s' % (self.start,self.end,self.ref.__repr__())
        
def subdivide_domain_map(dom_map,aspect,method,*args,**kwargs):
    new_aspects = dom_map.aspects + [aspect]
    cur_map = {}
    new_len = None
    for idx, x in np.ndenumerate(dom_map.mapping):
        cur_map[idx] = v = method(x.value_obj,*args,**kwargs)
        if new_len is None:
            new_len = len(v)
        else:
            if len(v) != new_len:
                raise Exception('Cannot subdivide domain into unequal parts')
    if not dom_map.is_virtual():
        new_shape = [new_len]
    else:
        new_shape = list(dom_map.mapping.shape) + [new_len]
    new_mapping = np.empty(new_shape,dtype=object)
    for k,v in cur_map.items():
        new_mapping[k] = np.array(v)
    return DomainMap(new_mapping,new_aspects)

def coordset_from_domain(in_domain):
    cs = gen_coordset(in_domain.coords['time'].mapping.value_obj,in_domain.coords['latlon'].mapping.value_obj)
    mask = in_domain.coords['latlon'].mapping.extent.mask
    return cs,mask

def task_layout_from_dom(in_dom):
    ntasks,nsubtasks = None,None
    for k,v in in_dom.coords.items():
        if 'task' in v.aspects:
            ntasks = v.mapping.shape[v.aspects.index('task')]
        if 'subtask' in v.aspects:
            nsubtasks = v.mapping.shape[v.aspects.index('subtask')]
    return ntasks,nsubtasks

def split_period_annual_chunked(in_dti,chunksize):
    out_a= dt.split_period(in_dti,'a')
    all_out = []
    for dti in out_a:
        all_out = all_out + [DomainMappedDatetimeIndex(d) for d in dt.split_period_chunks(dti,chunksize)]
    return all_out

'''
Methods for subdividing extents; used in domain decomposition
'''

def subdivide_extent_equal_cells_flatindex(extent,n_out):
    full_cell_count = extent.cell_count
    base = int(full_cell_count/n_out)
    rem = full_cell_count%n_out
    cell_counts = [base + 1 for i in range(rem)] + [base for range in range(n_out - rem)]
    
    start_index = 0
    out_extents = []
    
    for cc in cell_counts:
        out_extents.append(DomainMappedFlatIndex(FlatIndex(extent,start_index,start_index+cc-1)))
        start_index += cc
    
    return out_extents

def subdivide_extent_chunked(extent, chunk, min_cells):
    extents_out = extents.subdivide_extent_chunked(extent,chunk,min_cells)
    dme_out = [DomainMappedExtent(e) for e in extents_out]
    return dme_out
