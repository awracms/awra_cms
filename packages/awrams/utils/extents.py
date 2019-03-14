import numpy as np
from awrams.utils import geo
from awrams.utils.helpers import Indexer as _ix
from awrams.utils.helpers import index
from awrams.utils.geo import get_geounit, get_geopoint
from copy import deepcopy
from awrams.utils.awrams_log import get_module_logger
from awrams.utils import config_manager
import os

logger = get_module_logger('awrams.utils.extents')
import warnings


class Extent:
    def __init__(self,parent_ref,lat_offset=0,lon_offset=0,nlats=None,nlons=None,mask=False,areas=None,area_sum=None):
        if nlats is None:
            nlats = parent_ref.nlats
        if nlons is None:
            nlons = parent_ref.nlons

        if mask is not False:
            if (nlats,nlons) != mask.shape:
                raise Exception("Mask does not match supplied shape")

        self.parent_ref = parent_ref
        self._mask = mask
        self.nlats = nlats
        self.nlons = nlons
        self.lat_offset = lat_offset
        self.lon_offset = lon_offset
        self.ioffset = _ix(self._get_by_offset)
        self.icoords = _ix(self._get_by_coords)

        if areas is not None:
            self.set_areas(areas,area_sum)
        else:
            self._areas = None
            self._area = None

    def set_mask(self,mask):
        self._mask = mask

    def update_area_mask(self):
        '''
        Ensure areas share the same mask as the extent (masked cells set to 0.0)
        '''
        if self._areas is not None:
            self._areas[self.mask] = 0.0
            areas = np.ma.MaskedArray(self._areas,self.mask)
            self.set_areas(areas)

    def set_areas(self,areas,area_sum=None):
        self._areas = areas
        if area_sum is None:
            area_sum = areas.sum()
        self._area = area_sum
        
    @property 
    def mask(self):
        if self._mask is False:
            self._mask = create_mask(self.nlats,self.nlons)
        return self._mask

    @property
    def areas(self):
        if self._areas is None:
            from awrams.utils.gis import calculate_areas
            self._areas = calculate_areas(self)
            self._area = self._areas.sum()
        return self._areas

    @property
    def area(self):
        if self._area is None:
            _ = self.areas
        return self._area

    def _get_by_coords(self,idx):
        r = self.parent_ref.to_mode(geo.GeoReferenceMode.CORNER)
        
        def _get(x,r,i,op):
            #+++ Bit of a hack, need to refactor offset/orient etc to allow indexing rather than just names...
            if i == 0:
                return int(op((get_geounit(x) - r.origin.lat)*r.lat_orient / r.cell_size - self.lat_offset))
            else:
                return int(op((get_geounit(x) - r.origin.lon)*r.lon_orient / r.cell_size - self.lon_offset))

        out_idx = []
        for i, dim in enumerate(idx):
            if isinstance(dim,slice):
                s = 0 if dim.start is None else _get(dim.start,r,i,np.floor)
                if dim.stop is None:
                    e = None
                else:
                    e = _get(dim.stop,r,i,np.ceil)
                    if e <= s:
                        raise Exception("Invalid slice order",dim)
                out_idx.append(slice(s,e))
            else:
                out_idx.append(_get(dim,r,i,np.floor))
        return self._get_by_offset(tuple(out_idx))
                
    def _get_by_offset(self,idx):
        offsets = (self.lat_offset,self.lon_offset)
        spec = []
        for i, dim in enumerate(idx):
            if isinstance(dim,slice):
                s = 0 if dim.start is None else dim.start
                if s < 0: #
                    s = self.shape[i] + s #
                if dim.stop is None:
                    n = self.shape[i] - s
                elif dim.stop < 0:
                    n = self.shape[i] - s + dim.stop
                else:
                    n = dim.stop - s
            else:
                s = dim
                n = 1
                if s < 0:
                    s = self.shape[i] + s
            s = s + offsets[i]
            spec.append((s,n))
        out_mask = self.mask[idx].reshape(spec[0][1],spec[1][1])
        if self._areas is not None:
            out_areas = self.areas[idx].reshape(spec[0][1],spec[1][1])
        else:
            out_areas = None
        return Extent(self.parent_ref,spec[0][0],spec[1][0],spec[0][1],spec[1][1],out_mask,out_areas)

    @property
    def factory(self):
        return ExtentFactory(self)          
        
    @property
    def lat_index(self):
        return slice(self.lat_offset,self.lat_offset+self.nlats)

    @property
    def lon_index(self):
        return slice(self.lon_offset,self.lon_offset+self.nlons)

    @property
    def shape(self):
        return self.nlats,self.nlons

    @property
    def indices(self):
        '''
        Return data indices, latitude first
        '''
        return self.lat_index,self.lon_index

    @property
    def cell_count(self):
        return (~self.mask).sum()
        
    @property
    def cell_size(self):
        return self.parent_ref.cell_size
    
    @property
    def origin(self):
        return self.parent_ref.geo_for_cell(self.lat_offset,self.lon_offset)

    def trimmed(self):
        '''
        Return an equivalent extent trimmed to mask edges
        '''
        m_idx = np.where(self.mask==False)
        return self.ioffset[m_idx[0].min():m_idx[0].max()+1,m_idx[1].min():m_idx[1].max()+1]

    def copy(self):
        return deepcopy(self)

    def _translate(self,georef):
        '''
        Return an extent equivalent to <self>, but expressed in the coordinate system of <georef>
        '''
        #+++
        #How should we handle extents whose georef has a different orientation?
        #Inverting coordinates is not strictly translation, but might be expected nonetheless?
        #This is not handled at all right now...

        offset = self.parent_ref.get_offset(georef)

        if self.parent_ref.lat_orient == georef.lat_orient:
            new_extent = deepcopy(self)
        else:
            new_extent = Extent(georef,self.lat_offset - offset[0])
        
        
        
        new_extent.parent_ref = georef
        new_extent.lat_offset -= offset[0]
        new_extent.lon_offset -= offset[1]
        
        return new_extent

    def translate(self,georef,copy_data=True):
        '''
        Return an extent equivalent to <self>, but expressed in the coordinate system of <georef>
        '''
        #+++
        #Inverting coordinates is not strictly translation, but is most likely what is expected here...

        offset = georef.get_offset(self.geospatial_reference())

        if copy_data:
            transmit = deepcopy
        else:
            transmit = lambda x: x

        mask = transmit(self._mask)
        areas = transmit(self._areas)
        area_sum = transmit(self._area)

        if self.parent_ref.lat_orient != georef.lat_orient:
            if isinstance(mask,np.ndarray):
                mask = mask[::-1,:]
            if isinstance(areas,np.ndarray):
                areas = areas[::-1,:]

        new_extent = Extent(georef,offset[0],offset[1],self.nlats,self.nlons,mask,areas,area_sum)

        return new_extent
    
    def geospatial_reference(self):
        origin = self.parent_ref.geo_for_cell(self.lat_offset,self.lon_offset)
        return geo.GeoReference(origin,self.nlats,self.nlons,self.cell_size,self.parent_ref.lat_orient,self.parent_ref.lon_orient)
    
    def translate_localise_origin(self):
        return self.translate(self.geospatial_reference())
    
    @property
    def latitudes(self):
        return self.origin.lat + (self.cell_size * \
                  (np.arange(self.nlats) * self.parent_ref.lat_orient))
    @property
    def longitudes(self):
        return self.origin.lon + (self.cell_size * \
                  (np.arange(self.nlons) * self.parent_ref.lon_orient))

    def __repr__(self):
        return "origin: %s, shape: %s, cell_size: %s" % (self.origin, self.shape, self.cell_size)

    def to_coords(self):
        from awrams.utils import mapping_types as _mt
        return _mt.latlon_to_coords(self.latitudes.to_degrees(),self.longitudes.to_degrees())

    def bounding_rect(self,as_degrees=True):
        return self.geospatial_reference().bounding_rect(as_degrees)

    @classmethod
    def from_file(self,fn,load_mask=True):
        '''
        Build a GeospatialDomain from the supplied file.
        Will infer information where possible, but can override cellsize in the case of questionable data
        '''
        from os.path import splitext

        ext = splitext(fn)[1]
        if ext == '.flt':
            gsd = geo.load_georef_flt(fn,load_mask)
        elif ext == '.h5':
            gsd = geo.load_georef_h5mask(fn,load_mask)
        elif ext == '.nc':
            gsd = geo.load_georef_nc(fn,load_mask)
            
        return Extent(gsd[0],mask=gsd[1])

    def itercells(self,local=True):
        '''
        Iterate tuples of cell indices for unmasked cells in this extent
        If local is False, cell indices will include the offset from the extent's parent_ref
        '''
        lats,lons = np.where(self.mask == False)
        if not local:
            lats = lats + self.lat_offset
            lons = lons + self.lon_offset
        for i in range(len(lats)):
            yield lats[i],lons[i]

        
class ExtentFactory:
    def __init__(self,extent):
        self.full_extent = extent
        
    def get_by_cell_offset(self,x,y):
        return self.full_extent.ioffset[x,y]
    
    def get_by_cell_coords(self,lat,lon):
        return self.full_extent.icoords[lat,lon]
    
    def get_by_boundary_coords(self,lat_min,lat_max,lon_min,lon_max):
        return self.full_extent.icoords[lat_min:lat_max,lon_min:lon_max]
    
    def get_by_boundary_offset(self,xorigin,yorigin,xsize,ysize):
        return self.full_extent.ioffset[xorigin:xorigin+xsize,yorigin:yorigin+ysize]

    def get_from_multiple(self,source_extents):
        m = np.empty(self.full_extent.shape,dtype=bool)
        m[...] = True
        georef = self.full_extent.geospatial_reference()
        for s in source_extents:
            l_ex = s.translate(georef)
            m[l_ex.indices][l_ex.mask==False] = False
        return Extent(georef,mask=m)

def get_default_extent():
    warnings.warn("Pending deprecation: create extents from files instead",PendingDeprecationWarning)

    sys_settings = config_manager.get_system_profile().get_settings()

    full_mask_filename = os.path.join(sys_settings['DATA_PATHS']['MASKS'],sys_settings['DEFAULT_MASK_FILE'])
    return Extent.from_file(full_mask_filename)

def create_mask(nlats,nlons):
    return np.zeros((nlats,nlons),dtype=bool)

def from_latlons(lats,lons,mask=False):
    return Extent(geo.latlon_to_georef(lats,lons),mask=mask)

def get_from_multiple(source_extents):
    '''
    Get an extent containing all the cells of source_extents
    +++ This assumes all source extents contain compatible georefs with the same orientation/cell_size etc
    It needs a bit of a tidy-up!
    '''
    egr = source_extents[0].geospatial_reference()
    offsets = [egr.get_offset(e.geospatial_reference()) for e in source_extents]
    ncells = [(e.nlats,e.nlons) for e in source_extents]
    lat_offsets = np.array(offsets)[:,0]
    lon_offsets = np.array(offsets)[:,1]
    nlats = np.array(ncells)[:,0]
    nlons = np.array(ncells)[:,1]
    min_lat_offset = np.min(lat_offsets)
    min_lon_offset = np.min(lon_offsets)
    shape = np.max(lat_offsets+nlats)-np.min(lat_offsets),np.max(lon_offsets+nlons)-np.min(lon_offsets)
    m = np.empty(shape,dtype=bool)
    m[...] = True
    full_origin = egr.origin + geo.GeoPoint(egr.cell_size * min_lat_offset * egr.lat_orient, egr.cell_size * min_lon_offset * egr.lon_orient)
    georef = geo.GeoReference(full_origin,shape[0],shape[1],egr.cell_size,egr.lat_orient,egr.lon_orient)
    for s in source_extents:
        l_ex = s.translate(georef)
        m[l_ex.indices][l_ex.mask==False] = False
    return Extent(georef,mask=m)

'''
Methods for subdividing extents
'''

def subdivide_extent_chunked(extent,chunk,min_cells):
    extents_split0 = subdivide_extent_along_axis(extent,chunk,min_cells,0)
    extents_out = []
    for e in extents_split0:
        extents_out += subdivide_extent_along_axis(e,chunk,min_cells,1)
    extents_out_trimmed = [e.trimmed() for e in extents_out]
    return extents_out_trimmed

def subdivide_extent_along_axis(extent,chunk,min_cells,axis=0):
    e_out = []
    cur_extent, rem_extent = None,extent
    
    while(rem_extent is not None):
        cur_extent, rem_extent = split_extent_along_axis(rem_extent, chunk, min_cells, axis)
        e_out.append(cur_extent)
    
    return e_out

def split_extent_along_axis(extent, chunk, min_cells, axis):
    
    if extent.cell_count <= min_cells:
        raise Exception("Extent has fewer than min_cells")
    
    cur_cells = -1
    nblocks = 0
    
    while cur_cells <= min_cells:
        nblocks += 1
        cur_idx = [slice(0,min(chunk*nblocks,extent.shape[axis])),slice(None,None)]
        if axis == 1:
            cur_idx.reverse()
        
        cur_extent = extent.ioffset[cur_idx]
        cur_cells = cur_extent.cell_count
        
    rem_idx = [slice(min(chunk*nblocks,extent.shape[axis]),None),slice(None,None)]
    if axis == 1:
            rem_idx.reverse()
            
    rem_extent = extent.ioffset[rem_idx]
 
    if rem_extent.cell_count < min_cells:
        cur_extent = extent
        rem_extent = None
    return (cur_extent,rem_extent)

def subdivide_extent_equal_cells_flat(extent,n_out):
    '''
    Return n_out subextents with (as close as possible) equal number of cells,
    cells being divided based on the flattened extent
    '''
    full_cell_count = extent.cell_count
    base = int(full_cell_count/n_out)
    rem = full_cell_count%n_out
    cell_counts = [base + 1 for i in range(rem)] + [base for range in range(n_out - rem)]
    
    cell_map = np.arange(np.prod(extent.shape))
    cell_map.shape = extent.shape
    cell_map_masked = np.ma.MaskedArray(cell_map,extent.mask)
    cell_map_flat = cell_map_masked[~cell_map_masked.mask].flatten()
    
    start_idx = 0
    out_extents = []
    
    for cc in cell_counts:
        start_cell = cell_map_flat[start_idx]
        end_cell = cell_map_flat[start_idx+cc-1]
        boundaries = np.where(cell_map == start_cell),np.where(cell_map == end_cell)
        start_row, end_row = boundaries[0][0][0],boundaries[1][0][0]
        masklims = boundaries[0][1][0],boundaries[1][1][0]
        subex = extent.ioffset[start_row:(end_row+1),:].copy()
        subex.mask[0,:masklims[0]] = True
        subex.mask[-1,masklims[1]+1:] = True
        out_extents.append(subex)
        start_idx += cc
    return out_extents

def subdivide_extent(extent,max_cells):
    '''
    Subdivide extent into squares of at most max_cells*max_cells
    '''


    lat_divs, lat_rem = extent.nlats // max_cells, extent.nlats % max_cells
    lon_divs, lon_rem = extent.nlons // max_cells, extent.nlons % max_cells

    if lat_rem:
        lat_divs += 1
    if lon_rem:
        lon_divs += 1

    out = []

    l_ex = extent.translate_localise_origin()

    for lat in range(lat_divs):
        for lon in range(lon_divs):
            if lat == lat_divs-1:
                lat_idx = index[lat * max_cells:]
            else:
                lat_idx = index[lat * max_cells:(lat+1)*max_cells]
            if lon == lon_divs-1:
                lon_idx = index[lon * max_cells:]
            else:
                lon_idx = index[lon * max_cells:(lon+1)*max_cells]
            e = l_ex.ioffset[lat_idx,lon_idx]
            if e.cell_count > 0:
                out.append(e)

    return out

def split_extent(extent,ncells,start_cell=0):
    '''
    Return a subextent of extent, consisting of ncells, beginning at start_cell
    +++ Soon to be deprecated during calibration refactor
    '''
    outextent = extent.copy()

    outmask = np.ones(outextent.shape,dtype=bool)

    i = 0

    for c in extent.itercells():
        
        if i >= start_cell:
            outmask[c] = False
        
        i = i + 1
        
        if i == ncells + start_cell:
            break

    outextent.set_mask(outmask)
    outextent.update_area_mask()

    return outextent.trimmed()
