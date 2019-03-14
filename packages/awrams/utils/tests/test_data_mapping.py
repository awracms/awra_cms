import os
import shutil
from nose.tools import with_setup, nottest
import awrams.utils.mapping_types as mt
import awrams.utils.datetools as dt
from awrams.utils import extents, geo
from awrams.utils.io import data_mapping as dm
from awrams.utils import config_manager
import numpy as np

def setup_var_coords():
    global m_tvar
    global extent
    global test_path

    georef = geo.GeoReference((0,0),1000,1000,0.05)
    extent = extents.Extent(georef).ioffset[0:10,0:10]

    period = dt.dates('dec 2000 - jan 25 2001')
    tvar = mt.Variable('test_var','mm')
    
    m_tvar = mt.MappedVariable(tvar,mt.gen_coordset(period,extent),np.float32)

    test_path = os.path.join(os.path.dirname(__file__),'file_tests')

    shutil.rmtree(test_path,True)
    #os.makedirs(test_path)

def teardown_var_coords():
    global m_tvar
    global extent
    global test_path

    m_tvar = None
    extent = None
    shutil.rmtree(test_path,True)

def test_get_padded_by_coords():
    from awrams.utils.io.data_mapping import SplitFileManager
    from awrams.utils.mapping_types import gen_coordset
    import awrams.utils.datetools as dt

    data_paths = config_manager.get_system_profile().get_settings()['DATA_PATHS']

    path = os.path.join(data_paths['BASE_DATA'],'test_data','simulation','climate','temp_min_day')

    sfm = SplitFileManager.open_existing(path,'temp_min_day_*.nc','temp_min_day')
    # return sfm
    extent = sfm.get_extent().ioffset[200:230,200:230]
    period = dt.dates('2011')
    coords = gen_coordset(period,extent)

    data = sfm.get_padded_by_coords(coords)
    assert data.shape == coords.shape

@nottest
def run_schema_test(schema):
    new_sfm = dm.SplitFileManager(test_path,m_tvar)
    new_sfm.create_files(schema,clobber=True,leave_open=True)
    
    data = new_sfm.get_padded_by_coords(new_sfm.cs)
    assert(np.isnan(data).all())

    newdata = np.random.normal(size=data.shape).astype(np.float32)
    new_sfm.set_by_coords(new_sfm.cs,newdata)

    data = new_sfm.get_padded_by_coords(new_sfm.cs)
    assert((data==newdata).all())

    subcs = mt.gen_coordset(dt.dates('dec 9 2000 - jan 15 2001'),extent.ioffset[5,2])

    newdata = np.random.normal(size=subcs.shape).astype(np.float32)
    new_sfm.set_by_coords(subcs,newdata)
    assert((new_sfm.get_padded_by_coords(subcs) == newdata).all())
    assert((new_sfm.get_padded_by_coords(new_sfm.cs)[new_sfm.cs.get_index(subcs)] == newdata.reshape(dm.simple_shape(newdata.shape))).all())    

    subcs = mt.gen_coordset(dt.dates('dec 12 2000'),extent.ioffset[5,2:4])

    newdata = np.random.normal(size=subcs.shape).astype(np.float32)
    new_sfm.set_by_coords(subcs,newdata)
    assert((new_sfm.get_padded_by_coords(subcs) == newdata).all())
    assert((new_sfm.get_padded_by_coords(new_sfm.cs)[new_sfm.cs.get_index(subcs)] == newdata.reshape(dm.simple_shape(newdata.shape))).all())    

@with_setup(setup_var_coords,teardown_var_coords)
def test_create_flat():
    schema = dm.FlatFileSchema
    run_schema_test(schema)

@with_setup(setup_var_coords,teardown_var_coords)
def test_create_annual():
    schema = dm.AnnualSplitSchema
    run_schema_test(schema)

if __name__ == '__main__':
    test_get_padded_by_coords()