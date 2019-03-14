import numpy as np
import os
from os.path import join

from awrams.utils import config_manager
import awrams.utils.extents as extents
from awrams.utils import geo

from numpy.testing import assert_allclose
from nose.tools import nottest, assert_almost_equal

SYS_PROFILE = config_manager.get_system_profile().get_settings()
MASK_FILE = join(SYS_PROFILE['DATA_PATHS']['MASKS'],'web_mask_v5.h5')

@nottest
def build_mock_array(extent):

    data = np.zeros(shape=(extent.shape))
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            data[i,j] = i * data.shape[0] + j
    return data

# @nottest
def test_extent_all():
    e = extents.Extent.from_file(MASK_FILE)

    assert(e.shape == (681,841))
    assert(e.cell_count == 281655)
    assert(e.cell_size == 0.05)

# @nottest
def test_cell():
    e = extents.Extent.from_file(MASK_FILE)

    c = e.factory.get_by_cell_offset(300,400)
    assert(c.shape == (1,1))
    assert(c.indices[0] == slice(300, 301, None) and c.indices[1]== slice(400, 401, None))
    assert c.parent_ref == e.parent_ref
    assert not c.origin == e.origin

    ### test itercells, which generates cells from local mask, honours parent_ref cell indices
    cells = [cell for cell in c.itercells(False)]
    assert len(cells) == 1
    assert cells[0] == (300,400)

    cells = [cell for cell in c.itercells(True)]
    assert len(cells) == 1
    assert cells[0] == (0,0)

    ### translate
    c_t = c.translate_localise_origin()
    assert(c_t.indices[0] == slice(0, 1, None) and c_t.indices[1]== slice(0, 1, None))
    assert c_t.parent_ref.origin == c_t.origin


# @nottest
def test_bb_translate_mask():
    e = extents.Extent.from_file(MASK_FILE)

    bb = e.ioffset[100:400,100:400]

    bb_t = bb.translate_localise_origin()

    assert (bb.mask == bb_t.mask).all()

# @nottest
def test_translate_subdata():
    e = extents.Extent.from_file(MASK_FILE)
    mock_array = build_mock_array(e)    

    bb = e.ioffset[100:400,100:400]

    sub_data = mock_array[bb.indices]

    assert (sub_data == mock_array[100:400,100:400]).all()

    bb1 = bb.ioffset[20:30,20:30].translate(bb.geospatial_reference())

    print(bb1.indices)  

    assert(sub_data[bb1.indices] == mock_array[120:130,120:130]).all()

def test_translate_orient():
    cell_size = 0.05
    o0 = geo.GeoPoint.from_degrees(-12.0,112.0)
    o1 = geo.GeoPoint.from_degrees(-12.15,112.0)
    o2 = geo.GeoPoint.from_degrees(-13.0,112.0)
    g0 = geo.GeoReference(o0,20,20,cell_size,lat_orient=-1)
    g1 = geo.GeoReference(o1,4,5,cell_size,lat_orient=-1)
    g2 = geo.GeoReference(o2,18,10,cell_size,lat_orient=1)

    e0 = extents.Extent(g0)
    e1 = extents.Extent(g1)
    e2 = extents.Extent(g2)

    e1t = e1.translate(e0.geospatial_reference())

    assert((e1t.latitudes == e1.latitudes).all())
    assert(e1.origin == e1t.origin)
    assert(e1t.geospatial_reference() == e1.geospatial_reference())
    assert((e1t.parent_ref.origin - e1.parent_ref.origin).lat == 0.15)

    e2.set_mask(np.tri(*e2.shape).astype(bool))
    e2c = e2.translate(g0)

    assert((e2.areas == e2c.areas[::-1,:]).all())
    assert((e2c.latitudes.to_degrees()[::-1] == e2.latitudes.to_degrees()).all())


@nottest
def test_multiextent():
    gsd = extents.Extent.from_file(MASK_FILE)

    full = extents.ExtentFactory(extents.get_default_extent())


    m = dict(a=full.get_by_cell_offset(250,250),
             b=gsd.extent_from_cell_offset(275,275),
             c=gsd.extent_from_cell_offset(300,300),
             d=gsd.extent_from_boundary_offset(200,290,202,292))
    e = gsd.extent_from_multiple(m)
    # print(e.mask.shape)
    # print(e.mask[49:52,:2])
    # print(e.mask[:4,40:44])

    for k,cell in m.items():
        # print(k,cell)
        assert(e.contains(cell))

    # for cell in e:
    #     print("LLL",cell)
    #
    assert e.x_min == 250
    assert e.x_max == 300
    assert e.y_min == 200
    assert e.y_max == 300

    assert e.cell_count == 12

    # assert False

#@nottest
def test_area_with_osr():
#    gsd = extents.Extent.from_file(_MASK_FILE)
    full = extents.ExtentFactory(extents.Extent.from_file(MASK_FILE))

    try:
        c = full.get_by_cell_offset(300,200)
        assert_allclose(c.area, 27956330.541286)
        print("300,200",c.area)

        c = full.get_by_cell_offset(300,400)
        assert_allclose(c.area, 27956330.541286)
        print("300,400",c.area)

    except ImportError:
        pass

    try:
        import osr

        c = full.get_by_boundary_offset(300,200,3,2)

        assert_almost_equal(c.area,167671123.116,places=3)

        s = 0
        for cell in c.itercells(local=False):
            cc = full.get_by_cell_offset(*cell)
            s += cc.area
        assert s == c.area

    except ImportError:
        pass
    # assert False

@nottest
def test_area_without_osr():
    import awrams.utils.extents
    awrams.utils.extents._LONGLAT_TO_AEA = None

    #gsd = extents.Extent.from_file(_MASK_FILE)
    full = extents.ExtentFactory(extents.Extent.from_file(MASK_FILE))

    #c = gsd.extent_from_cell_offset(300,200)
    c = full.get_by_cell_offset(300,200)
    #c.compute_areas()
    # assert_almost_equal(c.area,27956330.541281,places=6) # with gdal
    assert_almost_equal(c.area,28044093.890163,places=6)

    #c = gsd.extent_from_boundary_offset(300,200,301,201)
    c = full.get_by_boundary_offset(300,200)
    #c.compute_areas()
    # assert_almost_equal(c.area,111803049.532,places=3) # with gdal
    assert_almost_equal(c.area,112153279.927,places=3)

    # assert False
