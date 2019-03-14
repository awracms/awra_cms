from nose.tools import nottest, with_setup
import os

def clear_files():
    import glob
    files = glob.glob(os.path.join(os.path.dirname(__file__),'*.nc'))
    for f in files:
        os.remove(f)

def setup():
    clear_files()
    from awrams.models.awral import model

    from awrams.utils import config_manager

    sys_settings = config_manager.get_system_profile().get_settings()
    model_profile = config_manager.get_model_profile('awral','v6_default')

    model_settings = model_profile.get_settings()
    model_settings['CLIMATE_DATASET'] = sys_settings['CLIMATE_DATASETS']['TESTING']
    
    global awral, input_map

    awral = model_profile.get_model(model_settings)
    input_map = model_profile.get_input_mapping(model_settings)


def tear_down():
    clear_files()

def climate_mod(input_map):
    input_map['precip_f'].args.pattern = "rain*"
    input_map['tmin_f'].args.pattern = "temp_min*"
    input_map['tmax_f'].args.pattern = "temp_max*"
    input_map['solar_f'].args.pattern = "solar*"
    input_map['wind_f'].args.pattern = "wind*"

# @nottest
@with_setup(setup,tear_down)
def test_SplitFileWriterNode():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    extent = extents.get_default_extent()


    from awrams.utils.nodegraph import nodes
    from awrams.simulation.ondemand import OnDemandSimulator

    #input_map = awral.get_default_mapping()

    from awrams.utils.nodegraph import nodes
    from awrams.utils.metatypes import ObjectDict

    # output_path = './'
    output_map = awral.get_output_mapping()
    output_map['qtot_save'] = nodes.write_to_annual_ncfile('./','qtot')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    period = dt.dates('2010-2011')
    extent = extent.ioffset[200,200:202]
    r = runner.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_snapshotfm_A():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    #input_map = awral.get_default_mapping()
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_ncfile_snapshot(
                            os.path.dirname(__file__), 's0')

    runner = OnDemandSimulator(awral, input_map, omapping=output_map)

    print("RUNNER NEW: multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[200, 200:202]
    r = runner.run(period,extent)


# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_snapshotfm_B():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    #input_map = awral.get_default_mapping()
    output_map = awral.get_output_mapping()

    output_map['s0_save'] = nodes.write_to_ncfile_snapshot(
                            os.path.dirname(__file__), 's0', mode='r+')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    print("RUNNER NEW (FILES EXISTING): multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[200, 200:202]
    r = runner.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_snapshotfm_C():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    #input_map = awral.get_default_mapping()
    output_map = awral.get_output_mapping()

    output_map['s0_save'] = nodes.write_to_ncfile_snapshot(
                            os.path.dirname(__file__), 's0', mode='r+')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    print("RUNNER OLD (FILES EXISTING): single cell, single year")
    period = dt.dates('2010')
    extent = e_all.ioffset[202, 202]
    r = runner.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_splitfm_D():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    print("RUNNER NEW: multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[200:202,200:202]

    #input_map = awral.get_default_mapping()
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0')
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    r = runner.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_splitfm_E():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    #input_map = awral.get_default_mapping()
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0',mode='w')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    print("RUNNER NEW (FILES EXISTING): multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[200:202,200:202]
    r = runner.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_splitfm_F():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    #input_map = awral.get_default_mapping()
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0',mode='w')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    print("RUNNER OLD (FILES EXISTING): single cell, single year")
    period = dt.dates('2015')
    extent = e_all.ioffset[202,202]
    r = runner.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_splitfm_G():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator


    print("RUNNER NEW: single cell ncf, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[202,202]

    #input_map = awral.get_default_mapping()
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0')
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map,omapping=output_map)
    r = runner.run(period,extent)

if __name__ == '__main__':
    # test_FileWriterNode()
    # test_SplitFileWriterNode()
    # test_output_graph_processing_flatfm()
    # test_output_graph_processing_splitfm()
    # test_output_graph()
    # test_OutputNode()
    pass
