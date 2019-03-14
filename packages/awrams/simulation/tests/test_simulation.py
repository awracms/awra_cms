from glob import glob
from os import remove
from os.path import join,dirname

from nose.tools import nottest,with_setup
import numpy as np

from awrams.utils import datetools as dt
from awrams.utils import extents
from awrams.utils.io.data_mapping import SplitFileManager
from awrams.utils.mapping_types import period_to_tc
from awrams.utils.nodegraph import nodes,graph
from awrams.utils.nodegraph.nodes import forcing_from_ncfiles
from awrams.utils import config_manager

@nottest
def get_initial_states(imap):
    
    data_paths = config_manager.get_system_profile('default').get_settings()['DATA_PATHS']
    data_path = join(data_paths['BASE_DATA'],'test_data','simulation')

    mapping = imap
    mapping['init_sr'] = nodes.init_state_from_ncfile(data_path,'sr_bal*','sr_bal')
    mapping['init_sg'] = nodes.init_state_from_ncfile(data_path,'sg_bal*','sg_bal')

    HRU = {'_hrusr':'_sr','_hrudr':'_dr'}
    for hru in ('_hrusr','_hrudr'):
        for state in ["s0","ss","sd",'mleaf']:
            mapping['init_'+state+hru] = nodes.init_state_from_ncfile(data_path,state+HRU[hru]+'*',state+HRU[hru])

@nottest
def get_initial_states_dict(imap,period,extent):
    data_map = {}
    data_paths = config_manager.get_system_profile('default').get_settings()['DATA_PATHS']
    data_path = join(data_paths['BASE_DATA'],'test_data','simulation')
    period = [period[0] - 1]
    node_names = {'mleaf_dr': 'init_mleaf_hrudr',
                  'mleaf_sr': 'init_mleaf_hrusr',
                  's0_dr': 'init_s0_hrudr',
                  's0_sr': 'init_s0_hrusr',
                  'ss_dr': 'init_ss_hrudr',
                  'ss_sr': 'init_ss_hrusr',
                  'sd_dr': 'init_sd_hrudr',
                  'sd_sr': 'init_sd_hrusr',
                  'sg_bal': 'init_sg',
                  'sr_bal': 'init_sr'}
    for k in 'mleaf_dr','s0_dr','sd_dr','sg_bal','ss_dr','mleaf_sr','s0_sr','sd_sr','sr_bal','ss_sr':
        sfm = SplitFileManager.open_existing(data_path,k+'*nc',k)
        data_map[node_names[k]] = sfm.get_data(period,extent)[0]
    nodes.init_states_from_dict(imap,data_map,extent)

@nottest
def setup():
    from awrams.utils import config_manager

    sys_settings = config_manager.get_system_profile('default').get_settings()
    model_profile = config_manager.get_model_profile('awral','v6_default')
    model_settings = model_profile.get_settings()
    
    
    global awral
    awral = model_profile.get_model(model_settings)
    
    global period
    period = dt.dates('dec 2010 - jan 2011')

    global input_map
    input_map = model_profile.get_input_mapping(model_settings)
    model_settings['CLIMATE_DATASET'] = sys_settings['CLIMATE_DATASETS']['TESTING']

    global output_map
    output_map = awral.get_output_mapping()

    global outpath
    outpath = join(dirname(__file__),'..','..','test_data','simulation','outputs')

    output_map['s0_ncsave'] = nodes.write_to_annual_ncfile(outpath,'s0')

    output_map['mleaf_hrudr_state'] = nodes.write_to_ncfile_snapshot(outpath,'mleaf_hrudr')

@nottest
def setup_v5():
    from awrams.utils import config_manager

    sys_settings = config_manager.get_system_profile('default').get_settings()
    model_profile = config_manager.get_model_profile('awral','v5_default')
    model_settings = model_profile.get_settings()
    
    
    global awral
    awral = model_profile.get_model(model_settings)
    
    global period
    period = dt.dates('dec 2010 - jan 2011')

    global input_map
    input_map = model_profile.get_input_mapping(model_settings)
    model_settings['CLIMATE_DATASET'] = sys_settings['CLIMATE_DATASETS']['TESTING']

    global output_map
    output_map = awral.get_output_mapping()

    global outpath
    outpath = join(dirname(__file__),'..','..','test_data','simulation','outputs')

    output_map['s0_ncsave'] = nodes.write_to_annual_ncfile(outpath,'s0')

    output_map['mleaf_hrudr_state'] = nodes.write_to_ncfile_snapshot(outpath,'mleaf_hrudr')


def tear_down():
    try:
        for f in glob(join(outpath,'*.nc')):
            remove(f)
    except:
        pass

@with_setup(setup,tear_down)
def test_ondemand_region():
    ### test region
    extent = extents.get_default_extent()
    extent = extent.icoords[-32:-34,115:118]
    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral,input_map,omapping=output_map)
    r = sim.run(period,extent)

@with_setup(setup_v5,tear_down)
def test_ondemand_region_v5():
    ### test region
    extent = extents.get_default_extent()
    extent = extent.icoords[-32:-34,115:118]
    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral,input_map,omapping=output_map)
    r = sim.run(period,extent)


@with_setup(setup,tear_down)
def test_ondemand_point():
    ### test point
    extent = extents.get_default_extent()
    extent = extent.icoords[-32.1,115.1]
    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral,input_map,omapping=output_map)
    r = sim.run(period,extent)

@with_setup(setup,tear_down)
def test_ondemand_with_mask():
    # Make output map with daily frequency
    output_map['mleaf_hrudr_state'] = nodes.write_to_ncfile_snapshot(outpath,'mleaf_hrudr', freq='D')

    period = dt.dates('25 dec 2010', '26 dec 2010')
    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral, input_map, omapping=output_map)
    r = sim.run(period, extents.get_default_extent())

    # Grab a new copy of the default extent in case the simulator mutated it
    default_mask = extents.get_default_extent().mask

    # Check that the results are masked arrays, using the first results and the final states
    # as examples. Then check the masks are the default mask - masked arrays ensure that masked
    # values are not used in computations.
    assert all(type(r[key] == np.ma.core.MaskedArray) for key in r.keys())
    assert all(type(r['final_states'][key] == np.ma.core.MaskedArray) for key in r['final_states'].keys())
    assert all(np.array_equal(r[key].mask[0], default_mask) for key in r.keys() if key != 'final_states')
    assert all(np.array_equal(r['final_states'][key].mask, default_mask) for key in r['final_states'].keys())

@with_setup(setup)
def test_climatology_point():
    ### test point
    period = dt.dates('dec 2010')
    extent = extents.get_default_extent()
    extent = extent.icoords[-30,120.5]

    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    assert not np.isnan(i['solar_f']).any()

@with_setup(setup)
def test_climatology_region():
    ### test region
    period = dt.dates('dec 2010')
    extent = extents.get_default_extent()
    extent = extent.icoords[-32:-35,115:118]

    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    assert not np.isnan(i['solar_f']).any()

@with_setup(setup)
def test_initial_states_point():
    period = dt.dates('dec 2010')

    ### test a single cell
    extent = extents.get_default_extent()
    extent = extent.icoords[-30,120.5]

    ### simulation with default initial states
    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_default = r['final_states']

    ### simulation with initial states read from nc files
    get_initial_states(input_map)
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init = r['final_states']

    ### compare final states with default states simulation
    ### should be different
    for k,o in outputs_init.items():
        assert not o == outputs_default[k]

    ### save initial states to compare
    ini_states = {}
    for k in i:
        try:
            if k.startswith('init'):
                ini_states[k] = i[k]
        except:
            pass

    ### simulation with initial states read from dict
    get_initial_states_dict(input_map,period,extent)
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init_dict = r['final_states']

    ### compare final states with other ini states simulation
    ### should be same
    for k,o in outputs_init_dict.items():
        assert o == outputs_init[k]

    ### compare initial states from both methods
    ### should be same
    for k in i:
        try:
            if k.startswith('init'):
                assert ini_states[k] == i[k]
        except:
            pass

@with_setup(setup)
def test_initial_states_region():
    period = dt.dates('dec 2010')
    ### test a region
    extent = extents.get_default_extent()
    extent = extent.ioffset[400:408,170:178]
    print(extent)

    ### simulation with default initial states
    from awrams.simulation.ondemand import OnDemandSimulator
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_default = r['final_states']

    ### simulation with initial states read from nc files
    get_initial_states(input_map)
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init = r['final_states']

    ### compare final states with default states simulation
    ### should be different
    for k,o in outputs_init.items():
        assert not (o == outputs_default[k]).any()

    ### save initial states to compare
    ini_states = {}
    for k in i:
        try:
            if k.startswith('init'):
                ini_states[k] = i[k]
        except:
            pass

    ### simulation with initial states read from dict
    get_initial_states_dict(input_map,period,extent)
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init_dict = r['final_states']

    ### compare final states with other ini states simulation
    ### should be same
    for k,o in outputs_init_dict.items():
        assert (o == outputs_init[k]).any()

    ### compare initial states from both methods
    ### should be same
    for k in i:
        try:
            if k.startswith('init'):
                assert ini_states[k] == i[k]
        except:
            pass

    ### simulation with initial states read from nc files
    get_initial_states(input_map)
    sim = OnDemandSimulator(awral,input_map)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init = r['final_states']

    ### compare final states with other ini states simulation
    ### should be same
    for k,o in outputs_init_dict.items():
        assert (o == outputs_init[k]).any()

    ### compare initial states from both methods
    ### should be same
    for k in i:
        try:
            if k.startswith('init'):
                assert ini_states[k] == i[k]
        except:
            pass

@with_setup(setup,tear_down)
def test_server():
    extent = extents.get_default_extent()
    extent = extent.icoords[-32:-35,115:118]

    from awrams.simulation.server import SimulationServer

    sim = SimulationServer(awral)
    sim.run(input_map,output_map,period,extent)

@with_setup(setup,tear_down)
def test_server_initial_states():
    extent = extents.get_default_extent()
    extent = extent.icoords[-32:-35,115:118]

    get_initial_states_dict(input_map,period,extent)

    from awrams.simulation.server import SimulationServer

    sim = SimulationServer(awral)
    sim.run(input_map,output_map,period,extent)

