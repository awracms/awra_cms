import pandas as pd
from nose.tools import nottest,with_setup

@nottest
def setup():

    from os.path import join,dirname

    data_path = join(dirname(__file__),'..','..','test_data','simulation')

    FORCING = {
        'tmin':   ('temp_min_day_[0-9][0-9][0-9][0-9].nc','temp_min_day',data_path),
        'tmin2da':('temp_min_day_[0-9][0-9][0-9][0-9].nc','temp_min_day',data_path),
        'tmax':   ('temp_max_day_[0-9][0-9][0-9][0-9].nc','temp_max_day',data_path),
        'precip': ('rain_day_[0-9][0-9][0-9][0-9].nc','rain_day',data_path),
        'solar':  ('solar_exposure_day_[0-9][0-9][0-9][0-9].nc','solar_exposure_day',data_path),
        'vprp':   ('temp_min_day_[0-9][0-9][0-9][0-9].nc','temp_min_day',data_path),
        'wind':   ('temp_min_day_[0-9][0-9][0-9][0-9].nc','temp_min_day',data_path),
        'fveg':   ('temp_min_day_[0-9][0-9][0-9][0-9].nc','temp_min_day',data_path),
        'alb':    ('temp_min_day_[0-9][0-9][0-9][0-9].nc','temp_min_day',data_path)
    }
    import awrams.models.et.settings
    awrams.models.et.settings.FORCING = {k+"_f":dict(zip(("pattern","nc_var","path"),v)) for k,v in FORCING.items()}

    from awrams.models.et.model import ETModel

    global et
    et = ETModel()

    global imap,omap
    imap = et.get_default_mapping()
    omap = et.get_output_mapping()

    global outpath
    outpath = join(dirname(__file__),'..','..','test_data','simulation','outputs')

    et.save_outputs(omap,path=outpath)

@nottest
def tear_down():
    try:
        from os import remove
        from os.path import join
        from glob import glob
        for f in glob(join(outpath,'*.nc')):
            remove(f)
    except:
        pass


@with_setup(setup,tear_down)
def test_sim():
    '''
    Required inputs
    mortons: tmin,tmax,solar,vapour pressure ### apparently not albedo
    fao56:   tmin,tmax,solar,vapour pressure,wind
    penpan:  tmin2da,tmax,solar,vapour pressure,wind
    penman:  tmin,tmax,solar,vapour pressure,wind,veg,albedo
    '''

    from awrams.utils import extents
    from awrams.simulation.server import SimulationServer

    et.set_models_to_run(["fao56","asce","morton_shallow_lake","morton_areal","penpan","penman"])

    extent = extents.get_default_extent()
    extent = extent.icoords[-32:-35,115:118]

    period = pd.date_range("1 dec 2010","31 jan 2011",freq='D')

    sim = SimulationServer(et)
    sim.run(imap,omap,period,extent)


@nottest
def test_asce():
    import awrams.models.settings
    import awrams.utils.nodegraph.nodes as nodes
    # awrams.models.settings.CLIMATE_DATA = _join(TRAINING_DATA_PATH,'climate/BOM_climate/')

    bom_data_path = '/data/cwd_awra_data/awra_inputs/climate_generated_sdcvd-awrap01/'
    wind_data_path = '/data/cwd_awra_data/awra_inputs/mcvicar_wind/'
    fveg_data_path = '/data/cwd_awra_data/awra_inputs/modis_fveg/8day/pv/'
    albedo_data_path = '/data/cwd_awra_data/awra_inputs/randall_albedo/'
    temp_min_2day_datapath = '/data/cwd_awra_data/awra_inputs/2dayAveTmin/'

    FORCING = {
        'tmin': ('temp_min*.nc','temp_min_day',bom_data_path + 'temp_min_day/'),
        'tmin2da': ('temp_min*.nc','temp_min_2dayave',temp_min_2day_datapath),
        'tmax': ('temp_max*.nc','temp_max_day',bom_data_path + 'temp_max_day/'),
        'precip': ('rain_day*.nc','rain_day',bom_data_path + 'rain_day/'),
        'solar': ('solar*.nc','solar_exposure_day',bom_data_path + 'solar_exposure_day/'), #,
        'vprp':('vapour_pressure*.nc', 'vapour_pressure', bom_data_path + 'vapour_pressure/'),#, #h09
        'wind':('wind*.nc', 'wind', wind_data_path),
        'fveg':('fveg*.nc', 'fveg', fveg_data_path),
        'alb':('albedo*.nc', 'albedo', albedo_data_path)
    }
    import awrams.models.et.settings
    awrams.models.et.settings.FORCING = {k+"_f":dict(zip(("pattern","nc_var","path"),v)) for k,v in FORCING.items()}

    '''
    Required inputs
    mortons: tmin,tmax,solar,vapour pressure ### apparently not albedo
    fao56:   tmin,tmax,solar,vapour pressure,wind
    penpan:  tmin2da,tmax,solar,vapour pressure,wind
    penman:  tmin,tmax,solar,vapour pressure,wind,veg,albedo
    '''

    from awrams.utils import extents
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.simulation.server import SimulationServer
    from awrams.models.et.model import ETModel

    et = ETModel()
    # et.set_models_to_run(["fao56","morton_shallow_lake","morton_areal","penpan","penman"])
    # et.set_models_to_run(["fao56","morton_areal","penpan"])
    # et.set_models_to_run(["morton_areal"])
    # et.set_models_to_run(["fao56"])
    # et.set_models_to_run(["penpan"])
    # et.set_models_to_run(["penman"])
    et.set_models_to_run(["asce"])
    # et.set_models_to_run(["morton_shallow_lake"])

    imap = et.get_default_mapping()
    # print(imap['fveg'])
    # print(imap['fveg_f'])
    # imap['fveg_f'] = nodes.forcing_from_modis_ncfiles(fveg_data_path,'fveg_filled.nc','fveg')
    # imap['fveg'] = nodes.mul('fveg_f',0.01)
    # print(imap['fveg_f'])
    # exit()
    omap = et.get_output_mapping()
    et.save_outputs(omap,path="./outputs/ss/")
    # et.save_outputs(omap,path="/data/cwd_awra_data/awra_test_outputs/Scheduled_et_sdcvt-awrap01/")
    # print(omap)

    # omap['nc_msl_et'] = nodes.write_to_annual_ncfile("./",'msl_et')

    extent = extents.get_default_extent()
    # extent = extent.icoords[-32:-34,115:118]
    # extent = extent.ioffset[440,95]
    # extent = extent.ioffset[490,90]
    # extent = extent.ioffset[277:304,50]
    # extent = extent.ioffset[256:265,60]
    # extent = extent.ioffset[455,770:791]
    # extent = extent.ioffset[230:240,120]
    # print("Extent",extent)

    # period = pd.date_range("1 jan 2010","31 dec 2011",freq='D')
    # period = pd.date_range("1 jan 2010",periods=1,freq='D')
    period = pd.date_range("1 jan 2011","31 dec 2011",freq='D')
    # period = pd.date_range("1 jan 2016","31 jan 2016",freq='D')
    # period = pd.date_range("1 jan 1990","30 apr 2018",freq='D')

    # sim = OnDemandSimulator(et,imap,omapping=omap)
    # print(sim.input_runner.input_graph['models']['exe'].get_dataspec())
    # r = sim.run(period,extent)
    # # print(r)

    sim = SimulationServer(et)
    sim.run(imap,omap,period,extent)

@nottest
def test_run():
    import awrams.models.settings
    import awrams.utils.nodegraph.nodes as nodes
    # awrams.models.settings.CLIMATE_DATA = _join(TRAINING_DATA_PATH,'climate/BOM_climate/')

    bom_data_path = '/data/cwd_awra_data/awra_inputs/climate_generated_sdcvd-awrap01/'
    wind_data_path = '/data/cwd_awra_data/awra_inputs/mcvicar_wind/'
    fveg_data_path = '/data/cwd_awra_data/awra_inputs/modis_fveg/8day/pv/'
    albedo_data_path = '/data/cwd_awra_data/awra_inputs/randall_albedo/'
    temp_min_2day_datapath = '/data/cwd_awra_data/awra_inputs/2dayAveTmin/'

    FORCING = {
        'tmin': ('temp_min*.nc','temp_min_day',bom_data_path + 'temp_min_day/'),
        'tmin2da': ('temp_min*.nc','temp_min_2dayave',temp_min_2day_datapath),
        'tmax': ('temp_max*.nc','temp_max_day',bom_data_path + 'temp_max_day/'),
        'precip': ('rain_day*.nc','rain_day',bom_data_path + 'rain_day/'),
        'solar': ('solar*.nc','solar_exposure_day',bom_data_path + 'solar_exposure_day/'), #,
        'vprp':('vapour_pressure*.nc', 'vapour_pressure', bom_data_path + 'vapour_pressure/'),#, #h09
        'wind':('wind*.nc', 'wind', wind_data_path),
        'fveg':('fveg*.nc', 'fveg', fveg_data_path),
        'alb':('albedo*.nc', 'albedo', albedo_data_path)
    }
    import awrams.models.et.settings
    awrams.models.et.settings.FORCING = {k+"_f":dict(zip(("pattern","nc_var","path"),v)) for k,v in FORCING.items()}

    '''
    Required inputs
    mortons: tmin,tmax,solar,vapour pressure ### apparently not albedo
    fao56:   tmin,tmax,solar,vapour pressure,wind
    penpan:  tmin2da,tmax,solar,vapour pressure,wind
    penman:  tmin,tmax,solar,vapour pressure,wind,veg,albedo
    '''

    from awrams.utils import extents
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.simulation.server import SimulationServer
    from awrams.models.et.model import ETModel

    et = ETModel()
    et.set_models_to_run(["morton_shallow_lake"])

    imap = et.get_default_mapping()
    # print(imap['fveg'])
    # print(imap['fveg_f'])
    # imap['fveg_f'] = nodes.forcing_from_modis_ncfiles(fveg_data_path,'fveg_filled.nc','fveg')
    # imap['fveg'] = nodes.mul('fveg_f',0.01)
    # print(imap['fveg_f'])
    # exit()
    omap = et.get_output_mapping()
    # et.save_outputs(omap,path="./outputs/ods/")
    et.save_outputs(omap,path="/data/cwd_awra_data/awra_test_outputs/Scheduled_et_sdcvt-awrap01/")
    # print(omap)

    # omap['nc_msl_et'] = nodes.write_to_annual_ncfile("./",'msl_et')

    extent = extents.get_default_extent()
    # extent = extent.icoords[-32:-34,115:118]
    # extent = extent.ioffset[440,95]
    # extent = extent.ioffset[490,90]
    # extent = extent.ioffset[277:304,50]
    # extent = extent.ioffset[256:265,60]
    # extent = extent.ioffset[455,770:791]
    # extent = extent.ioffset[230:240,120]
    # print("Extent",extent)

    # period = pd.date_range("1 jan 2010","31 dec 2011",freq='D')
    # period = pd.date_range("1 jan 2010",periods=1,freq='D')
    # period = pd.date_range("1 jan 2011","31 dec 2011",freq='D')
    # period = pd.date_range("1 jan 2016","31 jan 2016",freq='D')
    period = pd.date_range("1 jan 1990","30 apr 2018",freq='D')

    # sim = OnDemandSimulator(et,imap,omapping=omap)
    # print(sim.input_runner.input_graph['models']['exe'].get_dataspec())
    # r = sim.run(period,extent)
    # # print(r)

    sim = SimulationServer(et)
    sim.run(imap,omap,period,extent)

@nottest
def test_fao_static_wind():
    import awrams.models.settings
    import awrams.utils.nodegraph.nodes as nodes
    # awrams.models.settings.CLIMATE_DATA = _join(TRAINING_DATA_PATH,'climate/BOM_climate/')

    bom_data_path = '/data/cwd_awra_data/awra_inputs/climate_generated_sdcvd-awrap01/'
    wind_data_path = '/data/cwd_awra_data/awra_inputs/mcvicar_wind/'
    fveg_data_path = '/data/cwd_awra_data/awra_inputs/modis_fveg/8day/pv/'
    albedo_data_path = '/data/cwd_awra_data/awra_inputs/randall_albedo/'
    temp_min_2day_datapath = '/data/cwd_awra_data/awra_inputs/2dayAveTmin/'

    FORCING = {
        'tmin': ('temp_min*.nc','temp_min_day',bom_data_path + 'temp_min_day/'),
        'tmin2da': ('temp_min*.nc','temp_min_2dayave',temp_min_2day_datapath),
        'tmax': ('temp_max*.nc','temp_max_day',bom_data_path + 'temp_max_day/'),
        'precip': ('rain_day*.nc','rain_day',bom_data_path + 'rain_day/'),
        'solar': ('solar*.nc','solar_exposure_day',bom_data_path + 'solar_exposure_day/'), #,
        'vprp':('vapour_pressure*.nc', 'vapour_pressure', bom_data_path + 'vapour_pressure/'),#, #h09
        'wind':('wind*.nc', 'wind', wind_data_path),
        'fveg':('fveg*.nc', 'fveg', fveg_data_path),
        'alb':('albedo*.nc', 'albedo', albedo_data_path)
    }
    import awrams.models.et.settings
    awrams.models.et.settings.FORCING = {k+"_f":dict(zip(("pattern","nc_var","path"),v)) for k,v in FORCING.items()}

    '''
    Required inputs
    mortons: tmin,tmax,solar,vapour pressure ### apparently not albedo
    fao56:   tmin,tmax,solar,vapour pressure,wind
    penpan:  tmin2da,tmax,solar,vapour pressure,wind
    penman:  tmin,tmax,solar,vapour pressure,wind,veg,albedo
    '''

    from awrams.utils import extents
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.simulation.server import SimulationServer
    from awrams.models.et.model import ETModel

    et = ETModel()
    et.set_models_to_run(["fao56"])

    imap = et.get_default_mapping()
    imap['wind'] = nodes.spatial_from_file(awrams.models.et.settings.SPATIAL_FILE,'parameters/windspeed')


    omap = et.get_output_mapping()
    et.save_outputs(omap,path="./outputs/fao56_static_wind/")
    # et.save_outputs(omap,path="/data/cwd_awra_data/awra_test_outputs/Scheduled_et_sdcvt-awrap01/")
    # print(omap)

    # omap['nc_msl_et'] = nodes.write_to_annual_ncfile("./",'msl_et')

    extent = extents.get_default_extent()
    # extent = extent.icoords[-32:-34,115:118]
    # extent = extent.ioffset[440,95]
    # extent = extent.ioffset[490,90]
    # extent = extent.ioffset[277:304,50]
    # extent = extent.ioffset[256:265,60]
    # extent = extent.ioffset[455,770:791]
    # extent = extent.ioffset[230:240,120]
    # print("Extent",extent)

    period = pd.date_range("1 jan 2010","31 dec 2011",freq='D')
    # period = pd.date_range("1 jan 2010",periods=1,freq='D')
    # period = pd.date_range("1 jan 2011","31 dec 2011",freq='D')
    # period = pd.date_range("1 jan 2016","31 jan 2016",freq='D')
    # period = pd.date_range("1 jan 1990","30 apr 2018",freq='D')

    # sim = OnDemandSimulator(et,imap,omapping=omap)
    # print(sim.input_runner.input_graph['models']['exe'].get_dataspec())
    # r = sim.run(period,extent)
    # # print(r)

    sim = SimulationServer(et)
    sim.run(imap,omap,period,extent)

if __name__ == "__main__":
    # test_ondemand_msl()
    # test_asce()
    test_fao_static_wind()
