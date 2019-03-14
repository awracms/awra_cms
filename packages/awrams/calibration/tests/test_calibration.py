from nose.tools import nottest


def test_imports():
    import awrams.calibration

@nottest
def test_single_catchment():
    import awrams.calibration.calibrate as cal
    from awrams.calibration.sce import SCEOptimizer,ProxyOptimizer

    from awrams.utils import datetools as dt

    import pandas as pd
    import os

    cal_catchment = '421103' # '204007' # '421103' '105001'
    time_period = dt.dates('1990 - 1995') #2005 - 2010') #1990 - 1995')

    # path = os.path.join(os.path.dirname(__file__),'data')
    path = os.path.join(os.path.dirname(__file__),'..','..','test_data','calibration')

    from awrams.utils import catchments
    # Get the catchment as a spatial extent we can use as the bounds of the simulation
    try:
        db = catchments.CatchmentDB()
        spatial = db.get_by_id(cal_catchment)

    except ImportError as e:
        print(e)
        # read catchment extent from a pickle
        import pickle
        # path = os.path.join(os.path.dirname(__file__),'../../test_data')
        pkl = os.path.join(path,'extent_421103.pkl')
        spatial = pickle.load(open(pkl,'rb'))


    def change_path_to_forcing(imap):
        from awrams.utils.nodegraph import nodes

        # path = os.path.join(os.path.dirname(__file__),'../../test_data')

        FORCING = {
            'tmin'  : ('temp_min*','temp_min_day'),
            'tmax'  : ('temp_max*','temp_max_day'),
            'precip': ('rain_day*','rain_day'),
            'solar' : ('solar*'   ,'solar_exposure_day')
        }
        for k,v in FORCING.items():
            imap.mapping[k+'_f'] = nodes.forcing_from_ncfiles(path,v[0],v[1],cache=True)

    change_path_to_forcing(cal.input_map)

    csv = os.path.join(path,'q_obs.csv')
    qobs = pd.read_csv(csv,parse_dates=[0])
    qobs = qobs.set_index(qobs.columns[0])
    obs = qobs[cal_catchment]

    parameters = cal.get_parameter_df(cal.input_map.mapping)

    evaluator = cal.RunoffEvaluator(time_period,spatial,obs)

    # Create the SCE instance...
    sce = ProxyOptimizer(13,2,4,3,3,parameters,evaluator)
    sce.max_iter = 100

    sce.run_optimizer()

    # run with seed population...
    sce.run_optimizer(seed=sce.population.iloc[0])

    sce.terminate_children()


@nottest
def test_multiple_catchment():
    import os
    import pandas as pd
    import pickle
    import sys

    import awrams.calibration.calibration as c
    from awrams.models import awral
    from awrams.utils.metatypes import ObjectDict

    path = os.path.join(os.path.dirname(__file__),'..','..','test_data','calibration')
    awral.CLIMATE_DATA = path

    cal = c.CalibrationInstance(awral)
    print(sys.argv)
    if sys.argv[0].endswith('nosetests') or sys.argv[1].endswith('nosetests'):
        cal.node_settings.num_workers = 1
        cal.num_nodes = 1
        cal.termp.max_iter = 40

    cal.node_settings.catchment_ids = ['105001', '145003'] #['4508', '105001'] #, '145003']
    cal.node_settings.catchment_extents = pickle.load(open(os.path.join(path,'cal_catchment_extents.pkl'),'rb'))

    # cal.node_settings.run_period = pd.date_range("01/01/1950", "31/12/2011")
    # cal.node_settings.eval_period = pd.date_range("01/01/1981", "31/12/2011")
    cal.node_settings.run_period = pd.date_range("1 jan 2005", "31 dec 2010")
    cal.node_settings.eval_period = pd.date_range("1 jan 2005", "31 dec 2010")

    cal.node_settings.output_variables = ['qtot','etot','w0']
    awral.set_outputs({'OUTPUTS_CELL': ['qtot'], 'OUTPUTS_HRU': [], 'OUTPUTS_AVG': ['etot', 'w0']})

    cal.node_settings.observations.qtot = ObjectDict()
    cal.node_settings.observations.etot = ObjectDict()
    cal.node_settings.observations.w0 = ObjectDict()
    cal.node_settings.observations.qtot.source_type='csv'
    cal.node_settings.observations.etot.source_type='csv'
    cal.node_settings.observations.w0.source_type='csv'
    # HostPath is a portable paths API; it allows you specify common bases on multiple systems that are resovled at runtime
    cal.node_settings.observations.qtot.filename = os.path.join(path,'q_obs.csv')
    cal.node_settings.observations.etot.filename = os.path.join(path,'cmrset_obs.csv')
    cal.node_settings.observations.w0.filename   = os.path.join(path,'sm_amsre_obs.csv')
    # View the localised hostpath...
    # cal.node_settings.objective.localf.filename = os.path.join(os.path.dirname(__file__),'objectives','multivar_objectives.py')
    cal.node_settings.objective.localf.classname = 'TestLocalMulti'
    # cal.node_settings.objective.globalf.filename = os.path.join(os.path.dirname(__file__),'objectives','multivar_objectives.py')
    cal.node_settings.objective.globalf.classname = 'GlobalMultiEval'

    cal.setup_local()

    cal.run_local()


if __name__ == '__main__':
    test_multiple_catchment()
    # test_single_catchment()
