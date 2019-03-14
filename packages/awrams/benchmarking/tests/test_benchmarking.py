from nose.tools import nottest
import matplotlib
import os
matplotlib.use('agg')

from awrams.utils import config_manager
system_profile = config_manager.get_system_profile().get_settings()
TEST_DATA_PATH = system_profile['DATA_PATHS']['TEST_DATA']

def test_imports():
    import awrams.benchmarking
    import awrams.benchmarking.meta
    #import awrams.benchmarking.processing

    import awrams.benchmarking.benchmark
    import awrams.benchmarking.comparison
    import awrams.benchmarking.stats
    import awrams.benchmarking.utils

    import awrams.benchmarking.meta.oznet
    import awrams.benchmarking.meta.sasmas

    # import awrams.benchmarking.processing.oznet
    # import awrams.benchmarking.processing.sasmas

@nottest
def test_force_fail():
    assert False

#@nottest
def test_benchmark():
    from awrams.benchmarking.benchmark import Benchmark
    from awrams.benchmarking.utils import read_id_csv
    from awrams.utils import datetools as dt
    import os

    csv_path = os.path.join(TEST_DATA_PATH, 'benchmarking', 'runoff', 'q_obs.csv')

    b = Benchmark('QObs','qtot_avg')
    id_list = read_id_csv(os.path.join(TEST_DATA_PATH, 'benchmarking', 'catchment_ids.csv'))
    b.period = dt.dates("1981", "30/12/2011")
    b.load(csv_path,id_list)

    csv_path = os.path.join(TEST_DATA_PATH, 'benchmarking', 'runoff', 'awral_qtot_avg.csv')
    b.add_model('awral_v4', csv_path)

    assert hasattr(b.benchmark.models,'awral_v4')

    b.benchmark.stat_percentiles()
    b.benchmark.data_percentiles()
    b.benchmark.stat()
    b.benchmark.plot_regression()
    b.benchmark.plot_box('pearsons_r')
    b.benchmark.plot_cdf()
    b.benchmark.plot_timeseries('105001')
#    assert False

#@nottest
def test_benchmarksoilmoisture():
    from awrams.benchmarking.benchmark import BenchmarkSoilMoisture
    from awrams.benchmarking.utils import read_id_csv
    import awrams.benchmarking.meta.sasmas as sasmas
    from awrams.utils import datetools as dt
    import os

    sasmas_data_path = os.path.join(TEST_DATA_PATH, 'benchmarking', 'sasmas')

    b = BenchmarkSoilMoisture('SASMAS','soil_moisture', sasmas.meta)
    site_list = ['G6','K2','M1','S4']
    mod_site_list = ['SASMAS Soil moisture_' + site for site in site_list]
    b.period = dt.dates('2003-2011')
    b.load(sasmas_data_path,mod_site_list,convert_units=100.)

    csv_path = os.path.join(TEST_DATA_PATH, 'benchmarking', 'sasmas', 'awral_${v}.csv')
    b.add_model('awral_v4', csv_path)

    assert hasattr(b.benchmark,'top')
    assert hasattr(b.benchmark,'shallow')
    assert hasattr(b.benchmark,'middle')
    assert hasattr(b.benchmark,'deep')
    assert hasattr(b.benchmark,'profile')
    assert hasattr(b.benchmark.top.models,'awral_v4')

    b.benchmark.top.stat_percentiles()
    b.benchmark.middle.data_percentiles()
    b.benchmark.profile.stat()
    b.benchmark.shallow.plot_regression()
    b.benchmark.deep.plot_box('pearsons_r')
    b.benchmark.top.plot_cdf()
    b.benchmark.shallow.plot_timeseries('G6')
#    assert False
