from os.path import join
from awrams.utils.metatypes import objectify
import os
from logging import FATAL,CRITICAL,ERROR,WARNING,INFO,DEBUG
from awrams.utils.awrams_log import APPEND_FILE,TIMESTAMPED_FILE,ROTATED_SIZED_FILE,DAILY_ROTATED_FILE
from awrams.utils import config_manager


AWRAMS_BASE_PATH = str(config_manager.get_awrams_base_path())
BASE_DATA_PATH = str(config_manager.get_awrams_data_path())


# Mapping for /data/cwd_awra_data/awra_test_inputs/climate*
FORCING_MAP_AWAP = {
    'tmin': ('temp_min_day/temp_min_day*.nc', 'temp_min_day'),
    'tmax': ('temp_max_day/temp_max_day*.nc', 'temp_max_day'),
    'precip': ('rain_day/rain_day*.nc', 'rain_day'),
    'solar': ('solar_exposure_day/solar_exposure_day*.nc', 'solar_exposure_day'),
    'wind': ('wind/wind*.nc', 'wind')

}

config_options = {
    'CHUNKSIZES': {
        'TIME': 32,
        'SPATIAL': 32
    },

    'LOGGER_SETTINGS': {

        'APP_NAME': 'awrams',

        'LOG_FORMAT': '%(asctime)s %(levelname)s %(message)s',
        'LOG_TO_STDOUT': True,
        'LOG_TO_STDERR': False,
        'LOG_TO_FILE': False,

        # File logging options

        'FILE_LOGGING_MODE': APPEND_FILE,
        'LOGFILE_BASE': os.path.join(AWRAMS_BASE_PATH,'awrams'),

        #
        'LOG_LEVEL': INFO,
        'DEBUG_MODULES': [],

        # The following are the default values which affect DAILY_ROTATED_FILE and
        # ROTATED_SIZED_FILE modes only
        # If you select one of these FILE_LOGGING_MODEs you can then customise how 
        # many or what size the files are

        # ROTATED_SIZED_FILE mode is affected by these params:
        # How many files to rotate:
        'ROTATED_SIZED_FILES': 10,
        #Sze of the file before it rotates:
        'ROTATED_SIZED_BYTES': 20000,

        # DAILY_ROTATED_FILE mode is affected by:
        # How many files to rotate(on a daily basis) so 7 is a week's worth of daily
        # files
        'DAILY_ROTATED_FILES': 7    
    }

}

config_options = objectify(config_options)


def get_settings():
    TEST_DATA_PATH = join(BASE_DATA_PATH, 'test_data')
    TRAINING_DATA_PATH = join(BASE_DATA_PATH, 'training')
    benchmark_sites_file = join(BASE_DATA_PATH, 'benchmarking/SiteLocationsWithUniqueID.csv')
    SHAPEFILES = join(BASE_DATA_PATH, 'spatial/shapefiles')

    CLIMATOLOGIES = {
        'AWAP_DAILY': {
            'solar': (join(BASE_DATA_PATH, 'climatology/climatology_daily_solar_exposure_day.nc'), 'solar_exposure_day')
        }
    }

    if os.name == 'nt':
        COMPILER = 'CLANG_WINDOWS'
    else:
        COMPILER = 'GCC_DEFAULT'

    settings = {
        'DATA_PATHS': {
            'AWRAMS_BASE': AWRAMS_BASE_PATH,
            'BASE_DATA': BASE_DATA_PATH,
            'MASKS': join(BASE_DATA_PATH, 'spatial/masks'),
            'SHAPEFILES': SHAPEFILES,
            'CATCHMENT_SHAPEFILE': join(SHAPEFILES,'Final_list_all_attributes.shp'),
            'TEST_DATA': TEST_DATA_PATH,
            'TRAINING_DATA': TRAINING_DATA_PATH,
            'MODEL_DATA': join(BASE_DATA_PATH, 'model_data'),
            'CODE': join(AWRAMS_BASE_PATH, 'code'),
            'ASCAT': {
                'TRAINING': join(TRAINING_DATA_PATH, 'benchmarking/ascat/'),
                'TEST': join(TEST_DATA_PATH, 'benchmarking/ascat/')
            },
            'BUILD_CACHE': join(AWRAMS_BASE_PATH, 'build_cache')
        },
        'SIMULATION': {
                'SPATIAL_CHUNK': 128,
                'TIME_CHUNK': 32,
                'MIN_CELLS_PER_WORKER': 32,
                'TASK_BUFFERS': 3
        },
        # +++ Should move to external file so datasets can be shared between profiles
        'CLIMATE_DATASETS': {
            'TRAINING': {
                'FORCING': {
                    'PATH': join(TRAINING_DATA_PATH, 'climate/bom_awap'),
                    'MAPPING': FORCING_MAP_AWAP
                },
                'CLIMATOLOGY': CLIMATOLOGIES['AWAP_DAILY']
            },
            'TESTING': {
                'FORCING': {
                    'PATH': join(TEST_DATA_PATH, 'simulation/climate'),
                    'MAPPING': FORCING_MAP_AWAP
                },
                'CLIMATOLOGY': CLIMATOLOGIES['AWAP_DAILY']
            }
        },
        'BENCHMARKING': {
            'BENCHMARK_SITES': benchmark_sites_file,
            'MONTHLY_REJECTION_THRESHOLD': 15,
            'ANNUAL_REJECTION_THRESHOLD': 6,
            'SM_MODEL_VARNAMES': ['s0_avg', 'ss_avg', 'sd_avg'],
            'SM_MODEL_LAYERS': {'s0_avg': 100., 'ss_avg': 900., 'sd_avg':
                                5000.},
            'SM_OBSERVED_LAYERS': ('profile','top','shallow','middle','deep'),
            'FIG_SIZE': (14,6),
            'CELLSIZE': 0.05,
            'LANDSCAPE_VERSION_EQUIVALENCE': {"5":"45","5R":"45","5Q":"45"}
        },
        # Preferred compiler; referenced in model settings
        'COMPILER': COMPILER,
        
        'IO_SETTINGS' : {
            'CHUNKSIZES': config_options['CHUNKSIZES'],

            'DEFAULT_CHUNKSIZE': (config_options['CHUNKSIZES']['TIME'], \
                                  config_options['CHUNKSIZES']['SPATIAL'], \
                                  config_options['CHUNKSIZES']['SPATIAL']),

            'VAR_CHUNK_CACHE_SIZE': 2**20, # =1048576 ie 1Mb
            'VAR_CHUNK_CACHE_NELEMS': 1009, # prime number
            'VAR_CHUNK_CACHE_PREEMPTION': 0.75, # 1 for read or write only
            
            # '_fallthrough' will attempt to use _h5py, then netCDF4 if that fails
            'DB_OPEN_WITH': '_fallthrough', #"_h5py" OR "_nc"

            'MAX_FILES_PER_SFM': 32, # Maximum files allowed open in each SplitFileManager.
            # Maximum chunksize to read during extraction (in bytes)
            'MAX_EXTRACT_CHUNK': 2**24
        },

        'LOGGER_SETTINGS': config_options['LOGGER_SETTINGS'],
    
        # Used in extents.get_default_extent
        # Consider creating extents objects explicitly from files rather than using this method.
        # It exists for backwards compatibility, and will be deprecated
        'DEFAULT_MASK_FILE': 'web_mask_v5.h5'
    }

    return objectify(settings)
