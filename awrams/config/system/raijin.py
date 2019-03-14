"""
AWRAMS configuration for remote running on NCI Raijin (raijin.nci.org.au)

The defaults assume that each user will have their own config and code directories,
but most likely use shared input data (climate, model data etc)
"""


from awrams.utils.general import join_paths as join
from awrams.utils.metatypes import objectify
import os


"""
User specific settings

AWRAMS_BASE_PATH should point to your personal awrams install on raijin (ie containing
code and config directories)
BASE_DATA_PATH should point to whichever data you intend to use; this data is intended to
be read-only and therefore can be shared between multiple users

"""

# Replace these with your own paths
AWRAMS_BASE_PATH = 'SUGGESTED/g/data/er4/USERNAME/awrams'
BASE_DATA_PATH = '/g/data/er4/awracms_data'

"""
General settings

These mirror config/system/default.py and should not be edited.  Make a copy of
this file if you require a custom config

"""

config_options = {
    'REMOTE_SETTINGS': 
    {

        'HOST_SETTINGS': {
            'HOSTNAME': 'raijin.nci.org.au',
            'USERNAME': 'username' # replace with your raijin username
        },

        #'''
        #PBS Settings
        #'''

        'PBS_SETTINGS': {
            'PROJECT': 'er4', # Default project for PBS jobs +++ BoM/WRM raijin specific
            'JOB_QUEUE': 'normal', # Default job queue for PBS jobs
            'CORES_PER_NODE': 16, # Number of cores per compute node
            'MEM_PER_NODE': (32,'gb'), # Memory per compute note, units
            'NOTIFICATION': {
                'NOTIFY': True,
                'EMAIL': 'USER_EMAIL_ADDRESS', # replace with your email address
                'NOTIFY_OPTS': 'abe'
            }
        },

        #
        #Activation 
        #The following will be used in pbs runs  
        #

        'ACTIVATION': \
            """
            module unload intel-cc
            module unload openmpi

            module load intel-cc/2018.1.163
            module load openmpi/1.10.7

            source /g/data/er4/miniconda3/bin/activate awrams-12dev

            export AWRAMS_BASE_PATH={AWRAMS_BASE_PATH}
            export AWRAMS_DEFAULT_SYSTEM_PROFILE='raijin'
            {DEV_ACTIVATE}
            export PYTHONPATH=$PYTHONPATH:{AWRAMS_BASE_PATH}/code/user

            """
    },
    'DEV_MODE': True,
    'REPO_PATH': '/path/to/your/awrams/repository/'
}

config_options = objectify(config_options)


# Mapping for /data/cwd_awra_data/awra_test_inputs/climate*
FORCING_MAP_AWAP = {
    'tmin': ('temp_min_day/temp_min_day*.nc', 'temp_min_day'),
    'tmax': ('temp_max_day/temp_max_day*.nc', 'temp_max_day'),
    'precip': ('rain_day/rain_day*.nc', 'rain_day'),
    'solar': ('solar_exposure_day/solar_exposure_day*.nc', 'solar_exposure_day'),
    'wind': ('wind/wind*.nc', 'wind')

}


def get_settings():
    TEST_DATA_PATH = join(BASE_DATA_PATH, 'test_data')
    TRAINING_DATA_PATH = join(BASE_DATA_PATH, 'training')

    CLIMATOLOGIES = {
        'AWAP_DAILY': {
            'solar': (join(BASE_DATA_PATH, 'climatology/climatology_daily_solar_exposure_day.nc'), 'solar_exposure_day')
        }
    }

    settings = {
        'config_options': config_options,
        'DATA_PATHS': {
            'AWRAMS_BASE': AWRAMS_BASE_PATH,
            'BASE_DATA': BASE_DATA_PATH,
            'MASKS': join(BASE_DATA_PATH, 'spatial/masks'),
            'SHAPEFILES': join(BASE_DATA_PATH, 'spatial/shapefiles'),
            'TEST_DATA': TEST_DATA_PATH,
            'MODEL_DATA': join(BASE_DATA_PATH, 'model_data'),
            'CODE': join(AWRAMS_BASE_PATH, 'code'),
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
        'MPI_COMMON_OPTIONS': ['--oversubscribe', '--allow-run-as-root', '--mca plm_rsh_agent false', '-x TMPDIR=/dev/shm/'],
        'REMOTE_SETTINGS': config_options['REMOTE_SETTINGS'].copy(),
        'COMPILER': 'ICC_DEFAULT'
    }

    if config_options['DEV_MODE'] == True:
        DEV_ACTIVATE = 'export PYTHONPATH=$PYTHONPATH:{REPO_PATH}\n'.format(REPO_PATH=config_options['REPO_PATH'])
    else:
        DEV_ACTIVATE = ''

    activation_str = settings['REMOTE_SETTINGS']['ACTIVATION'].format(AWRAMS_BASE_PATH=AWRAMS_BASE_PATH,DEV_ACTIVATE=DEV_ACTIVATE)
    settings['REMOTE_SETTINGS']['ACTIVATION'] = ''.join([s.lstrip(' ')+'\n' for s in activation_str.splitlines()])

    return objectify(settings)
