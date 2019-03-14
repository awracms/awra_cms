'''
Default configuration for AWRA-L model (version 6.0)

DO NOT EDIT

Other profiles build on this configuration; if you wish to make modifications,
derive from this profile using get_model_profile (see v6_testing.py for an example)

For more significant modifications, create a new copy of this profile

'''

from os.path import dirname as _dirname
from awrams.utils.general import join_paths as _join
from os.path import expanduser as _expanduser
from awrams.utils import config_manager
from awrams.utils.metatypes import objectify
from awrams.models.awral import transforms, support
from awrams.utils.nodegraph import nodes
import numpy as np
from collections import OrderedDict
from awrams.models.awral.model import AWRALModel
from awrams.utils import parameters

# General configuration options
config_options = {
    
    # Model version string; used for paths in the build options

    'MODEL_VERSION' : 'v5',

    # List of parameterised model inputs.  The values are populated at runtime (normally from
    # a JSON file, or CalibrationResults HDF5)

    # Values are (min, max, fixed)
    # Where min, max are the (default) calibration ranges, and fixed a bool representing whether
    # the parameter is constant (True) or free (False) (ie free parameters will be calibrated)

    'PARAMETER_KEYS': ['alb_dry_hrudr', 'alb_dry_hrusr', 'alb_wet_hrudr', 'alb_wet_hrusr', 
                       'cgsmax_hrudr', 'cgsmax_hrusr', 'er_frac_ref_hrudr', 'fsoilemax_hrudr', 
                       'fsoilemax_hrusr', 'fvegref_g_hrudr', 'fvegref_g_hrusr', 'gfrac_max_hrudr', 
                       'gfrac_max_hrusr', 'hveg_hrusr', 'k_gw_scale', 'k_rout_int', 'k_rout_scale', 
                       'k0sat_scale', 'kdsat_scale', 'kr_coeff', 'kssat_scale', 'lairef_hrudr', 
                       'lairef_hrusr', 'ne_scale', 'pref_gridscale', 'rd_hrudr', 'rd_hrusr',
                       's_sls_hrudr', 's_sls_hrusr', 's0max_scale', 'sdmax_scale', 'sla_hrudr', 
                       'sla_hrusr', 'slope_coeff', 'ssmax_scale', 'tgrow_hrudr', 'tgrow_hrusr', 
                       'tsenc_hrudr', 'tsenc_hrusr', 'ud0_hrudr', 'ud0_hrusr', 'us0_hrudr', 
                       'us0_hrusr', 'vc_hrudr', 'vc_hrusr', 'w0lime_hrudr', 'w0lime_hrusr', 
                       'w0ref_alb_hrudr', 'w0ref_alb_hrusr', 'wdlimu_hrudr', 'wdlimu_hrusr',
                       'wslimu_hrudr', 'wslimu_hrusr'], 

    'MODEL_INPUTS': {

        "STATES_CELL": {
            "sg": "Groundwater storage (mm)",
            "sr": "Surface storage (mm)"
        },

        "STATES_HRU": {
            "mleaf": "Vegetation index",
            "s0": "Top soil moisture (mm)",
            "sd": "Deep soil moisture (mm)",
            "ss": "Shallow soil moisture (mm)"
        },

        "INPUTS_CELL": {
            "avpt": "Vapour pressure",
            "k0sat": "Hydraulic saturation (top)",
            "k_gw": "Groundwater drainage coefficient",
            "k_rout": "k_rout",
            "kdsat": "Hydraulic saturation (deep)",
            "kr_0s": "Interlayer saturation (top/shallow)",
            "kr_coeff": "kr_coeff",
            "kr_sd": "Interlayer saturation (shallow/deep)",
            "kssat": "Hydraulic saturation (shallow)",
            "pair": "Air pressure (pA)",
            "prefr": "prefr",
            "pt": "Precipitation (mm/day)",
            "radcskyt": "radcskyt",
            "rgt": "Solar radiation (MJ m^-2)",
            "s0max": "Maximum soil moisture (top)",
            "sdmax": "Maximum soil moisture (deep)",
            "slope": "slope",
            "slope_coeff": "slope_coeff",
            "ssmax": "Maximum soil moisture (shallow)",
            "tat": "Temperature average (degC)",
            "u2t": "Windspeed (m/s)"
        },

        "INPUTS_HRU": {
            "alb_dry": "Dry Soil Albedo",
            "alb_wet": "Wet Soil Albedo",
            "cgsmax": "Conversion Coefficient From Vegetation Photosynthetic Capacity Index to Maximum Stomatal Conductance",
            "er_frac_ref": "Ratio of Average Evaporation Rate Over Average Rainfall Intensity During Storms Per Unit Canopy Cover",
            "fhru": "Fraction of Simulation Cell with <HRU-type> Vegetation",
            "fsoilemax": "Soil Evaporation Scaling Factor When Soil Water Supply is Not Limiting Evaporation",
            "hveg": "Height of Vegetation Canopy",
            "laimax": "Maximum Leaf Area Index",
            "lairef": "Reference Leaf Area Index (at which fveg = 0.63)",
            "rd": "Root depth",
            "s_sls": "Specific Canopy Rainfall Storage Capacity Per Unit Leaf Area",
            "sla": "Specific Leaf Area",
            "tgrow": "Characteristic Time Scale for Vegetation Growth Towards Equilibrium",
            "tsenc": "Characteristic Time Scale for Vegetation Senescence Towards Equilibrium",
            "ud0": "Maximum Root Water Uptake Rates From Deep Soil",
            "us0": "Maximum Root Water Uptake Rates From Shallow Soil",
            "vc": "Vegetation Photosynthetic Capacity Index Per Unit Canopy Cover",
            "w0lime": "Relative Top Soil Water Content at Which Evaporation is Reduced",
            "w0ref_alb": "Reference Value of w0 Determining the Rate of Albedo Decrease With Wetness",
            "wdlimu": "Deep Water-Limiting Relative Water Content",
            "wslimu": "Shallow Water-Limiting Relative Water Content"
        },

        "INPUTS_HYPSO": {
            "height": "height",
            "hypsperc": "hypsperc",
            "ne": "ne"
        }
    },


    # Default (in-memory) model outputs.

    'OUTPUTS' : dict(
        # Outputs from individual HRUs
        OUTPUTS_HRU = ['s0', 'ss', 'sd', 'mleaf'],  
        # Outputs that are a weighted average of individual HRUs
        OUTPUTS_AVG = ['e0', 'etot', 'dd', 's0', 'ss', 'sd'],
        # Cell level outputs; ie those not contained within the HRU
        OUTPUTS_CELL = ['qtot', 'sr', 'sg'] 
    ),

    #Core Code Options
    #    - compiler settings
    #    - source/target file locations

    'BUILD_STRINGS' : dict(
        ICC_DEFAULT = "icc %s -march=native -std=c99 -static-intel --shared -fPIC -O3 -o %s",
        GCC_DEFAULT = "gcc %s -std=c99 --shared -fPIC -O3 -o %s",
        CLANG_WINDOWS = "clang %s --shared -std=c99 -O3 -o %s"
    )
}

config_options = objectify(config_options)

def get_settings(sys_settings=None):

    if sys_settings is None:
        sys_settings = config_manager.get_system_profile().get_settings()

    model_data_path = sys_settings['DATA_PATHS']['MODEL_DATA']
    model_code_path = _join(sys_settings['DATA_PATHS']['CODE'],'models')
    model_build_path = _join(sys_settings['DATA_PATHS']['BUILD_CACHE'],'models')

    version = config_options['MODEL_VERSION']

    model_config_path = config_manager.get_config_path('models/awral')

    profile = {
        'SPATIAL_FILE': _join(model_data_path, 'awral/spatial_parameters_v5.h5'),
        'PARAMETER_FILE': _join(model_config_path, 'parameters/DefaultParameters_v5.json'),
        'OUTPUTS': config_options['OUTPUTS'],
        'CLIMATE_DATASET': sys_settings['CLIMATE_DATASETS']['TRAINING'],
        'BUILD_SETTINGS': {
            'SRC_FILENAME_BASE': 'awral',
            'CORE_SRC_PATH': _join(model_code_path, 'awral/%s' % version),
            'BUILD_STR': config_options['BUILD_STRINGS'][sys_settings['COMPILER']],
            'BUILD_DEST_PATH': _join(model_build_path, 'awral/%s' % version)
        },
        'CONFIG_OPTIONS': config_options
    }

    return objectify(profile)


def get_model(model_settings=None):
    if model_settings is None:
        model_settings = get_settings()

    return AWRALModel(model_settings)

def get_input_mapping(model_settings=None):
    """
    Return the default input mapping for this model
    This is a dict of key:GraphNode mappings

    Return:
        mapping (dict)
    """

    if model_settings is None:
        model_settings = get_settings()

    mapping = {}

    #for k,spec in config_options['PARAMETER_SPECS'].items():
    #    mapping[k] = nodes.parameter_from_json(
    #        model_settings['PARAMETER_FILE'], k, spec[0],spec[1],spec[2])

    param_df = parameters.wirada_json_to_param_df(model_settings['PARAMETER_FILE'])
    mapping = parameters.param_df_to_mapping(param_df,mapping)

    SPATIAL_GRIDS = ['f_tree', 'height', 'hveg_dr', 'k0sat_v5', 'k_gw', 'kdsat_v5', 'kssat_v5', 'lai_max',
                     'meanPET', 'ne', 'pref', 's0fracAWC', 'slope', 'ssfracAWC', 'windspeed']

    for grid in SPATIAL_GRIDS:
        mapping[grid.lower()+'_grid'] = nodes.spatial_from_file(
            model_settings['SPATIAL_FILE'], 'parameters/%s' % grid)

    FORCING_DATA = model_settings['CLIMATE_DATASET']['FORCING']
    CLIMATOLOGY = model_settings['CLIMATE_DATASET']['CLIMATOLOGY']

    for k in ['tmin', 'tmax', 'precip']:
        var_map = FORCING_DATA['MAPPING'][k]
        mapping[k+'_f'] = nodes.forcing_from_ncfiles(FORCING_DATA['PATH'], var_map[0], var_map[1])

    mapping['solar_f'] = nodes.forcing_gap_filler(FORCING_DATA['PATH'],FORCING_DATA['MAPPING']['solar'][0], \
                                                  FORCING_DATA['MAPPING']['solar'][1],CLIMATOLOGY['solar'][0])

    mapping.update({
        'tmin': nodes.transform(np.minimum, ['tmin_f', 'tmax_f']),
        'tmax': nodes.transform(np.maximum, ['tmin_f', 'tmax_f']),
        'hypsperc_f': nodes.const_from_hdf5(model_settings['SPATIAL_FILE'], 'dimensions/hypsometric_percentile', ['hypsometric_percentile']),
        # Model needs 0-1.0, file represents as 0-100
        'hypsperc': nodes.mul('hypsperc_f', 0.01),
        'fday': transforms.fday(),
        'u2t': transforms.u2t('windspeed_grid','fday')
    })

    mapping['height'] = nodes.assign('height_grid')

    mapping['er_frac_ref_hrusr'] = nodes.mul('er_frac_ref_hrudr', 0.5)

    mapping['k_rout'] = nodes.transform(
        transforms.k_rout, ('k_rout_scale', 'k_rout_int', 'meanpet_grid'))
    mapping['k_gw'] = nodes.mul('k_gw_scale', 'k_gw_grid')

    mapping['s0max'] = nodes.mul('s0max_scale', 's0fracawc_grid', 100.)
    mapping['ssmax'] = nodes.mul('ssmax_scale', 'ssfracawc_grid', 900.)
    mapping['sdmax'] = nodes.mul('ssmax_scale','sdmax_scale','ssfracawc_grid',5000.)

    mapping['k0sat'] = nodes.mul('k0sat_scale', 'k0sat_v5_grid')
    mapping['kssat'] = nodes.mul('kssat_scale', 'kssat_v5_grid')
    mapping['kdsat'] = nodes.mul('kdsat_scale', 'kdsat_v5_grid')

    mapping['kr_0s'] = nodes.transform(
        transforms.interlayer_k, ('k0sat', 'kssat'))
    mapping['kr_sd'] = nodes.transform(
        transforms.interlayer_k, ('kssat', 'kdsat'))

    mapping['prefr'] = nodes.mul('pref_gridscale', 'pref_grid')
    mapping['fhru_hrusr'] = nodes.sub(1.0, 'f_tree_grid')
    mapping['fhru_hrudr'] = nodes.assign('f_tree_grid')
    mapping['ne'] = nodes.mul('ne_scale', 'ne_grid')
    mapping['slope'] = nodes.assign('slope_grid')
    mapping['hveg_hrudr'] = nodes.assign('hveg_dr_grid')

    mapping['laimax_hrusr'] = nodes.assign('lai_max_grid')
    mapping['laimax_hrudr'] = nodes.assign('lai_max_grid')

    mapping['pair'] = nodes.const(97500.)

    mapping['pt'] = nodes.assign('precip_f')
    mapping['rgt'] = nodes.transform(np.maximum, ['solar_f', 0.1])
    mapping['tat'] = nodes.mix('tmin', 'tmax', 0.75)
    mapping['avpt'] = nodes.transform(transforms.pe, 'tmin')
    mapping['radcskyt'] = transforms.radcskyt()

    mapping['init_sr'] = nodes.const(0.0)
    mapping['init_sg'] = nodes.const(100.0)
    for hru in ('_hrusr', '_hrudr'):
        mapping['init_mleaf'+hru] = nodes.div(2.0, 'sla'+hru)
        for state in ["s0", "ss", "sd"]:
            mapping['init_'+state+hru] = nodes.mul(state+'max', 0.5)

    return objectify(mapping)
