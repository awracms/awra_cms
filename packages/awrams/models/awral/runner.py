import cffi
import numpy as np
#from .template import _SOURCE_FN,_SOURCE_T_FN,_HEADER_FN,_HEADER_T_FN,_LIB_FN
from numbers import Number
import os
import shutil
import tempfile
from awrams.models.model import ModelRunner
from awrams.utils import config_manager
from hashlib import md5
from awrams.utils.awrams_log import get_module_logger

from .template import gen_templates, BASE_TEMPLATE
from awrams.utils import templates

logger = get_module_logger()

TYPEMAP = {np.float64: "double *", np.float32: "float *", np.dtype('float64'): "double *", np.dtype('float32'): "float *"}

def ccast(ndarr,ffi,to_type=np.float64,promote=True):
    if ndarr.dtype != to_type:
        if promote:
            ndarr = ndarr.astype(to_type)
        else:
            raise Exception("Incorrect dtype",ndarr.dtype,to_type)

    typestr = TYPEMAP[to_type]
    return ffi.cast(typestr,ndarr.ctypes.data)

def build_model(build_str):
    import subprocess

    #build_str = build_str % (source_filename,lib_filename)

    callstr = build_str.split(' ')
    from subprocess import Popen, PIPE
    pipe = Popen(callstr, stdout=PIPE,stderr=PIPE)
    out,err = pipe.communicate()

    return out,err

def get_lib_extension():
    if os.name == 'nt':
        return '.dll'
    else:
        return '.so'

def filename_for_hash(base_name,mhash,ext):
    return base_name + '_' + mhash + ext

def get_tmp_path():
    '''
    +++ Currently assumes Linux, will update for Windows version soon...
    '''

class AWRALBuilder:
    def __init__(self,model_settings):
        self.model_settings = model_settings

    def validate_or_rebuild(self,template=None,force=False):
        '''
        Checks whether an existing compiled header/library exist for this template
        Will recompile if not
        '''
        if template is None:
            template = BASE_TEMPLATE

        build_settings = self.model_settings['BUILD_SETTINGS']

        source_path = build_settings['CORE_SRC_PATH']
        build_path = build_settings['BUILD_DEST_PATH']

        os.makedirs(build_path,exist_ok=True)

        build_string_base = build_settings['BUILD_STR']
        filename_base = build_settings['SRC_FILENAME_BASE']

        source_t_filename = os.path.join(source_path,filename_base+'_t.c')
        header_t_filename = os.path.join(source_path,filename_base+'_t.h')

        mhash = model_hash(template,build_string_base,source_t_filename,header_t_filename)

        full_build_basename = os.path.join(build_path,filename_base)

        # Hashed filenames that will be stored in our build cache
        source_cache_filename = filename_for_hash(full_build_basename,mhash,'.c')
        header_cache_filename = filename_for_hash(full_build_basename,mhash,'.h')
        
        lib_cache_filename = filename_for_hash(full_build_basename,mhash,get_lib_extension())

        

        rebuild=force

        if not os.path.exists(header_cache_filename):
            rebuild = True

        if not os.path.exists(lib_cache_filename):
            rebuild = True

        if rebuild:

            #+++ build_str
            # Where to source? Refactor?

            logger.info("Rebuilding model")

            temp_dir = tempfile.mkdtemp()

            cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Get a unique temporary directory for this build

                # Construct template-filled files in the temp dir, use these for build
                source_build_filename = os.path.join(temp_dir,filename_base+'.c')
                header_build_filename = os.path.join(temp_dir,filename_base+'.h')
                lib_build_filename = os.path.join(temp_dir,filename_for_hash(filename_base,mhash,get_lib_extension()))

                build_string_full = build_string_base % (source_build_filename,lib_build_filename)
                
                template_filled = gen_templates(template)

                templates.transform_file(source_t_filename,source_build_filename,template_filled)
                templates.transform_file(header_t_filename,header_build_filename,template_filled)

                out,err = build_model(build_string_full)
                out,err = out.decode(),err.decode()
                if 'error' in out or 'error' in err:
                    logger.critical(out)
                    logger.critical(err)
                    raise Exception("Model build failed")
                else:
                    logger.info(out)
                    logger.info(err)
                
                shutil.copyfile(source_build_filename,source_cache_filename)
                shutil.copyfile(header_build_filename,header_cache_filename)
                shutil.copyfile(lib_build_filename,lib_cache_filename)
            
                logger.info("Build completed")

            finally:
                os.chdir(cwd)
                shutil.rmtree(temp_dir,True)


        return mhash

def model_hash(template,build_string,source_t_filename,header_t_filename):
    outstr = open(header_t_filename).read()
    outstr = outstr + open(source_t_filename).read()
    for k in sorted(template.keys()):
        outstr = outstr + k + str(sorted(template[k]))
    return md5(outstr.encode()).hexdigest()


def template_from_dataspecs(model_keys,dspec,outputs):

    new_template = dict(zip(BASE_TEMPLATE.keys(),[[] for k in BASE_TEMPLATE]))

    for k in ['OUTPUTS_AVG','OUTPUTS_CELL','OUTPUTS_HRU']:
        new_template[k] = outputs[k]

    hru_keys_base = [k for k in model_keys if '_hru' in k and not k.startswith('init_')]
    hru_keys = np.unique([k[:-6] for k in hru_keys_base])

    for k in hru_keys:
        d0 = dspec[k+'_hrusr']
        d1 = dspec[k+'_hrudr']
        mdims = max(len(d0.dims),len(d1.dims))
        if(mdims):
            ktype = 'INPUTS_SPATIAL_HRU'
        else:
            ktype = 'INPUTS_SCALAR_HRU'
            
        new_template[ktype].append(k)

    for k in model_keys:
        if not k in hru_keys_base \
            and not k.startswith('init_') and not k in ['height','hypsperc']:
            dims = dspec[k].dims
            if dims == ['time','cell']:
                new_template['INPUTS_FORCING'].append(k)
            elif dims == ['cell']:
                new_template['INPUTS_SPATIAL'].append(k)
            elif not len(dims):
                new_template['INPUTS_SCALAR'].append(k)
            else:
                print(k,dims) 
            
    return new_template

class FFIWrapper(ModelRunner):
    def __init__(self,model_settings,force_build=False,template=None,mhash=None):

        self.model_settings = model_settings
        self.builder = AWRALBuilder(model_settings)

        if mhash is not None:
            self.template = template
            self._init_ffi(mhash)
        else:
            self.reload(force_build,template)

        self._output_arr_dict = None

    def _invalidate(self):
        import gc

        gc.collect()

        self.ffi = None

        self.awralib = None

        self.forcing = None
        self.outputs = None
        self.states = None
        self.parameters = None
        self.spatial = None
        self.hruspatial = None
        self.hruparams = None
        self.hypso = None

        gc.collect()

    def _init_ffi(self,mhash):

        _imeta = self.model_settings['CONFIG_OPTIONS']['MODEL_INPUTS']
        self._STATE_KEYS = list(_imeta['STATES_CELL'])
        self._STATE_KEYS_HRU = list(_imeta['STATES_HRU'])
        self._HYPSO_KEYS = list(_imeta['INPUTS_HYPSO'])

        from cffi import FFI
        self.ffi = FFI()

        model_import_path = self.model_settings['BUILD_SETTINGS']['BUILD_DEST_PATH']
        filename_base = self.model_settings['BUILD_SETTINGS']['SRC_FILENAME_BASE']

        full_basename = os.path.join(model_import_path,filename_base)

        header_fn = filename_for_hash(full_basename,mhash,'.h')
        lib_fn = filename_for_hash(full_basename,mhash,get_lib_extension())

        with open(header_fn,'r') as fh:
            header_str = fh.read()
            header_str = ''.join([l+'\n' for l in header_str.splitlines() if not (l.startswith('#') or l.startswith('EXPORT_EXTENSION'))])
            self.ffi.cdef(header_str)

        self.awralib = self.ffi.dlopen(lib_fn)

        self.forcing = self.ffi.new("Forcing*")
        self.outputs = self.ffi.new("Outputs*")
        self.states = self.ffi.new("States *")
        self.parameters = self.ffi.new("Parameters *")
        self.spatial = self.ffi.new("Spatial *")
        self.hruspatial = self.ffi.new("HRUSpatial[2]")
        self.hruparams = self.ffi.new("HRUParameters[2]")
        self.hypso = self.ffi.new("Hypsometry *")

        self._state_vals = {}

    def reload(self,force_build=False,template=None):

        self._invalidate()

        if template is None:
            template = BASE_TEMPLATE

        mhash = self.builder.validate_or_rebuild(template,force=force_build)

        self.template = template

        self._init_ffi(mhash)

    def _cast(self,k,ndarr,to_type=np.float64,promote=True,target=None,force_copy=False):
        '''
        Ensures inputs are in correct datatypes for model.
        '''
        if target is None:
            target = self._temp_cast

        if not ndarr.flags['C_CONTIGUOUS']:
            ndarr = ndarr.flatten()

        if ndarr.dtype != to_type:
            if promote:
                ndarr = ndarr.astype(to_type)
            else:
                raise Exception("Incorrect dtype",ndarr.dtype,to_type)

        if force_copy:
            ndarr = ndarr.copy()

        target[k] = ndarr

        typestr = TYPEMAP[to_type]

        return self.ffi.cast(typestr,ndarr.ctypes.data)

    def _promote(self,k,v,shape,target=None):
        if target is None:
            target = self._temp_cast
        if isinstance(v,Number):
            out = np.empty(shape,dtype=np.float64)
            out[...] = v
            target[k] = v
            return out
        else:
            target[k] = v
            return v

    def _promote_except(self,v,shape):
        if isinstance(v,Number):
            raise Exception("Scalar %s supplied for spatial value" %v)
        else:
            return v

    def run_over_dimensions(self,inputs,dims):
        return self.run_from_mapping(inputs,dims['time'],dims['cell'])

    def run_from_mapping(self,mapping,timesteps,cells,recycle_states=True):
        #forcing_np = {}
        #forcealive = []

        self._temp_cast = {}

        promote = self._promote

        #for k in forcing_args:
        for k in self.template['INPUTS_FORCING']:
            nval = promote(k,mapping[k],(timesteps,cells))
            self.forcing.__setattr__(k,self._cast(k,nval,np.float32))

        if self._output_arr_dict is not None:
            outputs_np = self._output_arr_dict
        else:
            outputs_np = {}
        #outputs_hru_np = []
        
        ALL_OUTPUTS = self.template['OUTPUTS_AVG'] + self.template['OUTPUTS_CELL']

        for k in ALL_OUTPUTS:
            if k not in outputs_np:
                outputs_np[k] = arr = np.empty((timesteps,cells))
            self.outputs.__setattr__(k,self._cast(k,arr))

        for hru in range(2):
            for k in self.template['OUTPUTS_HRU']:
                full_k = k+'_hrusr' if hru is 0 else k+'_hrudr'
                if full_k not in outputs_np:
                    outputs_np[full_k] = arr = np.empty((timesteps,cells))
                self.outputs.hru[hru].__setattr__(k,self._cast(full_k,arr))

        if (len(self._state_vals) == 0):
            recycle_states = False

        if not recycle_states:
            self._state_vals = {}
            for k in self._STATE_KEYS:
                nval = promote(k,mapping['init_'+k],(cells,),target=self._state_vals)
                self.states.__setattr__(k,self._cast(k,nval,target=self._state_vals,force_copy=True))

            for k in self._STATE_KEYS_HRU:
                nval0 = promote(k+'_hrusr',mapping['init_'+k+'_hrusr'],(cells,),target=self._state_vals)
                self.states.hru[0].__setattr__(k,self._cast(k+'_hrusr',nval0,target=self._state_vals,force_copy=True))
                nval1 = promote(k+'_hrudr',mapping['init_'+k+'_hrudr'],(cells,),target=self._state_vals)
                self.states.hru[1].__setattr__(k,self._cast(k+'_hrudr',nval1,target=self._state_vals,force_copy=True))

        for k in self.template['INPUTS_SCALAR']:
            rs = mapping[k]
            self.parameters.__setattr__(k,rs)

        for k in self.template['INPUTS_SCALAR_HRU']:
            self.hruparams[0].__setattr__(k,mapping[k+'_hrusr'])
            self.hruparams[1].__setattr__(k,mapping[k+'_hrudr'])

        for k in self.template['INPUTS_SPATIAL']:
            nval = promote(k,mapping[k],(cells,))
            self.spatial.__setattr__(k,self._cast(k,nval))

        for k in self.template['INPUTS_SPATIAL_HRU']:
            self.hruspatial[0].__setattr__(k,self._cast(k+'_hrusr',promote(k+'_hrusr',mapping[k+'_hrusr'],(cells,))))
            self.hruspatial[1].__setattr__(k,self._cast(k+'_hrudr',promote(k+'_hrudr',mapping[k+'_hrudr'],(cells,))))
            
        for k in self._HYPSO_KEYS:
            nval = mapping[k]
            if k == 'height': #+++ Right now our hypso grids present data in the opposite order to the model's expectation
                nval = nval.T.astype(np.float64).flatten()
                self._temp_cast[k] = nval
            self.hypso.__setattr__(k,self._cast(k,nval))

        self.awralib.awral(self.forcing[0],self.outputs[0],self.states[0],\
            self.parameters[0],self.spatial[0],self.hypso[0],self.hruparams,self.hruspatial,timesteps,cells)

        self._temp_cast = {}

        outputs_np['final_states'] = self._state_vals
        
        return outputs_np

    def set_output_arrays(self,out_dict,timesteps,cells):
        '''
        Outputs will be written to a <outputs>, a user supplied set of numpy arrays
        They must be of the same dimensions as the corresponding run_from_mapping call
        The model may also write additional outputs; this call only guarantees that keys appearing in out_dict
        will be written to the corresponding arrays.
        Currently ignores the final states output group (ie this will always be generated at runtime; not user supplied)
        '''
        def validate(k,in_arr,timesteps,cells):
            out_type = np.float64
            if not in_arr.flags['C_CONTIGUOUS']:
                raise Exception("%s : Non-contiguous array" % k)
            if in_arr.dtype != out_type:
                raise Exception("%s : Mismatched type (%s,%s)" % (k,in_arr.dtype,out_type))
            if np.prod(in_arr.shape) != (timesteps*cells):
                raise Exception("%s : Mismatched array size" % k)

        OUTPUTS_HRU = []
        for hru in ['_hrusr','_hrudr']:
            for k in self.template['OUTPUTS_HRU']:
                OUTPUTS_HRU.append(k+'_hrusr')
        ALL_OUTPUTS = self.template['OUTPUTS_AVG'] + self.template['OUTPUTS_CELL'] + OUTPUTS_HRU

        outputs_np = {}

        for k,v in out_dict.items():
            if k not in ALL_OUTPUTS:
                raise Exception("Output %s requested but not produced by model" %k)
            validate(k,v,timesteps,cells)
            outputs_np[k] = v
        
        self._output_arr_dict = outputs_np

