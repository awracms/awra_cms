from awrams.models.model import Model
import os
import shutil
import glob
from awrams.utils.config_manager import get_model_profile
from awrams.utils.nodegraph.graph import get_dataspecs_from_mapping
from . import runner as fw

class AWRALModel(Model):

    def __init__(self,model_settings):

        #model_settings= get_model_profile('awral',profile).get_settings()

        self.model_settings = model_settings

        self.OUTPUTS = model_settings['OUTPUTS'].copy()

        self._SHARED = None
        
    def get_runner(self,dataspecs,shared=False):
        """
        Return a ModelRunner for this model
        
        Args:
            dataspecs (dict): Dataspecs (as returned from ExecutionGraph)
            shared (bool): Is this runner being used in a shared memory context?

        Returns:
            ModelRunner
        """

        if shared:
            if self._SHARED is not None:
                return fw.FFIWrapper(self.model_settings,mhash=self._SHARED['mhash'],template=self._SHARED['template'])
            else:
                raise Exception("Call init_shared before using multiprocessing")
        else:
            template = fw.template_from_dataspecs(self.get_input_keys(),dataspecs,self.OUTPUTS)
            return fw.FFIWrapper(self.model_settings,False,template)

    def init_shared(self,dataspecs):
        '''
        Call before attempting to use in multiprocessing
        '''

        template = fw.template_from_dataspecs(self.get_input_keys(),dataspecs,self.OUTPUTS)

        builder = fw.AWRALBuilder(self.model_settings)

        mhash = builder.validate_or_rebuild(template)

        self._SHARED = dict(mhash=mhash,template=template)

    def get_input_keys(self):
        """
        Return the list of keys required as inputs

        Returns:
            list

        """
        
        in_map = self.model_settings['CONFIG_OPTIONS']['MODEL_INPUTS']

        model_keys = []

        model_keys += list(in_map['INPUTS_CELL'])
        model_keys += ['init_' + k for k in in_map['STATES_CELL']]
        model_keys += list(in_map['INPUTS_HYPSO'])

        for hru in ('_hrusr','_hrudr'):
            model_keys += ['init_' +k+hru for k in in_map['STATES_HRU']]
            model_keys += [k+hru for k in in_map['INPUTS_HRU']]

        return model_keys


    def get_state_keys(self):
        """
        Return the list of keys representing model states

        Returns:
            list
        """

        in_map = self.model_settings['CONFIG_OPTIONS']['MODEL_INPUTS']

        state_keys = []
        state_keys += [k for k in in_map['STATES_CELL']]
        for hru in ('_hrusr','_hrudr'):
            state_keys += [k+hru for k in in_map['STATES_HRU']]

        return state_keys


    def get_output_variables(self):
        """
        Return the list of output variable keys for this model

        Returns:
            list
        """
        output_vars = []
        for v in self.OUTPUTS['OUTPUTS_AVG'] + self.OUTPUTS['OUTPUTS_CELL']:
            output_vars.append(v)
        for v in self.OUTPUTS['OUTPUTS_HRU']:
            output_vars.extend([v+'_hrusr',v+'_hrudr'])
        return output_vars

    def get_default_mapping(self,profile='default'):
        """
        Return the default input mapping for this model
        This is a dict of key:GraphNode mappings

        Return:
            mapping (dict)
        """

        raise Exception("Use model_profile.get_input_mapping instead")

    def rebuild_for_input_map(self,input_map,force=True):
        dataspecs = get_dataspecs_from_mapping(input_map,True)
        template = fw.template_from_dataspecs(self.get_input_keys(),dataspecs,self.OUTPUTS)
        builder = fw.AWRALBuilder(self.model_settings)
        mhash = builder.validate_or_rebuild(template,force=force)

        return mhash

    def clear_build_cache(self):
        build_path = self.model_settings['BUILD_SETTINGS']['BUILD_DEST_PATH']
        shutil.rmtree(build_path,ignore_errors=True)
        os.makedirs(build_path,exist_ok=True)

        
