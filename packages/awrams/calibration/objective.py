import multiprocessing as mp
import numpy as np
import pandas as pd
import os
from awrams.calibration.support import aggregate, flattened_areas

from awrams.utils.messaging import Message
from awrams.utils.messaging.general import get_traceback

def _map_input_group(input_schema):
    out_map = {}
    for k,v in input_schema.items():
        in_var = v.input_var
        if in_var not in out_map:
            out_map[in_var] = {}
        out_map[in_var][k] = v.agg_method
    return out_map

class ObjectiveProcess(mp.Process):
    '''
    Multiprocessing wrapper that receives model results and evaluates objective functions
    '''

    def __init__(self,buffer_manager,data_index_map,extent_map,run_period,objective_spec,owned_catchments):
        super().__init__()
        self.node_to_objective_q = mp.Queue()
        self.objective_to_node_q = mp.Queue()
        self.buffer_manager = buffer_manager
        self.data_index_map = data_index_map
        self.objective_spec = objective_spec
        self.owned_catchments = owned_catchments
        self.model_outputs = _map_input_group(objective_spec.catchment_objf.inputs_from_model)

        self.vol_weights = {}
        self.mean_weights = {}
        #+++ Move this calculation to main node loop/MPI collection
        for k in self.data_index_map.keys():
            e = extent_map[k]
            self.vol_weights[k] = flattened_areas(e)#<- mean over area
            self.mean_weights[k] = self.vol_weights[k] / e.area
            #self.weights[k] = flattened_areas(e) #<- volume summing

        eval_p = objective_spec.eval_period

        self.eval_index = np.s_[(eval_p[0]-run_period[0]).days:(eval_p[-1]-run_period[0]).days + 1]

    def build_objectives(self):

        '''
        Load observations
        '''
        observations = dict([(cid,dict()) for cid in self.owned_catchments])

        for ob_key,ob_val in self.objective_spec.observations.items():
            cur_obs = load_obs(ob_val)
            for cid in self.owned_catchments:
                observations[cid][ob_key] = np.array(cur_obs[cid][self.objective_spec.eval_period])

        '''
        Set up objective function (per catchment)
        '''

        objectivef = {}

        for cid in self.owned_catchments:
            objectivef[cid] = load_catchment_objective(self.objective_spec.catchment_objf,observations[cid],self.objective_spec.eval_period)

        return objectivef

    def run(self):
        try:
            
            
            # No sched_setaffinity in Windows, we handle this in MSMPI
            if os.name is not 'nt':
                all_cores = set(range(mp.cpu_count()))
                os.sched_setaffinity(0,all_cores)

            self.objectivef = self.build_objectives()
            
            self.buffer_manager.rebuild_buffers()

            while(True):
                msg = self.node_to_objective_q.get()

                if msg.subject == 'terminate':
                    return

                if msg.subject == 'evaluate':

                    catchment_id,buffer_id,params,data_index,translate = msg.content

                    indata = self.buffer_manager.map_buffer(buffer_id,data_index)

                    #for s in self.schema['model_outputs']:
                    #    _ = indata[s].T

                    lumped = dict(params=params)

                    for s,mapping in self.model_outputs.items():
                        cur_data = indata[s]
                        if translate:
                            cur_data = cur_data.T

                        for out_var,agg_method in mapping.items():
                            if agg_method == 'flat':
                                lumped[out_var] = cur_data
                            else:
                                if agg_method == 'mean':
                                    weights = self.mean_weights
                                elif agg_method == 'volume':
                                    weights = self.vol_weights

                                lumped[out_var] = aggregate(cur_data[self.eval_index],weights[catchment_id])

                    result = self.objectivef[catchment_id].evaluate(lumped)

                    self.objective_to_node_q.put(Message('catch_result',catchment=catchment_id,result=result))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.objective_to_node_q.put(Message('exception',exception=e,traceback=get_traceback()))



def load_obs(obs_spec):
    '''
    Right now this just loads a CSV file; expand to other sources (particularly NC/H5 for gridded data)
    '''
    return pd.DataFrame.from_csv(obs_spec)

def load_catchment_objective(objective_spec,obs,eval_period):
    from awrams.utils.nodegraph.nodes import funcspec_to_callable

    objf_class = funcspec_to_callable(objective_spec.funcspec)
    objf = objf_class(obs,eval_period,**objective_spec.init_args)
    return objf