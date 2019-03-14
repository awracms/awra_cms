import multiprocessing as mp
try:
    mp.set_start_method('forkserver')
    p = mp.Process()
    p.start()
    p.join()
    import os
    print("s: ",os.getpid())
except:
    pass

from awrams.utils.nodegraph.nodes import callable_to_funcspec, funcspec_to_callable
from awrams.utils.nodegraph.graph import get_dataspecs_from_mapping, ExecutionGraph

from awrams.utils.parameters import input_map_to_param_df

from awrams.calibration.allocation import allocate_catchments_to_nodes
from awrams.calibration.optimizer import OptimizerProcess
from awrams.calibration.support import *
from .logger import CalibrationLoggerProcess
from awrams.utils.messaging import Message
import numpy as np
from collections import OrderedDict
from time import sleep
import pickle
from awrams.utils.messaging.general import get_traceback

from .mpi_support import *

import signal

import mpi4py

mpi4py.rc(finalize=True)

def sigterm_handler(signal,frame):
    raise KeyboardInterrupt

def run_server(cal_spec):

    #(optimizer_spec,objective_spec,extent_map,node_alloc,catch_node_map,run_period,node_mapping,model,n_workers,logfile):

    prerun_action = cal_spec.get('prerun_action')
    if prerun_action is not None:
        prerun_action = funcspec_to_callable(prerun_action)
        prerun_action(cal_spec)

    optimizer_spec = cal_spec['optimizer_spec']
    objective_spec = cal_spec['objective_spec']
    extent_map = cal_spec['extent_map']
    node_alloc = cal_spec['node_alloc']
    catch_node_map = cal_spec['catch_node_map']
    run_period = cal_spec['run_period']
    node_mapping = cal_spec['node_mapping']
    model = cal_spec['model']
    n_workers = cal_spec['n_workers']
    n_sub_workers = cal_spec['n_sub_workers'] # cores per node
    logfile = cal_spec['logfile']


    model.init_shared(get_dataspecs_from_mapping(node_mapping,True))

    from mpi4py import MPI
    comm_world = MPI.COMM_WORLD

    '''
    Build optimizer
    '''
    #params = get_opt_parameters(run_graph)

    #opt_class = funcspec_to_callable(optimizer_spec.func_spec)
    #opt_args = optimizer_spec.opt_args
    #opt_args['parameters'] = params

    #
    import os
    #all_cores = set(range(mp.cpu_count()))

    #os.sched_setaffinity(0,all_cores)

    pspace = get_params_from_mapping(node_mapping)

    param_df = input_map_to_param_df(node_mapping)

    optimizer = OptimizerProcess(pspace,optimizer_spec)
    optimizer.daemon = True
    optimizer.start()



    '''
    Set up global objective
    '''

    local_objf_size = len(objective_spec.catchment_objf.output_schema)
    #global_objective = objective_spec['global_objf']

    '''
    Allocate extents to nodes
    '''
    #node_alloc, catch_node_map, _, __ = allocate_catchments_to_nodes(extent_map,nworkers)

    '''
    +++ Shift this logic to the allocation module?
    '''

    owning_workers = sorted([k+1 for k,v in node_alloc.items() \
                  if len([c for c in v['catchments'] if c[2]['owns']]) > 0])

    owned_counts = dict([(k,len([c for c in v['catchments'] if c[2]['owns']])) for k,v in node_alloc.items()])
    owned_counts = dict(filter(lambda x: x[1] > 0, owned_counts.items()))
    owned_counts = OrderedDict(sorted(owned_counts.items(),key = lambda x: x[0]))
    owned_counts = np.array(list(owned_counts.values()))

    objf_split_counts = [0] + list(owned_counts * local_objf_size)
    objf_split_offsets = [0] + list(np.cumsum(objf_split_counts)[:-1])

    '''
    Build ordered list of catchments, as received from nodes during objective function gather
    '''
    local_ids = []

    for i in range(len(node_alloc)):
        node_catch = node_alloc[i]['catchments']
        for c in node_catch:
            if c[2]['owns']:
                local_ids.append(c[0])

    '''
    Set up logger
    '''
    paramnames = list(pspace.params.keys())

    lschema = objective_spec.catchment_objf.output_schema
    gschema = objective_spec.global_objf.output_schema

    signal.signal(signal.SIGTERM, sigterm_handler)

    logger = CalibrationLoggerProcess(paramnames,local_ids,logfile,lschema,gschema,param_df,optimizer_spec.opt_meta)
    #logger.daemon = True
    logger.start()

   

    owning_incl = [0] + owning_workers

    for n in range(n_workers):
        node_message = {}
        node_message['owning_incl'] = owning_incl
        node_message['non_worker_exclusions'] = []
        node_message['extent_map_full'] = extent_map
        node_message['run_period'] = run_period
        node_message['objective_spec'] = objective_spec
        node_message['node_alloc'] = node_alloc
        node_message['catch_node_map'] = catch_node_map
        node_message['n_sub_workers'] = n_sub_workers
        node_message['objf_split_counts'] = objf_split_counts
        node_message['objf_split_offsets'] = objf_split_offsets
        node_message['node_mapping'] = node_mapping
        node_message['model'] = model

        comm_world.send(node_message,dest=n+1)

    '''
    Set up MPI
    '''


    g_world = comm_world.Get_group() 
    g_control_workers = g_world.Excl([])
    g_control_owning = g_world.Incl(owning_incl)

    comm_control_workers = comm_world.Create(g_control_workers)

    comm_control_owning = comm_world.Create(g_control_owning)


    waiting_tasks = {}
   
    global_objective = objective_spec.global_objf()
    gkey = global_objective.objective_key
        

    try:
        error_buf = bytearray(8 * 2*2000000)
        error_msg = comm_world.irecv(error_buf,tag=999)

        def check_errors():
            if error_msg.Test():
                print('Error received from node')
                print(pickle.loads(error_buf))
                raise Exception("NodeError")

        while(True):

            check_errors()
            
            if optimizer.finished.is_set():
                print("Completed")
                return
            
            completed = []

            for t in waiting_tasks.values():
                r,task,obj_data,_ = t[0],t[1],t[2],t[3]

                if r.Test():
                    task_id = task['task_id']
                    source_id = task['source_id']
                    objf_dict = build_objf_dict(obj_data,objective_spec.catchment_objf.output_schema)
                    gobj_val = global_objective.evaluate(objf_dict)
                    logger.msg_in.put(Message('log_results',local_scores=objf_dict,global_scores=gobj_val,parameters=task['params'],task_id=task['task_id'],meta=task['meta']))
                    optimizer.obj_in.put(dict(task_id=task_id,source_id=source_id,objf_val=gobj_val[gkey]))
                    completed.append(task_id)
            
            for tid in completed:
                waiting_tasks.pop(tid)        

            task = Message('task',njobs=0,jobs=[])

            while not optimizer.task_out.empty():
                task['jobs'].append(optimizer.task_out.get())
                task['njobs'] += 1

            if task['njobs'] > 0:
                task_str = pickle.dumps(task)
                mpi_bcast_send(comm_control_workers,task_str,wt=0.001,ifunc=check_errors)
            else:
                sleep(0.001)

            for t in task['jobs']:
                recv_buf = np.zeros((len(extent_map),local_objf_size))
                no_data = bytearray(0)
                r = comm_control_owning.Igatherv(no_data,[recv_buf,objf_split_counts,objf_split_offsets,MPI.DOUBLE],root=0)           
                waiting_tasks[t['task_id']] = r,t,recv_buf,no_data


    except Exception as e:
        print("Exception in server")
        print(e)
        print(get_traceback())
    except KeyboardInterrupt:
        print("Kb")
        pass
    finally:
        print("Server exiting, attempting cleanup")
        tmsg = Message('terminate')
        optimizer.control_in.put(Message('terminate'))
        logger.msg_in.put(Message('terminate'))
        logger.join()
        print("Logger closed")
        #logger.terminate()
        for n in range(n_workers):
            comm_world.send(tmsg,dest=n+1,tag=777)
        optimizer.join()
        print("Optimizer closed")

        postrun_action = cal_spec.get('postrun_action')
        if postrun_action is not None:
            postrun_action = funcspec_to_callable(postrun_action)
            postrun_action(cal_spec,locals())
        
        sys.stdout.flush()
        print("Server final exit")

def build_objf_dict(obj_data,output_schema):
    return dict([(k,obj_data[:,i]) for i,k in enumerate(output_schema)])

if __name__ == '__main__':

    import sys
    import pickle

    pklfile = sys.argv[1]

    cal_spec = pickle.load(open(pklfile,'rb'))
    run_server(cal_spec)

