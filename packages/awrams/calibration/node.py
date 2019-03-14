# Worker with partial catchments, local objf

import multiprocessing as mp
try:
    mp.set_start_method('forkserver')
    p = mp.Process()
    p.start()
    p.join()
    #pass
except:
    pass

import numpy as np
from awrams.utils.messaging import Message
from awrams.utils.messaging.buffers import *
from awrams.utils.messaging.general import get_traceback
from awrams.calibration.allocation import allocate_cells_to_workers
from awrams.calibration.worker import WorkerProcess
from awrams.calibration.objective import ObjectiveProcess
from awrams.calibration.support import ChildException
from awrams.utils.nodegraph.nodes import funcspec_to_callable
from awrams.utils.nodegraph.graph import get_dataspecs_from_mapping
import time
import os
import sys

from .mpi_support import *

import signal

def sigterm_handler(signal,frame):
    raise KeyboardInterrupt

#signal.signal(signal.SIGUSR1, sigterm_handler)

class CatchmentSpec:
    def __init__(self,catchment_id,owns_results,partial_remote,communicator=None,np_buf=None,\
        shm_buf=None,split_counts=None,split_offsets=None):
        self.catchment_id = catchment_id
        self.owns_results = owns_results
        self.partial_remote = partial_remote
        self.communicator = communicator
        self.np_buf = np_buf
        self.shm_buf = shm_buf
        self.split_counts = split_counts
        self.split_offsets = split_offsets

    def __repr__(self):
        return "%s, %s" % (self.catchment_id, self.owns_results)


class CalibrationNode:

    def __init__(self):
        self.pid = os.getpid()
        self.error_state = None

    def run_node(self):
        '''
        Set up objectives, start objf process
        '''

        '''
        Set up MPI and receive init; receive allocation info, set up communicators etc
        Catchment list will be built from allocation info, then have shared buffers attached where need-be
        '''
        #sub_node_watcher = Thread(self.gather_errors)
        from socket import gethostname
        print("Node starting on %s" % gethostname())    


        try:
            from mpi4py import MPI

            comm_world = MPI.COMM_WORLD

            init_msg = comm_world.recv(source=0)

            node_alloc = init_msg['node_alloc']
            catch_node_map = init_msg['catch_node_map']
            extent_map_full = init_msg['extent_map_full']
            run_period = init_msg['run_period']
            n_sub_workers = init_msg['n_sub_workers']
            owning_incl = init_msg['owning_incl']
            self.objective_spec = init_msg['objective_spec']
            objf_split_counts = init_msg['objf_split_counts']
            objf_split_offsets = init_msg['objf_split_offsets']
            node_mapping = init_msg['node_mapping']
            model = init_msg['model']
            #model_options = init_msg['model_options']

            self.model_outputs = self.objective_spec.catchment_objf.inputs_from_model
            self.local_objf_outputs = self.objective_spec.catchment_objf.output_schema

            '''
            non_worker_exclusions; list of global ranks which are not worker nodes or root
            extent_map_full; map catchment_ids to areas/masks (need to supply to workers for run_graph), plus aggregation
            run_period; period for which all cells are executed
            node_alloc; which catchments/cells belong to this node (incl ownership etc)
            catch_node_map; which nodes are attached to which catchments (determine communicator groups)
            n_sub_workers; number of sub_workers that this node should spawn
            schema
            '''

            '''
            Build objective spec
            '''

            g_world = comm_world.Get_group() 
            g_control_workers = g_world.Excl(init_msg['non_worker_exclusions'])
            comm_control_workers = comm_world.Create(g_control_workers)

            g_workers = g_control_workers.Excl([0])

            node_rank = g_workers.rank

            from awrams.calibration.allocation import build_node_splits

            ''' 
            Build catchmentspec dict;  data_index_map, communicators
            '''
            self.catchments = []

            data_index_map = build_data_index_map(node_alloc[node_rank]['catchments'])

            catchment_bufs = {}

            for cid,cell_info,ownership in node_alloc[node_rank]['catchments']:
                cspec = CatchmentSpec(cid,ownership['owns'],ownership['remote'])
                if cspec.partial_remote:
                    g_catch = g_workers.Incl(catch_node_map[cid])
                    cspec.communicator = comm_world.Create(g_catch)
                    cspec.split_counts,cspec.split_offsets = build_node_splits(node_alloc,catch_node_map,cid,len(run_period))
                    if cspec.owns_results:
                        data_shape = [extent_map_full[cid].cell_count,len(run_period)]
                        cspec.shm_buf = create_shm_dict(self.model_outputs,data_shape)
                        cspec.np_buf = shm_to_nd_dict(**cspec.shm_buf)
                        # Update buffer manager with these
                        catchment_bufs[cid] = cspec.shm_buf
                        #data_index_map_objf[cid] = None # We own the whole buffer, therefore there is no indexing

                self.catchments.append(cspec)

            self.owned_catchments = [c.catchment_id for c in self.catchments if c.owns_results]
            self.n_owned_catch = len(self.owned_catchments)
            self.catch_obj_index = dict([(c,i) for i,c in enumerate(self.owned_catchments)])

            if self.n_owned_catch > 0:
                g_control_owning = g_world.Incl(owning_incl)
                comm_control_owning = comm_world.Create(g_control_owning)


            worker_alloc = allocate_cells_to_workers(node_alloc[node_rank],extent_map_full,n_sub_workers)

            n_sub_workers = len(worker_alloc)

            '''
            Build shared memory buffers for workers/objective
            '''
            total_cells = sum(node_alloc[node_rank]['cell_counts'])
            data_shape = [len(run_period),total_cells]

            NBUFFERS = 2 # +++ Hardcode for now, just to get it working

            try:
                self.mcm = create_multiclient_manager(self.model_outputs,data_shape,NBUFFERS,n_sub_workers,True,catchment_bufs)
            except Exception as e:
                print('Error creating shared memory buffers')
                raise e

            '''
            Set up objective process
            '''
            obj_buf_handler = self.mcm.get_handler()

            self.objective = ObjectiveProcess(obj_buf_handler,data_index_map,extent_map_full,run_period,self.objective_spec,self.owned_catchments)
            self.objective.daemon = True

            self.objective.start()

            '''
            Set up model before sending to shared workers
            '''
            #model.set_outputs(schema['model_outputs'])
            #model.set_model_options()
            #model.init_shared(get_dataspecs_from_mapping(node_mapping,True))


            '''
            Set up worker nodes; Shared buffers created from allocation info and general node settings
            '''
            
            client_bufs = self.mcm.get_client_managers()

            self.workers = [WorkerProcess(client_bufs[i],worker_alloc[i],extent_map_full,node_mapping,model,self.model_outputs,run_period) for i in range(n_sub_workers)]

            for w in self.workers:
                w.daemon = True
                w.start()

            signal.signal(signal.SIGTERM, sigterm_handler)

            taskbuf = bytearray(1 * 2**20)

            error_buf = bytearray(2 * 2**20)
            error_msg = comm_world.irecv(error_buf,tag=777)

            def check_server_message():
                if error_msg.Test():
                    print('Termination request received from server')
                    raise KeyboardInterrupt

            #print ("Entering main nodeloop")

            start = None
            while(True):

                task = mpi_bcast_recv(comm_control_workers,root=0,ifunc=check_server_message)

                if start is None:
                    start = time.time()
                
                if task.subject == 'terminate':
                    return

                self.submit_task(task)

                for i in range(task['njobs']):
                    task_buffer_id,params = self.get_results() # poll worker queues for job completion

                    # All below should be split into a post-processing stage (ie separate function)
                    # Right now it is hardcoded for lumped multi-catchment calibration
                    queued_objf = []
                    queued_send = []

                    for c in self.catchments: # +++ Sort by <owns_no_partial,owns_partial,other_owns>
                        if c.owns_results: # responsible for sending back to global
                            if c.partial_remote: # needs data from others

                                comm_c = c.communicator

                                r = []
                                out_tmp = []
                                data_local = self.mcm.map_buffer(task_buffer_id,data_index_map[c.catchment_id])

                                for s in self.model_outputs:

                                    data_local_s = np.ascontiguousarray(data_local[s].T)

                                    recv_buf = c.np_buf[s]

                                    r.append(comm_c.Igatherv(data_local_s,[recv_buf,c.split_counts,c.split_offsets,MPI.DOUBLE],root=0)) # needs to collect to shared buffer; but no buffer rotation required (1 buffer per catchment per variable)
                                    out_tmp.append((data_local_s,recv_buf))
                                    

                                queued_objf.append([r,c.catchment_id,c.catchment_id,params,out_tmp])

                                
                            else:
                                self.submit_objf(c.catchment_id,task_buffer_id,params,data_index_map[c.catchment_id],False)
                        else: # needs to send data to others
                            #+++ Should aggregate before sending for purely lumped catchments
                            # Do aggregation here or in objf?
                            comm_c = c.communicator
                            
                            r = []
                            out_tmp = []
                            

                            data = self.mcm.map_buffer(task_buffer_id,data_index_map[c.catchment_id])

                            for s in self.model_outputs:
                                out_data = np.ascontiguousarray(data[s].T)
                                r.append(comm_c.Igatherv(out_data,[None,c.split_counts,c.split_offsets,MPI.DOUBLE],root=0))
                                out_tmp.append(out_data)

                            queued_send.append([r,out_tmp])

                    for r,catchment_id,buffer_id,params,out_data in queued_objf:
                        for _r in r:
                            mpi_wait(_r)
                            #_r.wait()

                        self.submit_objf(catchment_id,buffer_id,params,None,True)


                    for r,out_data in queued_send:
                        for _r in r:
                            mpi_wait(_r)
                            #_r.wait()

                    objf_vals = self.get_objective()

                    self.mcm.reclaim(task_buffer_id)

                    if self.n_owned_catch > 0:
                        r = comm_control_owning.Igatherv(objf_vals,[None,objf_split_counts,objf_split_offsets,MPI.DOUBLE],root=0)
                        #comm_control_owning.Gatherv(objf_vals,[None,objf_split_counts,objf_split_offsets,MPI.DOUBLE],root=0)
                        mpi_wait(r)
                        #r.wait()

        except KeyboardInterrupt:
            # User requested interrupt; clean up
            pass
        except ChildException as e:
            # Exception already reported; move to cleanup
            self.error_state = ChildException
        except Exception as e:
            # Unhandled exception, report and clean up
            print("Exception in node %s" % self.pid)
            print(e)
            print(get_traceback())
            sys.stdout.flush()
            self.error_state = ChildException
        finally:
            print("Node %s terminating" % self.pid)
            if self.error_state is not None:
                comm_world.isend(self.error_state,0,tag=999)
            self.terminate()
            print("Node %s complete" % self.pid)
            end=time.time()
            print(end-start)
            #+++ A bit of a nasty hack to get exits happening on Windows
            if os.name == 'nt':
                print("Windows environment detected; ignore the following error messages")
                print("(This is simply part of a cleanup routine)")
                comm_world.Abort()

    def terminate(self):
        #[os.kill(w.pid,signal.SIGINT) for w in self.workers]
        #os.kill(self.objective.pid,signal.SIGINT)
        try:
            self.objective.terminate()
            self.objective.join()
        except:
            pass
        try:
            [w.terminate() for w in self.workers]
            [w.join() for w in self.workers]  
        except:
            pass

    def _terminate(self,timeout=1):
        #self.submit_task(Message('terminate'))
        #self.objective.node_to_objective_q.put(Message('terminate'))

        alive = True

        t_s = time.time()

        while time.time() - t_s < timeout:
            alive = any(w.is_alive() for w in self.workers)
            if self.objective.is_alive():
                alive = True
            if not alive:
                break
        else:
            self.objective.terminate()
            [w.terminate() for w in self.workers]


        print('o')

    def submit_task(self,task):
        for w in self.workers:
            w.node_to_worker_q.put(task)

    def get_results(self):
        for w in self.workers:
            msg = w.worker_to_node_q.get()
            if msg.subject == 'exception':
                print("Exception in worker %s" % msg['pid'])
                print(msg['exception'])
                print(msg['traceback'])
                raise ChildException
            else:
                buf_id = msg['buf_id']
                params = msg['params']
        
        return buf_id, params

    def get_catchment_result(self):
        msg = self.objective.objective_to_node_q.get()
        if msg.subject == 'exception':
            print("Exception in objective function")
            print(msg['exception'])
            print(msg['traceback'])
            raise ChildException
        else:
            return msg['catchment'],msg['result']

    def submit_objf(self,catchment_id,buffer_id,params,data_index=None,translate=False):
        self.objective.node_to_objective_q.put(Message('evaluate',(catchment_id,buffer_id,params,data_index,translate)))

    def get_objective(self):
        out_data = np.zeros((self.n_owned_catch,len(self.local_objf_outputs)))
        
        for i in range(self.n_owned_catch):
            catch,result = self.get_catchment_result()
            out_data[self.catch_obj_index[catch]] = result

        return out_data

def build_data_index_map(catchments):
    data_index_map = {}
    cur_cell = 0
    for c in catchments:
        next_cell = cur_cell + c[1]['ncells']
        data_index_map[c[0]] = np.s_[:,cur_cell:next_cell]
        cur_cell = next_cell
    return data_index_map

if __name__ == '__main__':

    node = CalibrationNode()
    node.run_node()
