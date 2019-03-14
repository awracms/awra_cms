from collections import OrderedDict
from .cluster import TopologyType, TaskMap
from .domain import task_layout_from_dom, coordset_from_domain
from .waitables import *
import numpy as np
from awrams.utils.nodegraph import nodes,graph
from awrams.utils import extents
from time import sleep

class GraphNode:
    def __init__(self,mpi_env,group_rank,domain,mapping,statify_graph=True,separate_subtask_graph=False):
        self.MPI = mpi_env
        self.domain = domain
        self.mapping = mapping
        self.group_rank = group_rank

        self.in_contracts = OrderedDict()
        self.out_contracts = OrderedDict()

        self.waitables = []

        self.comms_msg = {}
        self.comms_data = {}
        self.msg_reqs = []
        self.req_map = {}
        self.src_members = {}
        self.dest_members = {}

        self.msg_handlers = {}

        self.recv_buffers = {}

        self._data_avail_map = OrderedDict()

        self.cur_task = (0,0)
        self.task_done = False

        self.local_dom = self.domain.realise(node=self.group_rank)

        self.ntasks,self.nsubtasks = task_layout_from_dom(self.domain)

        # This is a bit of a hack - it saves having separate logic for 
        # runs that don't use subtasks
        if self.nsubtasks is None:
            self.nsubtasks = 1

        self._sep_subtask_graph = separate_subtask_graph
        self._statify = statify_graph

        self.max_buffered_tasks = 3 # was 10

    def initialise(self):
        self._init_msg_reqs()
        self._init_data_avail_map(0,0) # ? necessary?
        self._set_avail(1)
        self._init_recv_data(0,0)

        self.request_task_input_to_max()

        self._init_graphs()
        #self.graph = graph.ExecutionGraph(self.mapping)

        self._downstream_ready = dict((cname, 0) for cname in self.out_contracts)
        if (len(self.out_contracts) == 0):
            self._downstream_ready['always'] = 1

        #self.mask = self.domain.coords[1].mask

    def _init_graphs(self):
        out_keys = self.get_required_mapping_outputs()
        if self._sep_subtask_graph:
            self.graphs = {}
            for i in range(self.nsubtasks):
                if self._statify:
                    cur_dom = self.local_dom.realise(subtask=i,task=0)
                    dyn_keys = [k for k,v in self.mapping.items() if 'dynamic' in v.properties]
                    cur_graph = graph.ExecutionGraph(graph.build_fixed_mapping(self.mapping,out_keys,dyn_keys,cur_dom.coords['time'].mapping.value_obj,cur_dom.coords['latlon'].mapping.value_obj))
                else:
                    cur_graph = graph.ExecutionGraph(self.mapping)
                self.graphs[i] = cur_graph
        else:
            self.graphs = {}
            if self._statify:
                cur_dom = self.local_dom.realise(subtask=0,task=0)
                dyn_keys = [k for k,v in self.mapping.items() if 'dynamic' in v.properties]
                cur_graph = graph.ExecutionGraph(graph.build_fixed_mapping(self.mapping,out_keys,dyn_keys,cur_dom.coords['time'].mapping.value_obj,cur_dom.coords['latlon'].mapping.value_obj))
            else:
                cur_graph = graph.ExecutionGraph(self.mapping)
            self.graphs[0] = cur_graph
        self.cur_graph = self.graphs[0]


    def get_required_mapping_outputs(self):
        '''
        Usually limited to the output keys contained in our data_contracts,
        but can be overloaded (eg for model runners that perform a separate run post-graph)
        '''
        return [v.src_var_name for k,v in self.out_contracts.items()]

    def _init_data_avail_map(self,task,subtask):
        self._data_avail_map[(task,subtask)]= dict([k,False] for k in self.in_contracts)

    def _set_avail(self,count):
        for i in range(count):
            self.notify_upstream_avail()

    def _init_msg_reqs(self):
        midx = 0
        for cname, contract in self.out_contracts.items():
            self.msg_reqs.append(self.comms_msg[cname].Ibarrier())
            self.msg_handlers[midx] = self.on_downstream_ready
            self.req_map[midx] = cname
            midx += 1

    def _init_recv_data(self,task,subtask):
        self.recv_buffers[(task,subtask)] = {}
        for cname, contract in self.in_contracts.items():
            self.begin_recv(cname,task,subtask)

    def run(self):
        self.active = True
        completed = False

        while(self.active):
            self.check_messages()
            self.update_waitables()

            if not completed:
                if (self.run_condition):
                    # +++
                    # This requires more mapping for cases than must be concatenated, let's just run it for now...
                    data_map = self.recv_buffers.pop(self.cur_task)
                    self.update_graph(data_map)
                    self.run_graph()

                    if len(self.out_contracts) > 0:
                        self.deliver_out_contracts()
                        self.decrement_all_downstream_ready()
                        #self._downstream_ready -= 1

                    completed = self.set_next_task()

                    if not completed:
                        self.request_task_input_to_max()

                else:
                    sleep(0.005)

            if (len(self.waitables) == 0) and (completed):
                self.cleanup()
                return

    def cleanup(self):
        pass

    def request_task_input_to_max(self):
        new_reqs = self.max_buffered_tasks - len(self.recv_buffers)
        for i in range(new_reqs):
            tnum = self.get_next_task(max(self.recv_buffers)) # Max will return the highest key value (task first, then subtask); ie the 'most recent'
            if tnum[0] < self.ntasks:
                self._init_data_avail_map(*tnum)
                self.notify_upstream_avail()
                self._init_recv_data(*tnum)

    def update_graph(self,data_map):
        #cur_graph = self.graphs[self.cur_task[1]]
        for cname, contract in self.in_contracts.items():
            src_name = contract.src_var_name
            dest_name = contract.dest_var_name
            self.cur_graph.input_graph[dest_name]['exe'].data = data_map[src_name]

    def set_next_task(self):
        task,subtask = self.get_next_task()
        self.cur_task = (task,subtask)
        if self._sep_subtask_graph:
            self.cur_graph = self.graphs[subtask]
        return task == self.ntasks

    def get_next_task(self,cur_task=None):
        if cur_task is None:
            cur_task = self.cur_task
        task,subtask = cur_task
        subtask = (subtask + 1) % self.nsubtasks
        if subtask == 0:
            task += 1
        return task,subtask
        

    def run_graph(self):
        # +++
        # Need to decide what to do re: flat/flattening/expanding
        # Can this be expressed at node/domain barriers, or is it more complex?

        self.active_dom = self.local_dom.realise(task=self.cur_task[0],subtask=self.cur_task[1])

        # +++ Should probably cache aspects of this (ie instead of regenerating an extent object 
        # for identical subtask mappings)
        cs,mask = coordset_from_domain(self.active_dom)

        # +++ hardcoding for flat data only, shouldn't be too hard to get the non-flat case working...
        self.cur_res = self.cur_graph.get_data_flat(cs,mask)


    @property
    def run_condition(self):
        return (min(self._downstream_ready.values()) > 0) and self.all_data_available and (self.cur_task is not None)

    @property
    def all_data_available(self):
        avail = True
        for k,v in self._data_avail_map[self.cur_task].items():
            avail = avail and v
        return avail

    def notify_upstream_avail(self):
        for cname,contract in self.in_contracts.items():
            msg_comm = self.comms_msg[cname]
            self.waitables.append(MPIWaitable(self.MPI,msg_comm.Ibarrier())) # need to store req?

    def on_downstream_ready(self,m_idx):
        # Handler for downstream nodes of contracts being ready
        cname = self.req_map[m_idx]
        self._downstream_ready[cname] += 1 #+++ Naive, assume all are syncing
        self.msg_reqs[m_idx] = self.comms_msg[cname].Ibarrier()

    def on_data_available(self,cname,task,subtask,in_data):
        contract = self.in_contracts[cname]
        var_name = contract.dest_var_name
        self._data_avail_map[(task,subtask)][cname] = True

        if contract.topology == TopologyType.MANY_TO_ONE:
            in_data = np.concatenate(in_data,axis=1)

        self.recv_buffers[(task,subtask)][var_name] = in_data

    def check_messages(self):
        # Check all of our available persist message receivers
        tres = self.MPI.Request.Testsome(self.msg_reqs)
        if tres is not None:
            for m_idx in tres:
                handler = self.msg_handlers[m_idx]
                handler(m_idx)

    def update_waitables(self):
        active = []
        for w in self.waitables:
            if not w.done():
                active.append(w)
        self.waitables = active

    def begin_recv(self,cname,task,subtask):
        contract = self.in_contracts[cname]
        data_comm = self.comms_data[cname]
        if contract.topology == TopologyType.ONE_TO_MANY:
            recv_dom = self.local_dom.realise(task=task,subtask=subtask)
            shape = recv_dom.shape(True) #+++ assume all data transmission is through flat buffers
            in_data = np.empty(shape=shape,dtype=contract.dtype)
            r = data_comm.Irecv(in_data,0)
            self.waitables.append(MPIWaitable(self.MPI,r,self.on_data_available,[cname,task,subtask,in_data]))
        elif contract.topology == TopologyType.MANY_TO_ONE:
            reqs = []
            req_data = []
            for i, src in enumerate(self.src_members[cname]):
                #+++ create buffer for each source node domain
                # ie realise domain for each node, alloc np.empty(node_shape)
                # then append whole event to waitaibles
                recv_dom = contract.src_group.domain.realise(task=task,subtask=subtask,node=i)
                shape = recv_dom.shape(True) #+++ assume all data transmission is through flat buffers
                in_data = np.empty(shape=shape,dtype=contract.dtype)
                r = data_comm.Irecv(in_data,src)
                reqs.append(r)
                req_data.append(in_data)
 
            #self.waitables.append()
            w = MPIMultiWaitable(self.MPI,reqs,self.on_data_available,[cname,task,subtask,req_data])
            self.waitables.append(w)

    def deliver_out_contracts(self):
        for cname,contract in self.out_contracts.items():
            data_comm = self.comms_data[cname]
            vname = contract.src_var_name
            data = self.cur_res[vname]
            self.begin_deliver(cname,data)
        self.cur_res = None

    def decrement_all_downstream_ready(self):
        for k,v in self._downstream_ready.items():
            self._downstream_ready[k] = v - 1

    def begin_deliver(self,cname,data):
        contract = self.out_contracts[cname]
        data_comm = self.comms_data[cname]
        if contract.topology == TopologyType.ONE_TO_MANY:
            reqs = []
            send_data = []
            for i,target in enumerate(self.dest_members[cname]):
                recv_dom = contract.dest_group.domain.realise(task=self.cur_task[0],subtask=self.cur_task[1],node=i)
                shape = recv_dom.shape(True) #+++ assume all data transmission is through flat buffers
                
                #+++ VERY BAD. We're assuming all indices are always from get_data_flat,
                # to a flattened reciever, _AND_ that they only use FlatIndex rather than subextents or anything else...
                l_idx = (slice(None,None),recv_dom.coords['latlon'].mapping.findex.get_relative_index(None))

                out_data = data[l_idx].astype(contract.dtype)
                if (shape != out_data.shape):
                    print("MISMATCH ", self.cur_task, contract.src_var_name, shape, out_data.shape,'\n',\
                           data.shape,l_idx)
                out_data = out_data.flatten()
                send_data.append(out_data)
                r = data_comm.Isend(out_data,target)
                reqs.append(r)
            self.waitables.append(MPIMultiWaitable(self.MPI,reqs,action_args=[send_data]))
        else: #+++ Assume for now no MANY_TO_MANY; ie we are sending all our data to one other source
            target = self.dest_members[cname][0]
            #+++ Also no flattening/expanding; just get this working for the sim/writer case
            out_data = data.astype(contract.dtype).flatten()
            r = data_comm.Isend(out_data,target)
            self.waitables.append(MPIWaitable(self.MPI,r,action_args=[[out_data]]))

class ModelRunnerNode(GraphNode):
    def __init__(self,mpi_env,group_rank,domain,mapping,model,statify_graph=True,separate_subtask_graph=True):
        super().__init__(mpi_env,group_rank,domain,mapping,statify_graph,separate_subtask_graph)
        self.model = model

    def get_required_mapping_outputs(self):
        return self.model.get_input_keys()

    def _init_graphs(self):
        super()._init_graphs()
        self.runners = {}
        if self._sep_subtask_graph:
            for i in range(self.nsubtasks):
                self.runners[i] = self.model.get_runner(self.graphs[i].get_dataspecs(True),shared=False)
        else:
            self.runners[0] = self.model.get_runner(self.graphs[0].get_dataspecs(True),shared=False)
        self.cur_runner = self.runners[0]

    def set_next_task(self):
        completed = super().set_next_task()
        if not completed:
            self.cur_runner = self.runners[self.cur_task[1]]
        return completed

    def run_graph(self):
        super().run_graph()
        timesteps,cell_count = self.active_dom.shape(True)
        model_res = self.cur_runner.run_from_mapping(self.cur_res,timesteps,cell_count)
        self.cur_res = model_res

class OutputGraphNode(GraphNode):
    def __init__(self,mpi_env,group_rank,domain,mapping,period,extent):
        super().__init__(mpi_env,group_rank,domain,mapping,False,False)
        self.period = period
        self.extent = extent

    def _init_graphs(self):
        self.graph = graph.OutputGraph(self.mapping)
        self.graph.initialise(self.period,self.extent)

    def update_graph(self,data_map):
        new_map = {}
        for cname, contract in self.in_contracts.items():
            src_name = contract.src_var_name
            dest_name = contract.dest_var_name
            new_map[dest_name] = data_map[src_name]
        self.data_map = new_map

    def run_graph(self):
        self.active_dom = self.local_dom.realise(task=self.cur_task[0],subtask=self.cur_task[1])

        # +++ Should probably cache aspects of this (ie instead of regenerating an extent object 
        # for identical subtask mappings)
        cs,mask = coordset_from_domain(self.active_dom)

        # +++ hardcoding for flat data only, shouldn't be too hard to get the non-flat case working...
        self.graph.set_data(cs,self.data_map,mask)

    def cleanup(self):
        self.graph.close_all()