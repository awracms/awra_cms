from awrams.utils import datetools as dt
from awrams.utils import extents
from collections import OrderedDict
from awrams.cluster.cluster import *
from awrams.cluster.domain import *
import numpy as np
from awrams.cluster.mpi_nodes import GraphNode, ModelRunnerNode, OutputGraphNode
from awrams.utils.nodegraph import nodes, graph

import os
import pickle

from subprocess import Popen,PIPE,STDOUT,signal
import sys

def launch_sim_from_pickle(pickle_file):
    '''
    Launch an MPI calibration job from the specified picklefile (usually built from cluster.build_pickle_from_spec)
    '''
    
    cspec = pickle.load(open(pickle_file,'rb'))
    nnodes = len(cspec.node_map)

    #MSMPI needs different switches
    if os.name == 'nt':
        call_str = ('mpiexec -n {nnodes} '
                'python -m awrams.cluster.mpi_node_entry {pickle_file}').format(**locals())
    else:
        call_str = ('mpiexec --oversubscribe --allow-run-as-root --mca plm_rsh_agent '
                    'false -n {nnodes} '
                    'python3 -m awrams.cluster.mpi_node_entry {pickle_file}').format(**locals())

    proc = Popen(call_str,stdout=PIPE,stderr=STDOUT,shell=True)

    cur_out = ' '

    try:

        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()

    return_code = proc.wait()
    
    return return_code

def build_sim_pickle(model,input_map,output_map,period,extent,n_workers, \
                     pickle_file,spatial_chunk,time_chunk,min_cells_per_worker):
    
    full_coords = OrderedDict()
    full_coords['time'] = DomainMap(DomainMappedDatetimeIndex(period))
    full_coords['latlon'] = DomainMap(DomainMappedExtent(extent))
    full_dom = Domain(full_coords)

    task_dom = full_dom.subdivide('time','task',split_period_annual_chunked,time_chunk)
    subtask_dom = task_dom.subdivide('latlon','subtask',subdivide_extent_chunked,spatial_chunk,n_workers*min_cells_per_worker)
    node_dom = subtask_dom.subdivide('latlon','node',subdivide_extent_equal_cells_flatindex,n_workers)


    # Construct a dataspec for our static forcing nodes
    dims = ['time','latitude','longitude']
    dspec = graph.DataSpec('array',dims,np.float32)

    f_out = {}
    new_map = input_map.copy()
    for k,v in input_map.items():
        if v.out_type == 'forcing':
            if 'io' in v.properties:
                f_out[k] = v
                new_map[k] = nodes.static(None,dspec,v.out_type,True)
            new_map[k].properties['dynamic'] = True

    worker_node = NodeType(ModelRunnerNode,dict(mapping=new_map,model=model,statify_graph=True,separate_subtask_graph=True))
    forcing_maps = OrderedDict([(k,dict([[k,v]])) for k,v in f_out.items()])



    def split_output_maps(omap):
        all_maps = OrderedDict()
        output_res = {}
        for k,v in omap.items():
            if v.node_type == 'output_variable':
                #all_maps[k] = {}
                child_map = graph.get_output_tree([k],omap)
                for child_k, child_v in child_map.items():
                    #all_maps[k][child_k] = child_v
                    if 'io' in child_v.properties and child_v.properties['io'] == 'w':
                        if k in output_res:
                            # Promote to largest size
                            # +++ Does not support mixed integer/floating point output (yet)
                            if np.dtype(child_v.args['dtype']).itemsize > np.dtype(output_res[k]).itemsize:
                                output_res[k] = child_v.args['dtype']
                        else:
                            output_res[k] = child_v.args['dtype']
                        if k not in all_maps:
                            all_maps[k] = {}
                # Second pass to only generate items that actually output to files
                for child_k, child_v in child_map.items():
                    if k in all_maps:
                        all_maps[k][child_k] = child_v  

        return all_maps, output_res
    
    output_maps, output_res = split_output_maps(output_map)

    node_groups = OrderedDict()

    for k,v in forcing_maps.items():
        node_group = NodeGroup(NodeType(GraphNode,dict(mapping=v,statify_graph=False,separate_subtask_graph=False)),subtask_dom)
        node_groups[k] = node_group

    for k,v in output_maps.items():
        node_group = NodeGroup(NodeType(OutputGraphNode,dict(mapping=v,period=period,extent=extent)),subtask_dom)
        node_groups[k] = node_group

    node_groups['worker_nodes'] = worker_group = NodeGroup(worker_node,node_dom,n_workers)

    data_contracts = OrderedDict()
    for k,v in forcing_maps.items():
        data_contracts[k+'_to_workers'] = DataContract(node_groups[k],worker_group,np.float32,k)

    for k,v in output_maps.items():
        data_contracts['workers_to_'+k] = DataContract(worker_group,node_groups[k],output_res[k],k)

    cspec = ClusterSpec(node_groups,data_contracts)

    pickle.dump(cspec,open(pickle_file,'wb'))
    
    return cspec