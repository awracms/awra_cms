from awrams.utils import datetools as dt
from awrams.utils import extents
from collections import OrderedDict
from awrams.cluster.cluster import *
from awrams.cluster.domain import *
import numpy as np
from awrams.cluster.mpi_nodes import GraphNode, ModelRunnerNode, OutputGraphNode
from awrams.utils.nodegraph import nodes, graph

import os
import sys

def launch_sim_from_pickle(pklfile):
    '''
    Launch an MPI calibration job from the specified picklefile (usually built from cluster.build_pickle_from_spec)
    '''
    import pickle
    cspec = pickle.load(open(pklfile,'rb'))
    nnodes = len(cspec.node_map)

    call_str = 'mpiexec --oversubscribe -n {nnodes} python3 mpi_node_entry.py {pklfile}'.format(**locals())


    from subprocess import Popen,PIPE,STDOUT,signal
    import sys
    proc = Popen(call_str.split(),stdout=PIPE,stderr=STDOUT,preexec_fn=os.setsid)

    cur_out = ' '

    try:

        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("Caught interrupt")
        proc.send_signal(signal.SIGINT)
        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()

    return_code = proc.wait()    


NWORKERS= 8

OUTPUT_RES = np.float64

full_extent = extents.get_default_extent()

period = dt.dates('dec 2010 - jan 2013')
extent = full_extent

full_coords = OrderedDict()
full_coords['time'] = DomainMap(DomainMappedDatetimeIndex(period))
full_coords['latlon'] = DomainMap(DomainMappedExtent(extent))
full_dom = Domain(full_coords)

task_dom = full_dom.subdivide('time','task',split_period_annual_chunked,64)
subtask_dom = task_dom.subdivide('latlon','subtask',subdivide_extent_chunks,(64,64))
node_dom = subtask_dom.subdivide('latlon','node',subdivide_extent_equal_cells_flatindex,NWORKERS)

#from awrams.models.awral import model
from awrams.utils import config_manager

#m = model.AWRALModel()

model_profile = config_manager.get_model_profile('awral','v6_default')

m = model_profile.get_model()

imap = model_profile.get_input_mapping()

# Construct a dataspec for our static forcing nodes
dims = ['time','latitude','longitude']
dspec = graph.DataSpec('array',dims,np.float32)

f_out = {}
new_map = imap.copy()
for k,v in imap.items():
    if v.out_type == 'forcing':
        if 'io' in v.properties:
            f_out[k] = v
            new_map[k] = nodes.static(None,dspec,v.out_type,True)
        new_map[k].properties['dynamic'] = True

worker_node = NodeType(ModelRunnerNode,dict(mapping=new_map,model=m,statify_graph=True,separate_subtask_graph=True))
forcing_maps = OrderedDict([(k,dict([[k,v]])) for k,v in f_out.items()])

def build_output_maps(m,save_vars,path):
    omap = m.get_output_mapping()
    all_maps = OrderedDict()
    for k in save_vars:
        all_maps[k] = {}
        all_maps[k][k] = omap[k]
        all_maps[k][k+'_ncwrite'] = nodes.write_to_annual_ncfile(path,k,dtype=OUTPUT_RES,ncparams=dict(zlib=True,complevel=1))
    return all_maps

output_maps = build_output_maps(m,['qtot','s0'],'./test_sim_outputs')

node_groups = OrderedDict()

for k,v in forcing_maps.items():
    node_group = NodeGroup(NodeType(GraphNode,dict(mapping=v,statify_graph=False,separate_subtask_graph=False)),subtask_dom)
    node_groups[k] = node_group

for k,v in output_maps.items():
    node_group = NodeGroup(NodeType(OutputGraphNode,dict(mapping=v,period=period,extent=extent)),subtask_dom)
    node_groups[k] = node_group

node_groups['worker_nodes'] = worker_group = NodeGroup(worker_node,node_dom,NWORKERS)

data_contracts = OrderedDict()
for k,v in forcing_maps.items():
    data_contracts[k+'_to_workers'] = DataContract(node_groups[k],worker_group,np.float32,k)

for k,v in output_maps.items():
    data_contracts['workers_to_'+k] = DataContract(worker_group,node_groups[k],OUTPUT_RES,k)

cspec = ClusterSpec(node_groups,data_contracts)

import pickle

pickle.dump(cspec,open('simspec.pkl','wb'))

print(len(cspec.node_map))

launch_sim_from_pickle('simspec.pkl')





