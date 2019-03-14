from enum import Enum
from pandas import DatetimeIndex
from awrams.utils import extents
from awrams.utils import mapping_types as mt
from numpy import prod

class TopologyType(Enum):
    ONE_TO_ONE = 1
    ONE_TO_MANY = 2
    MANY_TO_ONE = 3
    MANY_TO_MANY = 4

class TaskMap:
    def __init__(self,mapping):
        self.mapping = mapping

    def __iter__(self):
        return self.mapping.__iter__()

class DataContract:
    def __init__(self,src_group,dest_group,dtype,src_var_name,dest_var_name=None):
        self.src_group = src_group
        self.dest_group = dest_group
        self.dtype = dtype
        self.src_var_name = src_var_name
        if dest_var_name is None:
            dest_var_name = src_var_name
        self.dest_var_name = dest_var_name
        self.topology = self._determine_topology()
        #self.flatten = (not src_group.domain[0].flat) and dest_group.domain[0].flat 

    def _determine_topology(self):
        if self.src_group.size == 1:
            if self.dest_group.size == 1:
                return TopologyType.ONE_TO_ONE
            else:
                return TopologyType.ONE_TO_MANY
        else:
            if self.dest_group.size == 1:
                return TopologyType.MANY_TO_ONE
            else:
                raise Exception("MANY_TO_MANY not yet supported")

class NodeType:
    def __init__(self,node_class,node_args=None):
        if node_args is None:
            node_args = {}
        self.node_class = node_class
        self.node_args = node_args

class NodeGroup:
    def __init__(self,node_type,domain,size=1):
        self.node_type = node_type
        self.domain = domain
        self.size = size
        self.members = [] # This is filled in later by build_node_map

class NodeSpec:
    def __init__(self,group,group_rank):
        self.group = group
        self.node_type = group.node_type
        self.domain = group.domain
        self.group_rank = group_rank

    def instantiate(self,mpi_env):
        return self.node_type.node_class(mpi_env,self.group_rank,self.domain,**self.node_type.node_args)        

class ClusterSpec:
    def __init__(self,node_groups,data_contracts):
        self.node_groups = node_groups
        self.data_contracts = data_contracts
        self.build_node_map()

    def build_node_map(self):
        #from wip_support import StubNode
        #server_group = NodeGroup('server',NodeType('server',StubNode,dict(name='server')),[None])
        node_map = {}
        #node_map[0] = NodeSpec(server_group,0)
        for name,group in self.node_groups.items():
            cur_node_id = len(node_map)
            group.members = list(range(cur_node_id,cur_node_id+group.size))
            for i in range(group.size):
                node_map[len(node_map)] = NodeSpec(group,i)
        self.node_map = node_map


