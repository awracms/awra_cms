from collections import OrderedDict
from numbers import Number
from .nodes import DataSpec, InputNode,ProcessNode,get_expanded,get_flattened,static
import time
from awrams.utils.mapping_types import gen_coordset
import pandas as pd

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('graph')

def find_heads(nodes):
    '''
    Separate nodes into no-upstream-dependency (heads) and others (tails)
    '''
    #heads = []
    heads = OrderedDict()
    tails = {}
    for k,n in nodes.items():
        if n is None:
            raise Exception("Node value unspecified", k)
        if len(n.inputs) == 0:
            heads[k] = n
            #heads.append(k)
        else:
            tails[k] = n
    return heads, tails
    
def find_endpoint_keys(nodes):
    '''
    Return a list of keys of all nodes who have no nodes downstream
    '''

    has_downstream = set()

    for k,v in nodes.items():
        for d in v.inputs:
            has_downstream.add(d)

    return has_downstream.symmetric_difference((nodes.keys()))

def _get_input_tree(nodestr,nodes,tree=None):
    '''
    Return all nodes that are upstream of the specified outputs
    UNORDERED
    '''
    if tree is None:
        tree = {}
    node = nodes[nodestr]
    if nodestr not in tree:
        tree[nodestr] = node
        for i in node.inputs:
            if not isinstance(i, Number):
                _get_input_tree(i,nodes,tree)
    return tree

def _get_output_tree(nodestr,nodes,tree=None):
    '''
    Return all nodes that are downstream of the specified inputs
    UNORDERED
    '''
    if tree is None:
        tree = {}
    node = nodes[nodestr]
    if nodestr not in tree:
        tree[nodestr] = node
        for k,n in nodes.items():
            if nodestr in n.inputs:
                _get_output_tree(k,nodes,tree)
    return tree
    
def get_input_tree(outputs,nodes):
    '''
    Return all nodes that are upstream of the specified outputs
    UNORDERED
    '''
    tree = {}
    if isinstance(outputs,str):
        outputs = [outputs]
    for o in outputs:
        tree = _get_input_tree(o,nodes,tree)
    return tree

def get_output_tree(inputs,nodes):
    '''
    Return all nodes that are downstream of the specified inputs
    UNORDERED
    '''
    tree = {}
    if isinstance(inputs,str):
        outputs = [inputs]
    for i in inputs:
        tree = _get_output_tree(i,nodes,tree)
    return tree

def build_fixed_mapping(mapping,model_keys,dyn_keys,period,extent):
    """
    Args:
        mapping (dict): imap as produced by model.get_default_mapping()
        model_keys (list): keys that must be available as outputs of this graph
        dyn_keys (list): keys for dynamic nodes (ie runtime computed)
        period (DatetimeIndex): period over which to generate the mapping
        extent (extents.Extent): extent (or list of extents) over which to generate the mapping
    
    Returns:
        dict: mapping (containing static data as well as node mappings)
    
    """
    mapping = expand_const(mapping)
    
    dyn_head_mapping = dict([(k,v) for k,v in mapping.items() if k in dyn_keys])
    dyn_head_mapping = list(dyn_head_mapping)
    dynamic_mapping = get_output_tree(dyn_head_mapping,mapping)
    dynamic_keys = set(dynamic_mapping)
    
    static_keys = dynamic_keys.symmetric_difference(set(mapping.keys()))
    static_mapping = dict([k,mapping[k]] for k in static_keys)
    
    def coalesce_keys(key_list):
        out_list = []
        for keys in key_list:
            out_list += keys
        return set(out_list)

    kl = [v.inputs for k,v in dynamic_mapping.items()]
    extra_static_keys = coalesce_keys(kl).difference(dynamic_keys) # set of (static) keys required by the dynamic part of the graph
    
    req_static_outputs = (set(model_keys).intersection(static_keys)).union(extra_static_keys)
    
    #mapping = mapping.copy()
    
    #parameterised, fixed_mapping,fixed_endpoints = graph.split_parameterised_mapping(mapping,model_keys)
    static_egraph = ExecutionGraph(static_mapping)
    dspecs = static_egraph.get_dataspecs()
    
    if not isinstance(extent,list):
        extent = [extent]

    coords = [gen_coordset(period,e) for e in extent]
    masks = [e.mask for e in extent]
    
    res = static_egraph.get_data_flat(coords,masks,multi=True)
    static_data = dict((k,res[k]) for k in req_static_outputs)

    del(res)

    #static_mapping = graph.expand_const(fixed_mapping)

    out_mapping = dict([k,static(static_data[k],dspecs[k],static_mapping[k].out_type)] for k in req_static_outputs)
    out_mapping.update(dynamic_mapping)

    return out_mapping

def split_parameterised_mapping(mapping,output_keys,include_orphans=True):
    ### ++++ Refactor to externalise parameters
    # DEPRECATED - refer to build_fixed_mapping
    '''
    mapping: standard nodegraph mapping
    output_keys: list of keys that must appear as endpoints
    Returns a tuple containing:
    parameterised : graph of all nodes and their downstream dependants who have unfixed parameters
    fixed : graph of all nodes without unfixed parameters upstream
    fixed_endpoints : list of keys who are either endpoints in <fixed>, or are required by <parameterised>

    if include_orphans is set, the parameterised graph will include fixed params (but not their children)
    '''
    #Find all portions of the graph that are runtime parameterised

    param_heads = []

    for k,v in mapping.items():
        if v.node_type == 'parameter':
            if v.args['fixed'] == False:
                param_heads.append(k)

    parameterised = get_output_tree(param_heads,mapping)

    #Build the inverse graph (ie the remainder of the graph that is fixed during a run)

    fixed_keys= set(parameterised.keys()).symmetric_difference(set(mapping.keys()))
    fixed_mapping = dict([k,mapping[k]] for k in fixed_keys)
    
    fixed_endpoints = find_endpoint_keys(fixed_mapping)

    #Update endpoints to ensure all model inputs are registered
    for k in output_keys:
        if k in fixed_mapping and k not in fixed_endpoints:
            fixed_endpoints.add(k)

    
    #Add nodes that are required by the parameterised graph

    for kp,vp in parameterised.items():
        for k in fixed_mapping.keys():
            if k in vp.inputs:
                fixed_endpoints.add(k)

    if include_orphans:
        for k,v in mapping.items():
            if v.node_type == 'parameter':
                if v.args['fixed'] == True:
                    parameterised[k] = v
    
    return parameterised, fixed_mapping, fixed_endpoints

def build_graph(heads,tails=None):
    '''
    Return a dependency-resolved ordered set of keys
    '''
    if tails is None:
        heads,tails = find_heads(heads)
    
    def all_met(heads,n):
        for i in n.inputs:
            if not isinstance(i,Number):
                if i not in heads:
                    return False
        return True
    
    unmet = {}
    
    found = False
    for k,n in tails.items():
        if all_met(heads,n):
            heads[k] = n
            found = True
        else:
            unmet[k] = n
            
    if len(tails) == 0:
        found = True
            
    if not found:
        raise Exception("Unsolvable graph",unmet)
    
    if len(unmet) > 0:
        return build_graph(heads,unmet)
    else:
        return heads

class ExecutionGraph:
    def __init__(self,mapping):
        exe_list = build_graph(mapping)

        self.input_graph = OrderedDict()
        self.process_graph = OrderedDict()

        self.mapping = mapping

        self.const_inputs = {}

        try:
            for nkey in exe_list:
                inputs = mapping[nkey].inputs
                for i in inputs:
                    if isinstance(i,Number):
                        self.const_inputs[i] = i
                if len(inputs): # Processor
                    self.process_graph[nkey] = dict(exe=mapping[nkey].get_executor(),inputs=mapping[nkey].inputs)
                else: # Input endpoint
                    self.input_graph[nkey] = dict(exe=mapping[nkey].get_executor())
        except:
            logger.critical("Failed instantiation of node %s" % nkey)
            raise

    def set_parameters(self,parameters):
        #+++
        # Refactor into utility function rather than class method
        # Otherwise the classes will get populated with hundreds of single use tools...
        
        # Assume dataframe is our new calibration dataframe type
        if isinstance(parameters,pd.DataFrame):
            parameters = parameters['value']
        for k,v in parameters.items():
            # We may be supplied more parameters than this model takes, just set what is in there
            if k in self.input_graph:
                self.input_graph[k]['exe'].value = v

    def get_dataspecs(self,flat=False):
        dspecs = {}

        for k,v in self.const_inputs.items():
            dspecs[k] = DataSpec('scalar',[],type(v))

        for k,v in self.input_graph.items():
            if flat:
                dspecs[k] = v['exe'].get_dataspec_flat()
            else:
                dspecs[k] = v['exe'].get_dataspec()

        for k,v in self.process_graph.items():
            dspecs[k] = v['exe'].get_dataspec([dspecs[i] for i in v['inputs']])
        return dspecs

    def get_data(self,coords):

        node_values = self.const_inputs.copy()

        for k,v in self.input_graph.items():
            try:
                node_values[k] = v['exe'].get_data(coords)
            except Exception as e:
                [hru]("Failed generating %s with exception %s" % (k,repr(e)))
                raise

        for k,v in self.process_graph.items():
            try:
                node_values[k] = v['exe'].process([node_values[i] for i in v['inputs']])
            except Exception as e:
                logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
                raise
                
        return node_values

    def get_data_prepack(self,cid):
        #+++
        # No longer used, consider deprecated; may need to reinstate some variation of
        # this for 'partially' prepacked data (eg from build_static_mapping)
        
        node_values = self.const_inputs.copy()

        for k,v in self.input_graph.items():
            try:
                node_values[k] = v['exe'].value
            except AttributeError:
                node_values[k] = v['exe'].get_data_prepack(cid)
            except Exception as e:
                # print(k,coords)
                logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
                raise
                

        for k,v in self.process_graph.items():
            try:
                node_values[k] = v['exe'].process([node_values[i] for i in v['inputs']])
            except Exception as e:
                logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
                raise
                
        return node_values

    def get_data_flat(self,coords,mask=None,multi=False):
        if mask is None:
            import numpy as np

            if multi:
                mask = []
                for i in range(len(coords)):
                    mask[i] = np.zeros((coords[i].shape[1],coords[i].shape[2]),dtype=bool)
            else:
                mask = np.zeros((coords.shape[1],coords.shape[2]),dtype=bool)

        node_values = self.const_inputs.copy()

        try:
            if multi:
                for k,v in self.input_graph.items():
                    node_values[k] = v['exe'].get_data_multiflat(coords,mask)
            else:
                for k,v in self.input_graph.items():
                    node_values[k] = v['exe'].get_data_flat(coords,mask)
        
        except Exception as e:
            logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
            raise
                
        for k,v in self.process_graph.items():
            try:
                node_values[k] = v['exe'].process([node_values[i] for i in v['inputs']])
            except Exception as e:
                logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
                raise

        return node_values


def build_output_graph(heads,tails=None):
    '''
    Return a dependency-resolved ordered set of keys
    '''
    if tails is None:
        heads,tails = find_heads(heads)

    def all_met(heads,n):
        for i in n.inputs:
            if not isinstance(i,Number):
                if i not in heads:
                    return False
        return True

    unmet = {}

    found = False
    for k,n in tails.items():
        if all_met(heads,n):
            heads[k] = n
            found = True
        else:
            unmet[k] = n

    if len(tails) == 0:
        found = True

    if not found:
        print("UNMET:\n",unmet,'\n')
        raise Exception("Unsolvable graph")

    if len(unmet) > 0:
        return build_graph(heads,unmet)
    else:
        return heads

class OutputGraph:
    def __init__(self,mapping):
        '''
        +++

        Should refactor into ExecutionGraph rather than maintaining 2 cut'n'paste jobs...

        :param mapping: outputs mapping
        '''
        exe_list = build_output_graph(mapping)

        self.input_graph = OrderedDict()   ### model outputs
        self.save_graph = OrderedDict()    ### persistent outputs
        self.writer_graph = OrderedDict()  ### ncfile writers

        self.mapping = mapping

        # self.const_inputs = {}
        for nkey in exe_list:
            inputs = mapping[nkey].inputs
            if len(inputs):
                if mapping[nkey].out_type == 'ncfile':
                    self.writer_graph[nkey] = dict(exe=mapping[nkey].get_executor(),inputs=mapping[nkey].inputs)
                elif mapping[nkey].out_type == 'save':
                    self.save_graph[nkey] = dict(exe=mapping[nkey].get_executor(),inputs=mapping[nkey].inputs)
            else:
                if mapping[nkey].properties['io'] == 'from_model':
                    self.input_graph[nkey] = dict(exe=mapping[nkey].get_executor())


    def get_dataspecs(self,flat=False):
        dspecs = {}

        for k,v in self.input_graph.items():
            if flat:
                dspecs[k] = v['exe'].get_dataspec_flat()
            else:
                dspecs[k] = v['exe'].get_dataspec()

        for k,v in self.writer_graph.items():
            dspecs[k] = v['exe'].get_dataspec()

        return dspecs

    def set_data(self,coords,data_map,mask):
        ### constants for processing
        # node_values = self.const_inputs.copy()
        node_values = {}

        ### outputs from model
        for k,v in self.input_graph.items():
            try:
                v['exe'].set_data(data_map[k])
                node_values[k] = v['exe'].data
            except Exception as e:
                logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
                raise

        ### nodes for data persistence (for ondemand not mp server)
        for k,v in self.save_graph.items():
            try:
                node_values[k] = v['exe'].set_data([node_values[i] for i in v['inputs']])
            except Exception as e:
                logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
                raise

        ### nodes for writing data to file
        for k,v in self.writer_graph.items():
            try:
                v['exe'].process(coords,get_expanded(node_values[v['inputs'][0]],mask))
            except Exception as e:
                logger.critical("Failed generating %s with exception %s" % (k,repr(e)))
                raise

        return node_values

    def initialise(self,period,extent):
        for k,v in self.writer_graph.items():
            v['exe'].init_files(period,extent)

    def sync_all(self):
        for k,v in self.writer_graph.items():
            v['exe'].sync_all()

    def close_all(self):
        for k,v in self.writer_graph.items():
            v['exe'].close()

def map_rescaling_nodes(mapping,target_extent):
    #+++
    # Could possibly return a fully constructed ExecutionGraph?
    from awrams.utils.processing.rescaling import upscaling_node, downscaling_node
    from copy import deepcopy

    eg = ExecutionGraph(mapping)

    out_mapping = deepcopy(mapping)

    for k,v in eg.input_graph.items():
        if 'io' in mapping[k].properties:
            try:
                e = v['exe'].get_extent()
                if e.cell_size < target_extent.cell_size:
                    out_mapping[k] = downscaling_node(out_mapping[k],target_extent)
                elif e.cell_size > target_extent.cell_size:
                    out_mapping[k] = upscaling_node(out_mapping[k],target_extent)
            except Exception as exc:
                raise(Exception("Failed rescaling", k, type(v['exe']), exc))

    return out_mapping

def get_dataspecs_from_mapping(mapping,flat=True):
    egraph = ExecutionGraph(mapping)
    return egraph.get_dataspecs(flat)

def validate_mapping(imap,params):
    '''
    Return any keys from the required model parameters <params> that are absent from the input map <imap>
    '''
    missing = []
    for k in params:
        if k not in imap:
            missing.append(k)
    return missing

from numbers import Number

def expand_const(mapping):
    '''
    Convert any arguments in nodes expressed as Python Numbers to nodes.const nodes
    +++ Should we generate meaningful names for the constant nodes (rather than just naming them as the numeric value?)
    '''
    from awrams.utils.nodegraph.nodes import const as const_node

    mapping = mapping.copy()
    const_req = set()
    for v in mapping.values():
        for i in v.inputs:
            if isinstance(i,Number):
                const_req.add(i)
    
    for i in const_req:
        if i not in mapping:
            mapping[i] = const_node(i)
    return mapping