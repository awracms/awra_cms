from awrams.utils.nodegraph import nodes, graph

# Used frequently during template building, dynamic compilation etc
BASE_TEMPLATE = dict(
    OUTPUTS_HRU=[],
    OUTPUTS_AVG=[],
    OUTPUTS_CELL=[],
    INPUTS_SCALAR=[],
    INPUTS_SCALAR_HRU=[],
    INPUTS_SPATIAL=[],
    INPUTS_SPATIAL_HRU=[],
    INPUTS_FORCING=[]
)

def set_fast_forcing(node_mapping):
    from awrams.utils.io.data_mapping import map_filename_annual

    forcing_nodes = [k for k,v in node_mapping.items() if v.node_type == 'forcing_from_ncfiles']
    for f in forcing_nodes:
        node_mapping[f].args['map_func'] = nodes.callable_to_funcspec(map_filename_annual)
