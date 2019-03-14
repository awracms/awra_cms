import os
        
ARRTYPE = 'double *restrict'

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

def build_struct(name,tmembers,memtype,fmembers=None):
    out = []
    out.append('typedef struct {')
    for k in tmembers:
        out.append('    %s %s;' % (memtype,k))
    if not fmembers is None:
        for k in fmembers:
            out.append('    %s;' % k)
    out.append('} %s;\n' % name)
    return out

def gen_forcing_loads(varlist):
    out = []
    for v in varlist:
        out.append('float %s = inputs.%s[idx];' % (v,v))
    return out

def gen_const_loads(varlist):
    out = []
    for v in varlist:
        out.append('double %s = params.%s;' % (v,v))
    return out

def gen_hruconst_loads(varlist):
    out = []
    for v in varlist:
        out.append('double %s = hrup.%s;' % (v,v))
    return out

def gen_spatial_loads(varlist):
    out = []
    for v in varlist:
        out.append('double %s = spatial.%s[c];' % (v,v))
    return out

def gen_hruspatial_loads(varlist):
    out = []
    for v in varlist:
        out.append('double %s = hrus.%s[c];' % (v,v))
    return out

def gen_init_combined(varlist):
    out = []
    for k in varlist:
        out.append('outputs.%s[idx] = 0.0;' % k)
    return out    

def gen_combined_writes(varlist):
    out = []
    for k in varlist:
        out.append('outputs.%s[idx] += %s*fhru;' % (k,k))
    return out

def gen_cell_writes(varlist):
    out = []
    for k in varlist:
        out.append('outputs.%s[idx] = %s;' % (k,k))
    return out

def gen_hru_writes(varlist):
    out = []
    for k in varlist:
        out.append('outputs.hru[hru].%s[idx] = %s;' % (k,k))
    return out

def gen_struct_defs(tdict):
    # Assemble input structures

    IN_ARR_TYPE = 'const double *'
    IN_FORCING_TYPE = 'const float *'
    IN_SCALAR_TYPE = 'const double'

    STRUCT_OUT = []

    struct_dict = {
        'Forcing': ('INPUTS_FORCING',IN_FORCING_TYPE),
        'HRUParameters': ('INPUTS_SCALAR_HRU', IN_SCALAR_TYPE),
        'Parameters': ('INPUTS_SCALAR',IN_SCALAR_TYPE),
        'HRUSpatial': ('INPUTS_SPATIAL_HRU', IN_ARR_TYPE),
        'Spatial': ('INPUTS_SPATIAL', IN_ARR_TYPE)
    }

    for k,v in struct_dict.items():
        cur_keys, cur_type = tdict[v[0]],v[1]
        if len(cur_keys):
            out_struct = build_struct(k,cur_keys,cur_type)
        else:
            out_struct = ['typedef void* %s;'%k]
        STRUCT_OUT += out_struct

    # Assemble output structures

    OUTPUTS = tdict['OUTPUTS_AVG'] + tdict['OUTPUTS_CELL']

    if len(tdict['OUTPUTS_HRU']):
        outmem = ['HRUOutputs hru[2]']
        HRUOUTPUTS = build_struct('HRUOutputs',tdict['OUTPUTS_HRU'],ARRTYPE)
    else:
        outmem = None
        HRUOUTPUTS = []

    if len(OUTPUTS) or len(tdict['OUTPUTS_HRU']):
        OUTPUTS = build_struct('Outputs',OUTPUTS,ARRTYPE,outmem)
    else:
        OUTPUTS = ['typedef void* Outputs;']

    out = STRUCT_OUT + HRUOUTPUTS + OUTPUTS

    return out

def gen_templates(tdict):


    return {
        'STRUCT_DEFS': gen_struct_defs(tdict),
        'WRITE_HRU': gen_hru_writes(tdict['OUTPUTS_HRU']),
        'INIT_COMBINED': gen_init_combined(tdict['OUTPUTS_AVG']),
        'WRITE_COMBINED': gen_combined_writes(tdict['OUTPUTS_AVG']),
        'WRITE_CELL': gen_cell_writes(tdict['OUTPUTS_CELL']),
        'LOAD_FORCING': gen_forcing_loads(tdict['INPUTS_FORCING']),
        'LOAD_CONST': gen_const_loads(tdict['INPUTS_SCALAR']),
        'LOAD_SPATIAL': gen_spatial_loads(tdict['INPUTS_SPATIAL']),
        'LOAD_HRUSPATIAL': gen_hruspatial_loads(tdict['INPUTS_SPATIAL_HRU']),
        'LOAD_HRUCONST': gen_hruconst_loads(tdict['INPUTS_SCALAR_HRU'])
    }

