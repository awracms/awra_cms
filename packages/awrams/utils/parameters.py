import h5py
import pandas as pd
import json
from awrams.utils.nodegraph import nodes, graph
from awrams.utils.metatypes import PrettyObjectDict

def dataframe_to_h5(df,group):
    base_len = len(df)
    
    def series_to_ds(series,name):
        base_len = len(series)
        dtype = type(series[0])
        if dtype == str:
            dtype = h5py.special_dtype(vlen=str)
        ds = group.create_dataset(name, shape = (base_len,), dtype = dtype)
        for i in range(base_len):
            ds[i] = series[i]
            
    series_to_ds(df.index,'__index')
    
    for c in df.columns:
        series_to_ds(df[c],c)
        
    series_to_ds(df.columns,'__columns')

def h5_to_dataframe(group):
    df = pd.DataFrame(index=group['__index'][...],columns=group['__columns'][...])
    for c in df.columns:
        df[c] = group[c][...]
    return df

def input_map_to_param_df(input_map):
    param_map = dict([(k,v) for k,v in input_map.items() if v.node_type == 'parameter'])
    eg = graph.ExecutionGraph(param_map)
    
    out_df = pd.DataFrame(columns=['min_val','max_val','value','fixed','description'])
    
    for k,v in eg.input_graph.items():
        param = v['exe']
        out_df.loc[k] = dict(value = param.value, min_val = param.min_val, \
                           max_val = param.max_val, fixed = param.fixed, description = param.description)

    return out_df

def wirada_json_to_param_df(json_file):
    pdata_json = json.load(open(json_file,'r'))
    
    out_df = pd.DataFrame(columns=['min_val','max_val','value','fixed','description'])
    
    for param in pdata_json:
        out_df.loc[param['MemberName'].lower()] = dict(value = param['Value'], min_val = param['Min'], \
                                                       max_val = param['Max'], fixed = param['Fixed'], \
                                                       description = param['DisplayName'])
    return out_df


def param_df_to_mapping(param_df,existing=None,inplace=False):
    if existing is None:
        existing = PrettyObjectDict()
    elif inplace == False:
        existing = PrettyObjectDict(existing)
    for k,v in param_df.T.items():
        existing[k] = nodes.parameter(v['value'],v['min_val'],v['max_val'],v['fixed'],v['description'])
    return existing
