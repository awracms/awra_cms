import os

def lower_dict(in_dict):
    res = {}
    for k,v in list(in_dict.items()):
        res[k.lower()] = v
    return res

def map_dict(base_dict, new_dict):
    for k,v in list(new_dict.items()):
        if k in base_dict:
            base_dict[k] = v
    return base_dict

def invert_dict(in_d):
    return (dict([v,k] for k,v in in_d.items()))

def pretty_format_dict(in_d):
    keys = sorted(in_d.keys())
    out_str = '{\n'
    for k in keys:
        out_str += ('%s : %s\n' % (k,in_d[k]))
    return out_str + '}'

def join_paths(*args):
    return os.path.join(*args).replace(os.path.sep,'/')

class Indexer:
    '''
    Wrapper class that refers it's get/set item methods to another function
    '''
    def __init__(self,getter_fn,setter_fn = None):
        self.getter_fn = getter_fn
        self.setter_fn = setter_fn

    def __getitem__(self,idx):
        return self.getter_fn(idx)

    def __setitem__(self,idx,value):
        return self.setter_fn(idx,value)
