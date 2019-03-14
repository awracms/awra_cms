'''
Custom collection and container objects

ObjectDict instances are dict-like objects supporting both dictionary indexing and property
access. For example

>>> from awrams.utils.metatypes import ObjectDict
>>> o = ObjectDict()
>>> o['a'] = 5
>>> o['b'] = o.a * 2
>>>  o
{'a': 5, 'b': 10}

>>> o.c = o.b * 2
>>> o
{'c': 20, 'a': 5, 'b': 10}

>>> o.keys()
dict_keys(['c', 'a', 'b'])
'''

from collections import OrderedDict
from abc import ABC

class AWRAMSDict(ABC):
    pass

AWRAMSDict.register(dict)

def pretty_print_dict(in_dict,level=0):
    for k in sorted(in_dict):
        if isinstance(in_dict[k],AWRAMSDict):
            print(' '*level*4,k,': {')
            pretty_print_dict(in_dict[k],level+1)
            print(' '*level*4,'}')
        else:
            print(' '*level*4,k,': ', repr(in_dict[k]))

class New(object):
    '''
    Simple wrapper enabling members to be added to live objects ala Python 3
    '''
    def __init__(self):
        pass

class Menu(object):
    '''
    Starter framework for 'fluent' python
    WIP
    '''
    def __init__(self,desc=''):
        self.__doc__=desc
        self._public = []

    def __repr__(self):
        return "%s\n%s" % (self.__doc__, str(self._public))

    def __setattr__(self,k,v):
        if k[0] != '_': #public attribute
            self._public.append(k)
        else:
            object.__setattr__(self,k,v)

class ObjectDict(object):
    '''
    Wrapper for python dict that hides dict functions, makes
    dictionary items accessable via tab completion (as well as 
    regular dictionary indexing)
    '''
    _dict_type = dict

    def __init__(self,*args, **kwargs):
        if len(args) == 1:
            source = args[0]
            if isinstance(source,ObjectDict):
                self.__dict__ = source.__dict__.copy()
            elif type(source) == self._dict_type:
                self.__dict__ = source.copy()
            else:
                # Try to init this as a dict....
                self.__dict__ = self._dict_type(source)
        elif len(args) == 0:
            self.__dict__ = self._dict_type()
        self.update(kwargs)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __setitem__(self,k,v):
        return self.__setattr__(k,v)

    def __getitem__(self,k):
        return self.__dict__[k]

    def __getattr__(self,k):
        try:
            result = getattr(self.__dict__, k)
            return result
        except:
            return object.__getattribute__(self,k)

    def __pop__(self,k):
        self.__dict__.pop(k)
        
    def __call__(self, *args, **kwargs):
        return ObjectDict(self, *args, **kwargs)

    def __clearkeys__(self):
        self.__dict__ = {}

    def __keyset__(self):
        return set(self.__dict__.keys())

    def __repr__(self):
        return self.__dict__.__repr__()

    def __len__(self):
        return len(self.__dict__)

AWRAMSDict.register(ObjectDict)

class PrettyObjectDict(ObjectDict):
    '''
    ObjectDict that prettifies its __repr__ output
    '''
    def __repr__(self):
        return pretty_format_dict(self)


class OrderedObjectDict(ObjectDict):
    _dict_type = OrderedDict

def objectify_dict_type(in_type):
    '''
    Return and ObjectDict variant for the
    appropriate type
    '''
    if in_type == dict:
        return ObjectDict
    elif in_type == OrderedDict:
        return OrderedObjectDict
    else:
        raise Exception("Unsupported type %s" % in_type)

'''
class ObjectSet(ObjectDict):
    #Provides set functions for ObjectDicts
    
    def __init__(self,items=None):
        if items != None:
            for k,v in items:
                self.__dict__[k] = v

    def __and__(lhs,rhs):
        intersection = set(lhs.__dict__.items()) & set(rhs.__dict__.items())
        return ObjectSet(intersection)

    def __or__(self,filterset):
        union = set(lhs.__dict__.items()) | set(rhs.__dict__.items())
        return ObjectSet(union)
'''

class ObjectList(list):
    '''
    Tab completeable (fixed) list class
    Note: This does not reflect changes in lists (yet)
    '''
    def __init__(self,in_list):
        list.__init__(self,in_list)
        self.items = ObjectDict()
        idx = 0
        for item in self:
            self.items['i'+str(idx)] = item
            idx += 1

class UniqueList(list):

    def append(self,item):
        if not item in self:
            list.append(self,item)

    # def append(self,item):
    #     try:
    #         has_item = False
    #         for l_i in self:
    #             if item is l_i:
    #                 has_item = True
    #         if not has_item:
    #             list.append(self,item)
    #     except:
    #         raise

def objectify(in_dict=None,to_type=PrettyObjectDict):
    '''
    Build ObjectDict iteratively (ie child dictionaries are 
        also converted to objectdicts)
    '''
    od = to_type()
    for k,v in list(in_dict.items()):
        if isinstance(v,AWRAMSDict):
            od[k] = objectify(v)
        else:
            od[k] = v
    return od

class ObjectContainer(object):
    '''
    Tab-completable OrderedDict container
    '''

    def __init__(self,*args, **kwargs):
        self._container = OrderedDict()

    def _add_item(self,key,val):
        self._container[key] = val
        self.__setattr__(key, val)
   
    def _remove_key(self,key):
        self._container.pop(key)
        self.__dict__.pop(key)

    def _remove_val(self,val):
        for item in list(self._container.items()):
            if item[1] == val:
                key_to_pop = item[0]
        self._remove_key(key_to_pop)

    def items(self):
        return self._container.items()

    def __contains__(self,k):
        return k in self._container

    def __iter__(self):
        return list(self._container.values()).__iter__()

    def __setitem__(self,k,v):
        self._add_item(k,v)

    def __getitem__(self,k):
        if type(k) == int:
            return list(self._container.values())[k]
        else:
            return self._container[k]

    def __repr__(self):
        return self._container.__repr__()

    def __eq__(self,other):
        return self._container == other._container

    def __ne__(self,other):
        return self._container != other._container

    def __len__(self):
        return len(self._container)

    def __clearkeys__(self):
        #+ backwards support for objectdict
        keys = list(self._container.keys())
        for key in keys:
            self._remove_key(key)

class AutoCallSelector(ObjectContainer):
    '''
    Provides a menu-like container allowing functions
    to be called without brackets being passed in the commandline
    '''
    def __init__(self,*args,**kwargs):
        ObjectContainer.__init__(self,args,kwargs)

    def __getattribute__(self,k):
        if k == '_container':
            return ObjectContainer.__getattribute__(self,k)
        else:
            if k in list(self._container.keys()):
                result = self._container[k]()
                return result
            else:
                return ObjectContainer.__getattribute__(self,k)

class PrettyWrapper():
    '''
    Helper for AutoCallSelector - allows 'polite' returns of a null function
    Typically used with functools.partial - return pretty_null

    '''
    def __init__(self,fn_to_wrap,repr_str):
        self._fn = fn_to_wrap
        self._repr_str = repr_str
    def __call__(self):
        self._fn()
    def __repr__(self):
        return self._repr_str

def null_fn():
    pass

def pretty_null(repr_str=''):
    return PrettyWrapper(null_fn,repr_str)

class DataDict(object):
    '''
    Queryable list of dictionaries
    '''

    def __init__(self,inputs=None):
        '''
        inputs is a container of metadata -> item mappings,
        eg
        [({'source': awra_results, 'variable': 'qtot_avg'}, var_object)]
        '''
        from collections import OrderedDict
        if inputs is None:
            inputs = []
        self.items = []
        self.keys = ObjectDict()

        if inputs:
            for i in inputs:
                self.items.append(i)
                for k,v in list(i.items()):
                    if k[0] != '_':
                        if k in list(self.keys.keys()):
                            self.keys[k].append(v)
                        else:
                            self.keys[k] = UniqueList([v])

    def add_item(self,item):
        '''
        meta is a dictionary containing identifying metadata pointing to the item
        '''
        for k,v in list(item.items()):
            if k[0] != '_':
                if k in list(self.keys.keys()):
                    self.keys[k].append(v)
                else:
                    self.keys[k]= UniqueList()
                    self.keys[k].append(v)
        self.items.append(item)

    def add_query_item(self,query_keys,item):
        '''
        meta is a dictionary containing identifying metadata pointing to the item
        '''
        for k,v in list(item.items()):
            if k in query_keys:
                if k in list(self.keys.keys()):
                    self.keys[k].append(v)
                else:
                    self.keys[k] = UniqueList()
                    self.keys[k].append(v)
        self.items.append(item)

    def query(self,*args,**kwargs):
        '''
        Query for items matching all of the supplied pattern
        E.g  query(key0 = val0, key1 = val1)
        '''
        results = DataDict()
        good_map = {}
        bad_map = {}

        if args:
            pattern = args[0]
        else:
            pattern = {}

        pattern.update(kwargs)

        for k,v in pattern.items():
            for item in self.items:
                if k in item:
                    if item[k] == v:
                        good_map[item] = item
                    else:
                        bad_map[item] = item
                else:
                    bad_map[item] = item
        for item in list(good_map.keys()):
            if item not in list(bad_map.keys()):
                results.add_item(item)
        return results

    def __getitem__(self,idx):
        return self.items[idx]

    def __iter__(self):
        return self.items.__iter__()

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        repr_items = []
        for item in self.items:
            new_item = item.copy()
            new_item['data'] = '<...>'
            repr_items.append(new_item)
        return repr_items.__repr__()

def pretty_format_dict(in_dict,level=0,tab_size=4):
    out_str = ''
    for k in sorted(in_dict):
        if isinstance(in_dict[k],AWRAMSDict):
            out_str = out_str + ' ' * level * tab_size + k + ': {\n'
            out_str = out_str + pretty_format_dict(in_dict[k],level+1,tab_size)
            out_str = out_str + ' ' * level * tab_size + '}\n'
        else:
            out_str = out_str + ' ' * level * tab_size + k + ': ' + repr(in_dict[k]) + '\n'
    return out_str