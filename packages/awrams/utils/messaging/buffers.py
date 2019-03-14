'''
Support classes for providing shared memory numpy arrays

+++
We are moving away from using python multiprocessing/shared memory,
so most of this will be deprecated and/or refactored into non-shm classes (eg to
    provide support for local buffer groups/queues)

'''

import multiprocessing as mp
import numpy as np
import ctypes
from awrams.utils.metatypes import ObjectDict as o

_ctypes_to_numpy = {
    ctypes.c_float  : np.dtype('float32'),
    ctypes.c_double : np.dtype('float64'),
                'd' : np.dtype('float64')}

_numpy_to_ctypes = dict(list(zip(list(_ctypes_to_numpy.values()),
                            list(_ctypes_to_numpy.keys()))))

def shm_as_ndarray(mp_array, shape=None,order='C'):
    '''Given a multiprocessing.Array, returns an ndarray pointing to
    the same data.'''

    #+++
    #Need to take into account Fortran/C ordering

    # support SynchronizedArray:
    if not hasattr(mp_array, '_type_'):
        mp_array = mp_array.get_obj()
    dtype = _ctypes_to_numpy[mp_array._type_]
    result = np.frombuffer(mp_array, dtype)

    if shape is not None:
        result = result.reshape(shape,order=order)

    return np.asarray(result)

def init_shm_ndarray(shape,dtype=np.float64):
    '''Initialize a shared memory array of the supplied shape and dtype'''
    shm = mp.Array(_numpy_to_ctypes[np.dtype(dtype)],int(np.prod(shape)),lock=False)
    return shm

def init_shm_value(dtype=np.float64):
    '''Initialize a shared memory value of the supplied dtype'''
    shm = mp.Value(_numpy_to_ctypes[np.dtype(dtype)],lock=False)
    return shm

class BufferManager:
    '''
    Manages a queue of shared memory numpy arrays
    '''
    def __init__(self,buffers,queue,shape=None,build=True):
        self.queue = queue
        self.buffers_shm = buffers
        self.buf_shape = shape
        if build:
            self.rebuild_buffers()

    def rebuild_buffers(self):
        self.buffers_np = []
        for b in self.buffers_shm:
            self.buffers_np.append(shm_as_ndarray(b,self.buf_shape))

    def get_buffer(self,timeout=None):
        '''
        Retrieve a buffer from the queue
        '''
        buffer_id = self.queue.get(timeout=timeout)
        return buffer_id, self.map_buffer(buffer_id)

    def reclaim(self,buffer_id):
        '''
        Return a used buffer to the queue
        '''
        self.queue.put(buffer_id)

    def map_buffer(self,buffer_id,indices=None):
        '''
        Return mapped nd_arrays of the appropriate shape
        '''
        if not indices:
            return self.buffers_np[buffer_id]
        else:
            return self.buffers_np[buffer_id][indices]



def create_managed_buffers(count,shape,dtype=np.float64,build=True):
    buffers,q = create_buffers(count,shape,dtype)
    return BufferManager(buffers,q,shape,build)

def create_buffers(buf_count,buf_shape,dtype=np.float64):
    '''
    Initialise buf_count buffers of supplied shape and datatype
    '''
    buffers = []
    q = mp.Queue()
    for i in range(buf_count):
        buffers.append(init_shm_ndarray(buf_shape,dtype))  
        q.put(i)

    return buffers,q

def create_dict_buffers(recordables,buf_count,buf_len,dtype=np.float64):
    '''
    Initialise buf_count dictionaries of supplied length and datatype
    '''
    buffers = []
    q = mp.Queue()
    for i in range(buf_count):
        cur_buf_shm = {}
        for k in recordables:
            cur_buf_shm[k] = init_shm_ndarray(buf_len,dtype)
        buffers.append(cur_buf_shm)  
        q.put(i)

    return buffers,q

def create_shm_dict(keys,shape,dtype=np.float64):
    '''
    Create dict of shm buffers
    '''
    buffers = o()
    shapes = o()
    for k in keys:
        buffers[k] = init_shm_ndarray(shape,dtype)
        shapes[k] = shape
    return o(buffers=buffers,shape=shapes)

def create_shm_dict_inputs(shape_map,dtype=np.float64):
    '''
    Create dict of shm buffers
    '''
    buffers = o()
    shapes = o()
    for k in shape_map:
        if shape_map[k] is None or len(shape_map[k])==0:
            buffers[k] = init_shm_ndarray((1,),dtype)
        else:
            buffers[k] = init_shm_ndarray(shape_map[k],dtype)
        shapes[k] = shape_map[k]
    return o(buffers=buffers,shapes=shapes)

def shm_to_nd_dict_inputs(buffers,shapes,order='C'):
    '''
    Convert a dictionary of shared memory arrays into process-local numpy arrays
    Shape can be a single common shape, or a dictionary of shapes
    '''
    nd_dict = o()

    for k,v in list(buffers.items()):
        if shapes[k] is None or len(shapes[k])==0:
            nd_dict[k] = shm_as_ndarray(v,(1,),order) #np.asscalar(a)
        else:
            nd_dict[k] = shm_as_ndarray(v,shapes[k],order)
    return nd_dict

def shm_to_nd_dict(buffers,shape=None,order='C'):
    '''
    Convert a dictionary of shared memory arrays into process-local numpy arrays
    Shape can be a single common shape, or a dictionary of shapes
    '''
    nd_dict = o()
    to_shape = shape
    for k,v in list(buffers.items()):
        if hasattr(shape,'items'):
            to_shape = shape[k]
        nd_dict[k] = shm_as_ndarray(v,to_shape,order)
    return nd_dict

class BufferedDictHandler:
    '''
    Mixin providing base functionality for Manager classes
    '''
    def __init__(self,buffers,build=True):
        self.buffers_shm = buffers
        self.buffers_nd = {}
        if build:
            self.rebuild_buffers()

    def rebuild_buffers(self):
        '''
        Reconvert shared-mem buffers to numpy
        (e.g after passing process boundaries)
        '''
        for k, v in self.buffers_shm.items():
            self.buffers_nd[k] = (shm_to_nd_dict(**v))

    def map_buffer(self,buffer_id,index=None):
        '''
        Return mapped nd_arrays of the appropriate length
        '''
        out_bufs = self.buffers_nd[buffer_id]
        
        if index is None:
            return out_bufs
        else:
            sub_out = {}
            for k,v in out_bufs.items():
                sub_out[k] = v[index]
            return sub_out
  

class BufferedDictManager(BufferedDictHandler):
    '''
    Manages shared memory dictionaries of numpy arrays
    '''
    def __init__(self,buffers,queue,build=True):
        super().__init__(buffers,build)
        self.queue = queue


    def get_buffer(self,index=None,timeout=None):
        '''
        Retrieve a buffer from the queue
        '''
        buffer_id = self.queue.get(timeout=timeout)
        buffers = self.map_buffer(buffer_id,index=index)
        return buffer_id, buffers

    def reclaim(self,buffer_id):
        '''
        Return a used buffer to the queue
        '''
        self.queue.put(buffer_id)


class MultiClientBufferedDictManager(BufferedDictHandler):
    '''
    Manages buffered dicts that will be used by multiple clients simultaneously
    Use BufferedDictManager for client access
    '''
    def __init__(self,buffers,client_queues,build=True):
        super().__init__(buffers,build)
        self.client_queues = client_queues

    def reclaim(self,buffer_id):
        '''
        Return a used buffer to the queue
        '''
        for q in self.client_queues:
            q.put(buffer_id)

    def get_client_managers(self):
        return [BufferedDictManager(self.buffers_shm,q,False) for q in self.client_queues]

    def get_handler(self):
        return BufferedDictHandler(self.buffers_shm,False)


def create_multiclient_manager(keys,shape,nbuffers,nclients,build=True,extra_bufs=None):
    client_queues = [mp.Queue() for q in range(nclients)]

    bufs = {}
    for i in range(nbuffers):
        buf = create_shm_dict(keys,shape)
        [q.put(i) for q in client_queues]
        bufs[i] = buf
    if extra_bufs is not None:
        bufs.update(extra_bufs)

    return MultiClientBufferedDictManager(bufs,client_queues,build)
