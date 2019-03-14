import numpy as np
from awrams.utils.messaging import buffers
import multiprocessing as mp


class DataSpec:
    def __init__(self,valtype,dims,dtype):
        self.valtype = valtype
        self.dims = dims
        self.dtype = dtype

    def __repr__(self):
        return '%s, %s %s' % (self.valtype,self.dtype,self.dims)

class BufferGroup:
    def __init__(self,buf_specs,max_dims,build_nd=True):
        self._validate(buf_specs,max_dims)        
        self.buffers = {}
        self.buf_specs = buf_specs
        self.max_dims = max_dims

        for k, v in buf_specs.items():
            self.buffers[k] = buffers.init_shm_ndarray([max_dims[d] for d in v.dims],v.dtype)
        
        self._ndflat = {}
            
        if build_nd:
            self._build_nd()
            
    def _build_nd(self):
        for k in self.buf_specs:
            self._ndflat[k] = buffers.shm_as_ndarray(self.buffers[k])
            
    def map_dims(self,dims):

        out_nd = {}

        msize = {}
        for k,spec in self.buf_specs.items():
            shape = tuple([dims[d] for d in spec.dims])
            size = msize.get(shape)
            if size is None:
                size = msize[shape] = np.product(shape)
            out_nd[k] = self._ndflat[k][0:size].reshape(shape)
        return out_nd
        
    def _validate(self,buf_specs,max_dims):
        for k,v in buf_specs.items():
            for dim in v.dims:
                if not dim in max_dims:
                    raise Exception("No maximum size specified for %s:%s" % (k,dim))

class BufferGroupManager:
    def __init__(self,buffers,queue,build=True):
        self.queue = queue
        self.buffers = buffers
        if build:
            self.rebuild_buffers()

    def rebuild_buffers(self):
        for b in self.buffers:
            b._build_nd()

    def get_buffer(self,timeout=None):
        buffer_id = self.queue.get(timeout=timeout)
        return buffer_id, self.buffers[buffer_id]

    def reclaim(self,buffer_id):
        self.queue.put(buffer_id)

    def map_buffer(self,buffer_id,meta=None):
        '''
        Return mapped nd_arrays of the appropriate shape
        '''
        return self.buffers[buffer_id]

def create_managed_buffergroups(dspecs,max_dims,count,build=False):
    bufs = []
    q = mp.Queue()

    for i in range(count):
        bgroup = BufferGroup(dspecs,max_dims,False)
        bufs.append(bgroup)
        q.put(i)

    return BufferGroupManager(bufs,q,build)