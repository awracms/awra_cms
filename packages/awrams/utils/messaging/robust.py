from multiprocessing import Process
from multiprocessing.queues import Empty, Full
from awrams.utils.messaging.general import message

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('robust')

class ControlInterrupt(Exception):
    pass


class ChunksComplete(Exception):
    pass


def chunk_message(chunk_idx,period_idx,data=None):
    chunk_msg = message('chunk')
    content = chunk_msg['content']
    content['chunk_idx'] = chunk_idx
    content['period_idx'] = period_idx

    if data is None:
        data = {}

    content['data'] = data

    return chunk_msg

def to_chunks(sub_extents):
    '''
    Convert a group of extents to their indices
    '''
    return [Chunk(*s.indices) for s in sub_extents]

def subdivide(length,chunksize):
    '''
    subdivide a 1-dimensional line of length (length) into slices of size (chunksize)
    '''
    from numpy import ceil, arange

    ngroups = int(ceil((length/chunksize)-1))
    start = arange(ngroups+1)*chunksize
    end = start+chunksize
    end[-1] = length
    return [slice(start[i],end[i]) for i in range(ngroups+1)]

class Chunk:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        xshape = x.stop-x.start if isinstance(x,slice) else 1
        yshape = y.stop-y.start if isinstance(y,slice) else 1
        self.shape = (xshape,yshape)

    def __repr__(self):
        return str((self.x,self.y))

def shape_idx(shape):
    return [slice(0,n) for n in shape]

class SharedMemClient:
    def __init__(self,buffers,build=False):
        self.buffers = buffers
        if build:
            self.rebuild_buffers()

    def rebuild_buffers(self):
        #Call after a process boundary to remap sharedmem->np
        for m in self.buffers.values():
            m.rebuild_buffers()

    def get_buffer(self,pool='main',timeout=1):
        buf_id, np_arr = self.buffers[pool].get_buffer(timeout=timeout)
        return dict(pool=pool,id=buf_id), np_arr

    def reclaim_buffer(self,buf):
        self.buffers[buf['pool']].reclaim(buf['id'])

    def map_buffer(self,buf,shape=None):
        if not shape is None:
            shape = shape_idx(shape)
        return self.buffers[buf['pool']].map_buffer(buf['id'],shape)

    def get_buffer_safe(self,pool='main'):
        buf = None
        while buf is None:
            try:
                buf, arr = self.get_buffer(pool)
            except Empty:
                if self.poll_control():
                    raise ControlInterrupt
        return buf, arr

class ControlMaster(Process):
    def __init__(self,controlq,statusq,qs):
        Process.__init__(self)
        self.controlq = controlq
        self.statusq = statusq
        self.qs = qs
        self.exception_raised = False

    def run(self):
        try:
            self.active = True
            while self.active:
                self.poll_control()
        except:
            pass

    def poll_control(self):
        # if not self.controlq.empty():
        msg = self.controlq.get()
        if msg['subject'] == 'terminate':
            self.active = False
            self.exception_raised = True
            self.send()
        elif msg['subject'] == 'finished':
            self.active = False
            self.send()


    def send(self):
        try:
            msg = message('terminate')
            for q in self.qs:
                q.put(msg,timeout=1)
            self.active = False
            if self.exception_raised:
                self.statusq.put({'content': {}, 'subject': 'exception_raised'})
            else:
                self.statusq.put({'content': {}, 'subject': 'finished'})
        except Full:
            raise

class PollingChild(Process):
    def __init__(self,qin,qout):
        Process.__init__(self)
        self.qin = qin
        self.qout = qout

    def run(self):
        try:
            self.run_setup()
            self.active = True
            while self.active:
                if not self.poll_control():
                    self.process()

        except:
            import traceback
            import sys

            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("*** print_tb:")
            traceback.print_tb(exc_traceback)
            print("*** print_exception:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)

            self.qout['control'].put(message('terminate'))

        finally:
            self.cleanup()
            self.qout['control'].put(message('terminated',pid=self.pid,obj_class=self.__class__.__name__))

    def run_setup(self):
        pass

    def cleanup(self):
        pass

    def process(self):
        '''
        The main work loop; any calls to queues
        in here must time out in order to check
        for control messages
        '''
        raise Exception("Not implemented")

    def poll_control(self):

        if not self.qin['control'].empty():
            msg = self.qin['control'].get()
            return self._handle_control(msg)
        else:
            return False

    def terminate(self):
        self.active = False

    def _handle_control(self,msg):
        '''
        Return True if there is a termination condition
        '''
        if msg['subject'] == 'terminate':
            self.active = False
            return True
        else:
            return False

    def send(self,target,msg):
        try:
            self.qout[target].put(msg,timeout=1)
            return True
        except Full:
            return False

    def recv(self,source):
        try:
            msg = self.qin[source].get(timeout=1)
            return msg
        except Empty:
            return None

    def _send_log(self,msg):
        self.qout['log'].put(msg)


class StubWorker(PollingChild,SharedMemClient):
    def __init__(self,qin,qout,buffers):
        PollingChild.__init__(self,qin,qout)
        SharedMemClient.__init__(self,buffers)

    def run_setup(self):
        #self.rebuild_buffers()
        self.cur_chunk = None

    def process(self):
        if self.cur_chunk is None:
            self.recv_chunk()
        else:
            self.handle_current_chunk()

    def recv_chunk(self):
        msg = self.recv('chunks')
        if msg is not None:
            self.cur_chunk = msg

    def handle_current_chunk(self):
        '''
        Send back any buffers we've received, and generate some output
        '''
        c = self.cur_chunk['content']

        period_idx = c['period_idx']
        chunk_idx = c['chunk_idx']

        if self.send('chunks',self.cur_chunk):
            self.cur_chunk = None

class StreamingIterator(PollingChild,SharedMemClient):
    '''
    File_maps is a dictionary of variable: fmap
    fmap is a dict of period_idx: {filename: value, time_index: value}
    '''

    def __init__(self,qin,qout,buffers):
        PollingChild.__init__(self,qin,qout)
        SharedMemClient.__init__(self,buffers)
        
    def run_setup(self):
        self.rebuild_buffers()
        self.cur_item = None

    def cleanup(self):
        pass

    def get_buffer_safe(self):
        buf = None
        while buf is None:
            try:
                buf, arr = self.get_buffer()
            except Empty:
                if self.poll_control():
                    raise ControlInterrupt
        return buf, arr

    def process(self):
        if self.cur_item is None:
            try:
                buf,arr = self.get_buffer_safe()
                item_id = self.get_direct(arr)
                self.cur_item = (item_id,buf)
            except ControlInterrupt:
                return
            except StopIteration:
                self.terminate()
                return
        else:
            if self.send('data',self.cur_item):
                self.cur_item = None

    def get_direct(self,arr):
        raise Exception("Not implemented")

