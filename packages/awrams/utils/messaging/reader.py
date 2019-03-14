"""
Standalone input reader module
Communicates with the rest of the system via ZMQ
"""

from multiprocessing.queues import Empty
import time
import numpy as np

from awrams.utils.messaging.general import message
#from Support.Messaging.binding import QueueChild
from awrams.utils.messaging.robust import PollingChild, SharedMemClient, Chunk, ChunksComplete, ControlInterrupt, shape_idx

from awrams.utils.io import db_open_with
db_opener = db_open_with()

CHUNKS_FIRST=0
PERIODS_FIRST=1

MAX_RETRIES = 3

def to_chunk_idx(cell,c_shape):
    return (cell[0]/c_shape[0],cell[1]/c_shape[1])

def offset_slice(in_slice,offset):
    return slice(in_slice.start+offset,in_slice.stop+offset)

class StreamingReader(PollingChild,SharedMemClient):
    '''
    File_maps is a dictionary of variable: fmap
    fmap is a dict of period_idx: {filename: value, time_index: value}
    '''

    def __init__(self,qin,qout,buffers,file_maps,chunks,periods,order=None):
        PollingChild.__init__(self,qin,qout)
        SharedMemClient.__init__(self,buffers)

        self.file_maps = file_maps

        self.variables = list(file_maps.keys())

        self.chunks = chunks
        self.periods = periods
        self.order = order

        self.cur_f = dict([(k,None) for k in self.variables])

        self.cur_ds = dict([(k,None) for k in self.variables])
        self.time_indices = dict([(k,None) for k in self.variables])
        self.nc_vars = dict([(k,v['nc_var']) for k,v in self.file_maps.items()])

        self._file_mode = 'r'

        
    def run_setup(self):
        self.rebuild_buffers()
        self._set_chunks(self.chunks,self.periods,self.order)
        self.cur_chunk = None

    def cleanup(self):
        self.close_all()

    def close_all(self):
        '''
        Close all open datasets
        '''

        for variable in self.variables:
            if self.cur_ds[variable] is not None:
                self.cur_ds[variable].close()

    def process(self):
        if self.cur_chunk is None:
            try:
                self._select_next_chunk()
                self.cur_chunk = self.read_active_chunk()#
            except ControlInterrupt:
                return
            except ChunksComplete:
                self.terminate()
                return
        else:
            if self.send('chunks',self.cur_chunk):
                self.cur_chunk = None

    def read_active_chunk(self):
        chunk_msg = message('chunk')
        content = chunk_msg['content']
        content['chunk_idx'] = self.cur_chunk_idx
        content['period_idx'] = self.cur_period_idx
        data = content['data'] = {}

        chunk = self.chunks[self.cur_chunk_idx]

        for variable in self.variables:
            if self.cur_ds[variable] is not None:

                #+++
                # Need to ensure timeout/polling 
                # on get_buffer; this should probably
                # be made more general...
                buf = None
                while buf is None:
                    try:
                        buf, arr = self.get_buffer()
                    except Empty:
                        if self.poll_control():
                            raise ControlInterrupt

                read_complete = False
                cur_retries = 0

                time_idx = self.time_indices[variable]

                if isinstance(time_idx,slice):
                    time_shape = time_idx.stop-time_idx.start
                elif isinstance(time_idx,int):
                    time_shape = 1
                else:
                    time_shape = len(time_idx)

                out_shape = (time_shape,chunk.shape[0],chunk.shape[1])
                write_idx = [slice(0,time_shape)] + shape_idx(chunk.shape)

                while not read_complete:
                    try:
                        read_var = self.cur_ds[variable][self.nc_vars[variable]]
                        arr[write_idx] = read_var[time_idx,chunk.x,chunk.y]
                        read_complete = True
                    except:
                        if cur_retries < MAX_RETRIES:
                            print("Retrying read of %s (%s)" % (variable,self.cur_f[variable]))
                            time.sleep(1)
                            self.cur_ds[variable].close()
                            self.cur_ds[variable] = db_opener(self.cur_f[variable],'r')
                            cur_retries += 1
                        else:
                            raise

                data[variable] = dict(buffer = buf, shape = out_shape)

        return chunk_msg


    def _set_chunks(self,chunks,periods,order=None):
        '''
        Order : 'chunk' or 'period'; ie which dimension to iterate first
        '''
        self.chunks = chunks
        self.periods = periods
        
        if order is None:
            order = CHUNKS_FIRST

        self.order = order

        if order == CHUNKS_FIRST:
            self.cur_period_idx = 0
            self.cur_chunk_idx = -1
        elif order == PERIODS_FIRST:
            self.cur_period_idx =-1
            self.cur_chunk_idx = 0

        self._set_cur_period(0)

    def _select_next_chunk(self):
        if self.order == CHUNKS_FIRST:
            self.cur_chunk_idx += 1
            if self.cur_chunk_idx == len(self.chunks):
                self.cur_chunk_idx = 0
                self.cur_period_idx +=1
                if self.cur_period_idx == len(self.periods):
                    raise ChunksComplete
                self._set_cur_period(self.cur_period_idx)
        elif self.order == PERIODS_FIRST:
            self.cur_period_idx +=1
            if self.cur_period_idx == len(self.periods):
                self.cur_period_idx = 0
                self.cur_chunk_idx += 1
                if self.cur_chunk_idx == len(self.chunks):
                    raise ChunksComplete
            self._set_cur_period(self.cur_period_idx)

    def _clear_open(self,variable):
        '''
        Close any open files and clear locals relating to the specified variable
        '''
        if self.cur_ds[variable] is not None:
            self.cur_ds[variable].close()
            self.cur_ds[variable] = None
        self.cur_f[variable] = None

    def _set_cur_period(self,pidx):
        '''
        Load appropriate files for current active period
        '''
        for variable in self.variables:
            fmap = self.file_maps[variable]
            pmap = fmap['period_map'].get(pidx)
            if pmap is None:
                self._clear_open(variable)
            else:
                filename = pmap['filename']
                if self.cur_f[variable] != filename:
                    self._clear_open(variable)
                    self.cur_ds[variable] = db_opener(filename,self._file_mode)
                    self.cur_f[variable] = filename 
                self.time_indices[variable] = pmap['time_index']


class StreamingReaderAggregate(StreamingReader):
    '''
    File_maps is a dictionary of variable: fmap
    fmap is a dict of period_idx: {filename: value, time_index: value}
    '''
    def __init__(self,qin,qout,buffers,file_maps,chunks,periods,seasons,order=None,num_agg=3):
        StreamingReader.__init__(self,qin,qout,buffers,file_maps,chunks,periods,order)

        self.seasons = seasons
        self.periods = periods
        self.season_indices = dict([(k,None) for k in self.variables])
        self.num_agg = num_agg

    def read_active_chunk(self):
        chunk_msg = message('chunk')
        content = chunk_msg['content']
        content['chunk_idx'] = self.cur_chunk_idx
        content['period_idx'] = self.cur_period_idx
        data = content['data'] = {}

        chunk = self.chunks[self.cur_chunk_idx]

        for variable in self.variables:
            if self.cur_ds[variable] is not None:

                #+++
                # Need to ensure timeout/polling
                # on get_buffer; this should probably
                # be made more general...
                buf = None
                while buf is None:
                    try:
                        buf, arr = self.get_buffer()
                    except Empty:
                        if self.poll_control():
                            raise ControlInterrupt

                read_complete = False
                cur_retries = 0

                time_idx = self.time_indices[variable]

                season_shape = self.season_indices[variable]

                out_shape = (season_shape,chunk.shape[0],chunk.shape[1])
                write_idx = [slice(0,season_shape)] + shape_idx(chunk.shape)

                while not read_complete:
                    try:
                        read_var = self.cur_ds[variable][self.nc_vars[variable]]
                        ### for non-overlapping windows...window size must be <=12 months
                        # arr[write_idx] = read_var[time_idx,chunk.x,chunk.y].reshape((self.num_agg,-1,out_shape[1],out_shape[2]),order='F').mean(axis=0)
                        ### for moving window...ie kernel moves 1 month steps
                        in_data = read_var[time_idx,chunk.x,chunk.y]
                        kernel = np.ones((self.num_agg,))/float(self.num_agg)
                        arr[write_idx] = (np.apply_along_axis(lambda m: np.convolve(m,kernel,mode='valid'), axis=0, arr=in_data)[-1::-12,:])[::-1,:]
                        read_complete = True
                    except AttributeError:
                        if cur_retries < MAX_RETRIES:
                            print("Retrying read of %s (%s)" % (variable,self.cur_f[variable]))
                            time.sleep(1)
                            self.cur_ds[variable].close()
                            # h5py_cleanup_nc_mess()
                            self.cur_ds[variable] = db_opener(self.cur_f[variable],'r')
                            cur_retries += 1
                        else:
                            raise

                data[variable] = dict(buffer = buf, shape = out_shape)

        return chunk_msg

    def _set_cur_period(self,pidx):
        '''
        Load appropriate files for current active period
        '''
        for variable in self.variables:
            fmap = self.file_maps[variable]
            pmap = fmap['period_map'].get(pidx)

            if pmap is None:
                self._clear_open(variable)
            else:
                filename = pmap['filename']
                if self.cur_f[variable] != filename:
                    self._clear_open(variable)
                    self.cur_ds[variable] = db_opener(filename,self._file_mode)
                    self.cur_f[variable] = filename
                self.time_indices[variable] = pmap['time_index']
                self.season_indices[variable] = len(self.seasons[pidx])
