from awrams.utils.messaging.robust import *
from awrams.utils.nodegraph import graph
from awrams.utils import mapping_types as mt
from copy import deepcopy
import numpy as np

class StreamingReader(PollingChild,SharedMemClient):
    '''
    MP Reader
    '''

    def __init__(self,inq,outq,buffers,extents,periods,mapping,state_keys):
        PollingChild.__init__(self,inq,outq)
        SharedMemClient.__init__(self,buffers)

        self.finished = False
        self._set_chunks(extents,periods)
        self.mapping = mapping
        self.state_period = dict([(i,-1) for i in range(len(extents))])
        self.state_buffers = dict([(i,None) for i in range(len(extents))])
        self.state_keys = state_keys
        self.recycle_states = False
        self.cur_chunk = None
        self.finished = False
        
        self.daemon = True

    def run_setup(self):
        self.rebuild_buffers()
        self.exe_graph = graph.ExecutionGraph(self.mapping)

    def collect_states(self):
        while not self.qin['state_return'].empty():
            msg = self.qin['state_return'].get()
            m = msg['content']
            chunk = m['chunk_idx']
            buf = self.map_buffer(m['buffer'])

            self.state_period[chunk] = m['period_idx']
            
            self.state_buffers[chunk] = deepcopy(buf.map_dims(dict(cell=self.chunks[chunk].cell_count)))

            self.reclaim_buffer(m['buffer'])

    def process(self):
        try:
            self.collect_states()

            if self.cur_chunk is None:
                if not self.finished:
                    if self._state_ready():
                        self.cur_chunk = self.read_active_chunk()
                        self._select_next_chunk()
            else:
                if self.send('chunks',self.cur_chunk):
                    self.cur_chunk = None
        except ControlInterrupt:
            return
        except ChunksComplete:
            self.terminate()
            return
            
    def read_active_chunk(self):
        extent = self.chunks[self.cur_chunk_idx]
        period = self.periods[self.cur_period_idx]
        coords = mt.gen_coordset(period,extent)
        results = self.exe_graph.get_data_flat(coords,extent.mask)

        buf_id, out_buf = self.get_buffer_safe('inputs')

        #+++ Ensure all dimensions filled out; static dimensions will always be max_dims
        # Can we guarantee this?
        dims = out_buf.max_dims.copy()
        dims.update(dict(cell=extent.cell_count,time=len(period)))

        target_data = out_buf.map_dims(dims)

        for k,v in results.items():
            target_data[k][...] = v

        valid_keys = list(results.keys())

        if self.recycle_states:
            for k in self.state_keys:
                init_k = 'init_' + k

                target_data[init_k][...] = self.state_buffers[self.cur_chunk_idx][k]
                if not init_k in valid_keys:
                    valid_keys.append(init_k)

        chunk_msg = message('chunk')
        content = chunk_msg['content']
        content['chunk_idx'] = self.cur_chunk_idx
        content['period_idx'] = self.cur_period_idx
        content['buffer'] = buf_id
        content['dims'] = dims
        content['valid_keys'] = valid_keys

        return chunk_msg    

    def _init_state_cycling(self):
        ###+++?
        #
        for k in self.state_keys:
            init_k = 'init_' + k
            if init_k in self.exe_graph.input_graph:
                del(self.exe_graph.input_graph[init_k])

        self.recycle_states = True

    def _state_ready(self):
        if self.cur_period_idx == 0:
            return True
        else:
            return (self.state_period[self.cur_chunk_idx] == (self.cur_period_idx-1))

    def _set_chunks(self,chunks,periods):
        '''
        Order : 'chunk' or 'period'; ie which dimension to iterate first
        '''
        self.chunks = chunks
        self.periods = periods

        self.cur_period_idx = 0
        self.cur_chunk_idx = 0

    def _select_next_chunk(self):
        self.cur_chunk_idx += 1
        if self.cur_chunk_idx == len(self.chunks):
            self.cur_chunk_idx = 0
            self.cur_period_idx +=1
            if self.cur_period_idx == len(self.periods):
                self.finished = True
            if self.cur_period_idx == 1:
                self._init_state_cycling()
