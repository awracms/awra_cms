from awrams.utils.messaging.robust import PollingChild, SharedMemClient, Chunk, to_chunks
from awrams.utils.messaging.general import message
from awrams.utils.nodegraph import graph
from awrams.utils.mapping_types import gen_coordset
from awrams.utils import datetools as dt

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('writer')

import time

class OutputGraphRunner(PollingChild,SharedMemClient):
    def __init__(self,qin,qout,buffers,extents,periods,mapping):
        PollingChild.__init__(self,qin,qout)
        SharedMemClient.__init__(self,buffers)

        self._set_chunks(extents,periods)
        self.mapping = mapping
        self.cur_chunk = None
        self.cur_chunk_count = 0
        self.completed = 0
        self.finished = False

        self.daemon = True

    def run_setup(self):
        # import os
        # self.pid = os.getpid()
        # print("writer pid: %d"%self.pid,flush=True)
        #logger.info("writer pid: %d",self.pid)
        self.rebuild_buffers()
        self.exe_graph = graph.OutputGraph(self.mapping)

    def cleanup(self):
        self.exe_graph.close_all()

    def process(self):
        if self.cur_chunk is None:
            self.recv_chunk()
        else:
            self.handle_current_chunk()

    def recv_chunk(self):
        msg = self.recv('chunks')
        if msg is not None:
            self.cur_chunk = msg['content']

    def handle_current_chunk(self):
        period_idx = self.cur_chunk['period_idx']

        # if period_idx != self.cur_period_idx:
        #     self._set_cur_period(period_idx)

        chunk_idx = self.cur_chunk['chunk_idx']

        bgroup = self.map_buffer(self.cur_chunk['buffer'])
        bdata = bgroup.map_dims(self.cur_chunk['dims'])

        coords = gen_coordset(self.periods[period_idx],self.chunks[chunk_idx])
        self.exe_graph.set_data(coords,bdata,self.chunks[chunk_idx].mask)

        self.reclaim_buffer(self.cur_chunk['buffer'])
        #     in_data = self.map_buffer(shm_buf,data['shape'])
        #
        #     pwrite_idx = self.time_indices[v]
        #     chunk = self.chunks[chunk_idx]
        #
        #     write_idx = (pwrite_idx,chunk.x,chunk.y)
        #
        #     # ENFORCE_MASK +++
        #     if self.enforce_mask:
        #         subex = self.extents[chunk_idx]
        #         in_data[:,subex.mask==True] = FILL_VALUE
        #
        #     self._write_slice(write_var,in_data,write_idx)


        self.cur_chunk = None
        self.cur_chunk_count += 1

        completed = self.cur_chunk_count/len(self.chunks)*100.
        if completed - self.completed > 5:
            self._send_log(message("completed %.2f%%" % completed))
            # logger.info("completed %.2f%%",completed)
            self.completed = completed

        if self.cur_chunk_count == len(self.chunks):
            # logger.info("Completed period %s - %d of %d",dt.pretty_print_period(self.periods[self.cur_period_count]),self.cur_period_count+1,len(self.periods))
            self._send_log(message("Completed period %s - %d of %d" % (dt.pretty_print_period(self.periods[self.cur_period_count]),self.cur_period_count+1,len(self.periods))))
            self.completed = 0
            self.cur_chunk_count = 0
            self.cur_period_count += 1
            self.exe_graph.sync_all()
            if self.cur_period_count == len(self.periods):
                self._send_log(message("terminate"))
                self.terminate()

    def _write_slice(self,write_var,in_data,write_idx):
        try:
            write_var[write_idx] = in_data[:write_idx[0].stop-write_idx[0].start,:]
        except AttributeError:
            write_var[write_idx] = in_data

    def _set_chunks(self,chunks,periods):
        '''
        Order : 'chunk' or 'period'; ie which dimension to iterate first
        '''
        self.chunks = chunks
        self.periods = periods
        # print("_SET_CHUNKS",chunks,periods)

        self.cur_period_count = 0
        self.cur_chunk_count = 0

