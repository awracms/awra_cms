from awrams.utils.ts import processing
from awrams.utils.messaging.robust import *

FILL_VALUE = -999.


class ChunkedTimeResampler(PollingChild,SharedMemClient):
    def __init__(self,qin,qout,buffers,sub_extents,in_periods,out_freq,method='sum',enforce_mask=True,use_weights=False):
        PollingChild.__init__(self,qin,qout)
        SharedMemClient.__init__(self,buffers)

        self.sub_extents = sub_extents
        self.in_periods = in_periods
        self.out_freq = out_freq
        self._methodstr = method
        self._weighted = use_weights
        self.enforce_mask = enforce_mask

    def run_setup(self):
        self.rebuild_buffers()
        self.cur_chunk = None
        self.out_chunk = None

        self.resample_indices = []

        if self._weighted:
            self.weights = []

        for p in self.in_periods:
            if self._weighted:
                res_idx,weights = processing.build_weighted_resample_index(p,self.out_freq)
                self.weights.append(weights)
            else:
                res_idx = processing.build_resample_index(p,self.out_freq)
            self.resample_indices.append(res_idx)

        import numpy as np
        self.method = getattr(np,self._methodstr)

    def process(self):
        if self.cur_chunk is None:
            self.recv_chunk()
        else:
            if self.out_chunk is None:
                self.handle_current_chunk()
            else:
                self.send_out_chunk()

    def recv_chunk(self):
        msg = self.recv('chunks')
        if msg is not None:
            self.cur_chunk = msg

    def get_buffer_safe(self):
        buf = None
        while buf is None:
            try:
                buf, arr = self.get_buffer()
            except Empty:
                if self.poll_control():
                    raise ControlInterrupt
        return buf, arr

    def handle_current_chunk(self):
        '''
        Send back any buffers we've received, and generate some output
        '''
        c = self.cur_chunk['content']
        period_idx = c['period_idx']
        chunk_idx = c['chunk_idx']
        in_data = c['data']

        ridx = self.resample_indices[period_idx]

        subex = self.sub_extents[chunk_idx]

        data = {}

        for k,v in in_data.items():
            in_buf = v['buffer']

            if len(ridx) > 0:
                in_arr = self.map_buffer(in_buf,v['shape'])

                out_buf, out_arr = self.get_buffer_safe()

                if self._weighted:
                    res_data = processing.resample_with_weighted_mean(in_arr,ridx,self.weights[period_idx])
                else:
                    res_data = processing.resample_with_index(in_arr,ridx,self.method)
            
                out_arr[shape_idx(res_data.shape)] = res_data

                if self.enforce_mask:
                    out_arr[:,subex.mask==True] = FILL_VALUE

                data[k] = dict(buffer = out_buf, shape = res_data.shape)

            self.reclaim_buffer(in_buf)

        self.out_chunk = chunk_message(chunk_idx,period_idx,data)
        self.send_out_chunk()

    def send_out_chunk(self):
        if self.send('chunks',self.out_chunk):
            self.cur_chunk = None
            self.out_chunk = None