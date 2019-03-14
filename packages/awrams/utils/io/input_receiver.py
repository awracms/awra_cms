from awrams.utils.messaging.general import *
import zmq
import uuid
import subprocess as subprocess
import os
import awrams.utils.awrams_log
from awrams.utils import extents

logger = awrams.utils.awrams_log.get_module_logger('input_receiver')

AWRAPATH=os.path.dirname(__file__)

class StreamingInputBridge:
    def __init__(self,control_p=None,req_p=None,launch_reader=True):
        
        self.cur_chunk_idx = -1
        self.cur_data = {}

        ctx = zmq.Context()

        if control_p is None:
            control_p = gen_ipc_handle()

        if req_p is None:
            req_p = gen_ipc_handle()

        self.req_s = ctx.socket(zmq.PAIR)
        self.req_s.connect(req_p)

        if launch_reader:
            self.reader_p = subprocess.Popen(['python', os.path.join(AWRAPATH,'awra_input_reader.py'), control_p, req_p])
            if self.reader_p.poll() is not None:
                raise Exception("Error running reader process")

    def terminate(self):
        #+++ Need to handle case of terminating when not launching subprocess...
        self.req_s.send_pyobj(message('terminate'))
        self.reader_p.wait()

    def send(self,m):
        self.req_s.send_pyobj(m)

    def recv(self):
        m = self.req_s.recv_pyobj()

        if m['subject'] == 'exception':
            m = m['content']
            logger.critical("Exception from reader process:\n%s", m['traceback'])
            raise m['exception']

        return m

    def recv_array(self):
        return recv_array(self.req_s)

    def setup(self,inputs,periods,extent,mode='global'):
        '''
        inputs : {variable_name : {year0: filename, year1: filename}} 
        periods : [period0, period1] etc
        extent : awra extent object
        '''
        self.set_inputs(inputs)
        self.set_extent(extent,mode)
        self.set_periods(periods)

    def set_extent(self,extent,mode='global'):
        m = message('set_extent', extent=extent, mode=mode)
        self.send(m)
        r = self.recv()['content']
        self.chunk_map = r['chunks']
        self.geo_ref = r['geo_ref']

    def set_periods(self,periods):
        m = message('set_periods', periods = periods)
        self.send(m)
        self.recv()
        self.cur_period_idx = -1

    def set_inputs(self,inputs):
        '''
        inputs is a dictionary of the format:
        {variable_name : {year0: filename, year1: filename}} etc
        '''
        imap = {}
        for k,v in list(inputs.items()):
            imap[k] = v.get_year_map()
            v.connect_reader_bridge(self)
        m = message('set_inputs', file_maps = imap)
        self.send(m)
        self.recv()

    def set_active_period(self,period):
        pass

    def get_chunks(self,chunk_idx,period_idx):
        if (self.cur_chunk_idx != chunk_idx) or (self.cur_period_idx != period_idx):
            m = message('request_chunks',chunk_idx=chunk_idx,period_idx=period_idx)
            self.req_s.send_pyobj(m)

            #+++
            #Inserting poll to trap timeouts

            if self.req_s.poll(10000) == 0:
                raise Exception("Timed out waiting for input reader")

            reply = self.recv()

            msg = reply['subject']

            if msg == 'transmit_chunks':
                if reply['content']['chunk_idx'] != chunk_idx:
                    raise Exception("Chunk index mismatch")

# +++ if no data is found for period then reply['variables'] is empty, and silently returns previously read self.cur_data
                for v in reply['content']['variables']:
                    self.cur_data[v] = self.recv_array()

                self.cur_chunk_idx = chunk_idx
                self.cur_period_idx = period_idx

            return dict((v,self.cur_data[v]) for v in reply['content']['variables'])
# +++ DS to review
        return self.cur_data

    def get_chunk(self,variable,chunk_idx,period_idx):
        # if (self.cur_chunk_idx != chunk_idx) or (self.cur_period_idx != period_idx):
        #     m = message('request_chunks',chunk_idx=chunk_idx,period_idx=period_idx)
        #     self.req_s.send_pyobj(m)

        #     reply = self.recv()

        #     msg = reply['message']

        #     if msg == 'transmit_chunks':
        #         if reply['chunk_idx'] != chunk_idx:
        #             raise Exception("Chunk index mismatch")

        #         for v in reply['variables']:
        #             self.cur_data[v] = self.recv_array()

        #         self.cur_chunk_idx = chunk_idx
        #         self.cur_period_idx = period_idx
        self.get_chunks(chunk_idx,period_idx)
        return self.cur_data[variable]

class ClimateInputBridge:
    """
    Wrapper for the simulation server which provides cells in order
    """
    def __init__(self, inputs, periods, extent,existing_inputs=False):

        if existing_inputs:
            mapped_inputs = inputs
        else:
            from awrams.utils.ts.gridded_time_series import SplitDataSet
            mapped_inputs = dict((k,SplitDataSet(k,path)) for k,path in list(inputs.items()))

        self.bridge = StreamingInputBridge()
        self.bridge.setup(mapped_inputs,periods,extent)

        self.extent = extents.from_boundary_coords(*self.bridge.geo_ref.bounds_args(),parent_ref=self.bridge.geo_ref,compute_areas=False)

        self.cur_chunk = NULL_CHUNK
        self.cur_chunk_idx = -1
        self.cur_period_idx = -1

        self._terminated = False


    def get_cell_dict(self,cell,buffer_dict=None):
        '''
        Write cells to the supplied buffers; return keys of data written
        +++ : should convert input bridge to use python shared mem
        '''
        lcell = self.extent.localise_cell(cell)

        if not self.cur_chunk.contains(lcell):
            self.cur_chunk_idx += 1
            self.cur_chunk = self.bridge.chunk_map[self.cur_chunk_idx]
            self.cur_data = self.bridge.get_chunks(self.cur_chunk_idx,self.cur_period_idx)

        cell_idx = self.cur_chunk.idx(lcell)

        if buffer_dict is not None:
            for k,v in list(self.cur_data.items()):
                buffer_dict[k][:self.cur_p_len] = v[:,cell_idx]
            return list(self.cur_data.keys())
        else:
            out_data = {}
            for k,v in list(self.cur_data.items()):
                out_data[k] = v[:,cell_idx]
            return out_data

    def retrieve_cell_dict(self,cell):
        lcell = self.extent.localise_cell(cell)

        if not self.cur_chunk.contains(lcell):
            # Reclaim chunk buffers first
            self.cur_chunk_idx += 1
            self.cur_chunk = self.bridge.chunk_map[self.cur_chunk_idx]
            self.cur_data = self.bridge.get_chunks(self.cur_chunk_idx,self.cur_period_idx)

        cell_idx = self.cur_chunk.idx(lcell)

        out_data = {}

        for k,v in list(self.cur_data.items()):
            out_data[k] = v[:,cell_idx]

        return out_data

    def set_active_period(self,period):
        '''
        Must be called before every (split) period
        '''
        self.cur_p_len = len(period)
        self.cur_period_idx += 1
        self.cur_chunk_idx = -1
        self.cur_chunk = NULL_CHUNK

    def terminate(self):
        #+++
        #Possible for this to happen twice and stall; hacking a fix until reconverted to PyMP vs ZMQ
        if self._terminated == False:
            self.bridge.terminate()
            self._terminated = True
