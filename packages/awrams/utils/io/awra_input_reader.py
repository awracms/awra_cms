"""
Standalone input reader module
Communicates with the rest of the system via ZMQ
"""

from io import StringIO
import zmq
import numpy as np
import sys
import traceback
import os
import pickle
import multiprocessing as mp
import awrams.utils.datetools as dt
import awrams.utils.extents as extents

from awrams.utils.messaging.general import *
from awrams.utils.profiler import Profiler
from awrams.utils.io.db_helper import _nc as db_opener
# from awrams.utils.io.db_helper import _h5py as db_opener

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('input_reader')

def to_chunk_idx(cell,c_shape):
    return (cell[0]/c_shape[0],cell[1]/c_shape[1])

def offset_slice(in_slice,offset):
    return slice(in_slice.start+offset,in_slice.stop+offset)

def build_chunk_map(ref_extent,subset_extent,c_shape):
    '''
    Returns a pair of chunk maps in the coordinates of both the reference and subset extents
    Assumes just 2d (spatial) chunking; ignore time
    '''
    #c_shape = (1,50)
    d_shape = ref_extent.shape #(681,841)

    l_extent = subset_extent.translate_to_origin(ref_extent)

    offset = subset_extent.x_min - ref_extent.x_min, subset_extent.y_min - ref_extent.y_min

    def from_chunk_idx(x,y):
        if c_shape[0] == 1:
            x_idx = x
        else:
            raise Exception("Unsupported chunk layout")

        y_start = y * c_shape[1]
        y_end = (y+1) * c_shape[1]
        if y_end > d_shape[1]:
            y_end = d_shape[1]

        return Chunk(x_idx,slice(y_start,y_end))

    n_x_chunks = int(np.ceil(d_shape[0]/float(c_shape[0])))
    n_y_chunks = int(np.ceil(d_shape[1]/float(c_shape[1])))

    # This is the 'file global' mask
    chunk_mask = np.zeros(shape=(n_x_chunks,n_y_chunks),dtype=bool)

    for cell in l_extent:
        chunk_idx = to_chunk_idx(cell,c_shape)
        chunk_mask[chunk_idx] = True

    global_chunks = []
    subset_chunks = []

    for x in range(n_x_chunks):
        for y in range(n_y_chunks):
            if chunk_mask[x,y]:
                chunk = from_chunk_idx(x,y)
                global_chunks.append(chunk)
                subset_chunks.append(Chunk(chunk.x - offset[0],offset_slice(chunk.y,-offset[1])))

    return global_chunks, subset_chunks


class StreamingReaderProcess(mp.Process):
    '''
    Wrapper to use in python multiprocessing mode
    '''
    def __init__(self,control_p,req_p):
        mp.Process.__init__(self)
        self.control_p = control_p
        self.req_p = req_p

    def run(self):
        reader = StreamingReader(self.control_p,self.req_p)
        self.reader.run()

class StreamingReader:
    '''
    Provides a ZMQ interface to streaming split file NetCDF data
    '''

    def __init__(self,control_p,req_p,read_ahead = 3):
        ctx = zmq.Context()

        self.control_s = ctx.socket(zmq.PUSH)
        self.control_s.connect(control_p)

        self.req_s = ctx.socket(zmq.PAIR)        
        self.req_s.bind(req_p)
        self.client_logger = configure_logging_to_zmq_client(self.req_s)
        self.pid = os.getpid()

        self.cur_ds = {}
        self.variables = []
        self.cur_data = {}

        self.read_ahead = read_ahead

        self.data_extent = extents.default()

    def _handle_exception(self,e):
        #sio = StringIO()
        #traceback.print_exc(file=sio)
        m = message('exception', pid=self.pid, exception=e,traceback=get_traceback())
        
        #+++
        #need sensible error handling for request queues
        
        self.req_s.send_pyobj(m)
        #self.control_s.send_pyobj(m)

    def set_inputs(self, file_maps):
        self.file_maps = file_maps
        self.variables = []

        for v in list(self.file_maps.keys()):
            self.variables.append(v)
            self.cur_ds[v] = None

        ref_ds = db_opener(list(self.file_maps[v].values())[0],'r')
        lats = ref_ds.variables['latitude']
        lons = ref_ds.variables['longitude']

        #+++
        # Only here since climate data still has wider extents than our mask grid

        if len(lats) > 681 or len(lons) > 841:
            self.data_extent = extents.default()
        else:
            self.data_extent = extents.from_boundary_coords(lats[0],lons[0],lats[-1],lons[-1])

        ref_ds.close()

        self.req_s.send_pyobj(message('OK'))

    def terminate(self):
        self.close_all()
        self.active = False

    def clear_req_map(self):
        self.req_map = {}
        for v in self.variables:
            self.req_map[v] = False

    def active_variables(self):
        out_vars = []
        for v in self.variables:
            if self.cur_ds[v] != None:
                out_vars.append(v)
        return out_vars

    def all_reqs_sent(self):
        for v in self.variables:
            if self.cur_ds[v] != None:
                if self.req_map[v] == False:
                    return False
        return True

    def close_all(self):
        '''
        Close all open datasets
        '''
        for v in self.variables:
            if self.cur_ds[v] != None:
                self.cur_ds[v].close()
                self.cur_ds[v] = None

    def notify(self,m):
        self.control_s.send_pyobj(m)

    def handle_message(self,m):

        msg = m['subject']
        m = m['content']
        if msg == 'terminate':
            self.terminate()
        elif msg == 'request_chunks':
            self.handle_request(m['chunk_idx'],m['period_idx'])
        elif msg == 'set_periods':
            self.set_periods(m['periods'])
        elif msg == 'set_extent':
            self.set_extent(m['extent'],m['mode'])
        elif msg == 'set_inputs':
            self.set_inputs(m['file_maps'])
        else:
            raise Exception("Unknown message", msg)

    def set_extent(self,extent,mode):
        self._build_chunk_map(extent,mode)

    def _build_chunk_map(self,target_extent,mode):
        #+++ hardcoding for now
        c_shape = (1,50)

        self.chunk_map, t_chunks = build_chunk_map(self.data_extent,target_extent,c_shape)

        if mode == 'global':
            chunks = self.chunk_map
        elif mode == 'local':
            chunks = t_chunks

        m = message('chunk_map', chunks = chunks, geo_ref = self.data_extent.geospatial_reference())
        self.req_s.send_pyobj(m)

    def set_cur_period(self,period):
        '''
        Load appropriate files for current active period
        '''

        self.cur_chunk_idx = -1

        self.cur_period_len = len(period)

        start = period[0]
        end = period[-1]

        p_year = dt.start_of_year(start)

        self.cur_start_idx = (start - p_year).days
        self.cur_end_idx = (end - p_year).days + 1

        for v in self.variables:
            if self.cur_ds[v] != None:
                self.cur_ds[v].close()
                self.cur_ds[v] = None
            if start.year in self.file_maps[v]:
                self.cur_ds[v] = db_opener(self.file_maps[v][start.year],'r')

    def set_periods(self,split_periods):
        '''
        Supply a list of periods in the order that the destination processor expects
        '''
        self.cur_period_idx = 0
        self.split_periods = split_periods
        self.set_cur_period(split_periods[0])
        #self.populate_next_chunk()
        self.req_s.send_pyobj(message('OK'))

    def handle_request(self, chunk_idx, period_idx):

        if self.cur_chunk_idx == -1:
            self.populate_next_chunk()

        if self.cur_chunk_idx != chunk_idx:
            raise Exception("Mismatched chunk indices")

        if self.cur_period_idx != period_idx:
            raise Exception("Period index mismatch. Current : %s, Requested %s", self.cur_period_idx,period_idx)

        variables = self.active_variables()

        m = message('transmit_chunks',chunk_idx=chunk_idx,variables=variables)
        
        self.req_s.send_pyobj(m)

        for v in variables:
            send_array(self.req_s,self.cur_data[v])

        if self.cur_chunk_idx == (len(self.chunk_map)-1):
            if self.cur_period_idx == (len(self.split_periods)-1):
                return
            else:
                self.cur_period_idx += 1
                self.set_cur_period(self.split_periods[self.cur_period_idx])

        self.populate_next_chunk()
            

    def populate_next_chunk(self):
        #chunk_idx = self.extent_map.keys()[self.cur_emap_idx]

        self.cur_data = {}

        self.cur_chunk_idx += 1

        chunk = self.chunk_map[self.cur_chunk_idx]

        for v in self.variables:
            if self.cur_ds[v] != None:
                #self.cur_data[v] = self.cur_ds[v][v][self.cur_start_idx:(self.cur_end_idx+1),chunk.x,chunk.y]
                self.cur_data[v] = self.cur_ds[v][v][self.cur_start_idx:self.cur_end_idx,chunk.x,chunk.y]

    def run(self):

        #signal.signal(signal.SIGINT, signal.SIG_IGN)
        pr = Profiler()
        pr.begin_profiling()

        self.active = True
        
        try:

            self.clear_req_map()

            while self.active:
                pr.start('waiting')
                msg = self.req_s.recv_pyobj() # (['request',variable,row_idx])
                self.handle_message(msg)
                pr.stop('waiting')

        except Exception as e:
            self._handle_exception(e)
            raise
        finally:
            pr.end_profiling()
            #self.control_s.send_pyobj({'message': 'occupancy', 'process': 'Inputs (%s)' % self.pid, 'value': pr.occupancy()})

if __name__ == "__main__":
    try:
        reader = StreamingReader(sys.argv[1],sys.argv[2])
        reader.run()
    except Exception as e:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback)
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit("Exception running input reader")
