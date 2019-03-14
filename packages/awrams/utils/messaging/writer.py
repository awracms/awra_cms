from awrams.utils.messaging.robust import PollingChild, SharedMemClient, Chunk, to_chunks

import netCDF4 as nc
#from awra import db_open_with
#+++ Need to use NCD (as opposed to h5py) when writing to LSD compressed datasets
db_opener = nc.Dataset#db_open_with()

FILL_VALUE = -999.

class MultifileChunkWriter(PollingChild,SharedMemClient):
    def __init__(self,qin,qout,buffers,file_maps,extents,periods,enforce_mask=True):
        PollingChild.__init__(self,qin,qout)
        SharedMemClient.__init__(self,buffers)

        self.file_maps = file_maps
        self.variables = list(file_maps.keys())

        self.extents = extents
        self.chunks = to_chunks(extents)
        self.periods = periods

        self.cur_f = dict([(k,None) for k in self.variables])
        self.cur_ds = dict([(k,None) for k in self.variables])
        self.time_indices = dict([(k,None) for k in self.variables])

        self.nc_vars = dict([(k,v['nc_var']) for k,v in self.file_maps.items()])

        self._file_mode = 'a'
        self.enforce_mask = enforce_mask
        
    def run_setup(self):
        self.rebuild_buffers()
        self.cur_chunk = None
        self.cur_chunk_count = 0
        self._set_cur_period(0)

    def cleanup(self):
        self.close_all()

    def close_all(self):
        '''
        Close all open datasets
        '''

        for variable in self.variables:
            if self.cur_ds[variable] is not None:
                self.cur_ds[variable].close()
                self.cur_ds[variable] = None

    def _clear_open(self,variable):
        '''
        Close any open files and clear locals relating to the specified variable
        '''
        if self.cur_ds[variable] is not None:
            self.cur_ds[variable].close()
            self.cur_ds[variable] = None
        self.cur_f[variable] = None

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

        if period_idx != self.cur_period_idx:
            self._set_cur_period(period_idx)

        chunk_idx = self.cur_chunk['chunk_idx']

        for v,data in self.cur_chunk['data'].items():
            write_var = self.cur_ds[v].variables[self.nc_vars[v]]
            shm_buf = data['buffer']
            in_data = self.map_buffer(shm_buf,data['shape'])

            pwrite_idx = self.time_indices[v]
            chunk = self.chunks[chunk_idx]

            write_idx = (pwrite_idx,chunk.x,chunk.y)

            # ENFORCE_MASK +++
            if self.enforce_mask:
                subex = self.extents[chunk_idx]
                in_data[:,subex.mask==True] = FILL_VALUE

            self._write_slice(write_var,in_data,write_idx)
            # try:
            #     # write_idx[0] is a slice
            #     write_var[write_idx] = in_data[:write_idx[0].stop-write_idx[0].start,:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
            # except AttributeError:
            #     # write_var[write_idx] = in_data[:,:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
            #     # write_var[write_idx] = in_data
            #     try:
            #         # write_idx[0] is a single int
            #         write_var[write_idx] = in_data[write_idx[0],:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
            #     except IndexError:
            #         # write_idx[0] is a list of ints (case for multi_month_deciles)
            #         write_var[write_idx] = in_data[:len(write_idx[0]),:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
            #     # write_var[write_idx] = in_data

            self.reclaim_buffer(shm_buf)

        self.cur_chunk = None
        self.cur_chunk_count += 1

        if self.cur_chunk_count == len(self.chunks):
            self.cur_chunk_count = 0

            if self.cur_period_idx < len(self.periods)-1:
                self.cur_period_idx +=1
                self._set_cur_period(self.cur_period_idx)
            else:
                self.terminate()

    def _write_slice(self,write_var,in_data,write_idx):
        try:
            write_var[write_idx] = in_data[:write_idx[0].stop-write_idx[0].start,:]
        except AttributeError:
            write_var[write_idx] = in_data

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

        self.cur_period_idx = pidx

class MultifileChunkWriterSpecial(MultifileChunkWriter):
    def _write_slice(self,write_var,in_data,write_idx):
        try:
            # write_idx[0] is a slice
            write_var[write_idx] = in_data[:write_idx[0].stop-write_idx[0].start,:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
        except AttributeError:
            # write_var[write_idx] = in_data[:,:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
            # write_var[write_idx] = in_data
            try:
                # write_idx[0] is a single int
                write_var[write_idx] = in_data[write_idx[0],:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
            except IndexError:
                # write_idx[0] is a list of ints (case for multi_month_deciles)
                write_var[write_idx] = in_data[:len(write_idx[0]),:write_idx[1].stop-write_idx[1].start,:write_idx[2].stop-write_idx[2].start]
            # write_var[write_idx] = in_data


