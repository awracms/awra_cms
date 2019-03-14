import numpy as np
from awrams.utils.config_manager import get_system_profile

CHUNKSIZE = get_system_profile().get_settings()['IO_SETTINGS']['DEFAULT_CHUNKSIZE']

class InputReader:
    def __init__(self, variable, nlons=886):
        self.ncd_time_slices = None
        self.variable = variable
        self.row = None
        self.nlons=886

    def __getitem__(self, spatial_slice):
        return np.ma.concatenate(
                        [nct[0].variables[self.variable][nct[1],spatial_slice[0],spatial_slice[1]]
                        for nct in self.ncd_time_slices])

class InputChunkReader(InputReader):
    # +++ Not referenced?
    #from memory_tester import memory_usage
    def __getitem__(self, cell):
        if self.row is None:
            self.row_chunk = int(cell[1] / CHUNKSIZE[2])
            self.read_chunks((cell[0], self.row_chunk)) #int(cell[1] / CHUNKSIZE[2]) * CHUNKSIZE[2]))
            self.row = cell[0]

        if cell[0] != self.row or self.row_chunk != int(cell[1] / CHUNKSIZE[2]):
            self.row_chunk = int(cell[1] / CHUNKSIZE[2])
            self.read_chunks((cell[0], self.row_chunk)) #, int(cell[1] / CHUNKSIZE[2]) * CHUNKSIZE[2]))
            self.row = cell[0]

        return self.chunk[:, cell[1] % CHUNKSIZE[2]]

    #@memory_usage
    def read_chunks(self, _chunk_idx):
        slice_start = _chunk_idx[1] * CHUNKSIZE[2]
        slice_end = (_chunk_idx[1] + 1) * CHUNKSIZE[2] > self.nlons \
                    and self.nlons \
                    or (_chunk_idx[1] + 1) * CHUNKSIZE[2]
        self.chunk = np.ma.concatenate(
                    [nct[0].variables[self.variable][nct[1], _chunk_idx[0], slice_start : slice_end]
                    for nct in self.ncd_time_slices])
        #self.chunk = np.ma.concatenate(
        #            [nct[0][self.variable][nct[1], _chunk_idx[0], slice_start : slice_end]
        #            for nct in self.ncd_time_slices])

