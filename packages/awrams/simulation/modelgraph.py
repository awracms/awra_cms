'''
Process-based runner that runs an executiongraph, then runs a model over the graph outputs
'''

from awrams.utils.messaging.robust import *
from awrams.utils.nodegraph import graph, nodes
from awrams.utils import mapping_types as mt

class ModelGraphRunner(PollingChild,SharedMemClient):
    def __init__(self,inq,outq,buffers,chunks,periods,mapping,model):
        PollingChild.__init__(self,inq,outq)
        SharedMemClient.__init__(self,buffers)

        self.chunks = chunks
        self.periods = periods

        self.mapping = mapping
        self.state_keys = model.get_state_keys()
        self.model = model

        self.recycling = False
        self.daemon = True

    def run_setup(self):
        #print("worker pid: %d"%self.pid,flush=True)
        self.rebuild_buffers()
        self.exe_graph = graph.ExecutionGraph(self.mapping)
        self.cur_chunk = None
        self.out_chunk = None
        self.runner = self.model.get_runner(self.exe_graph.get_dataspecs(True),shared=True)

    def recv_chunk(self):
        msg = self.recv('chunks')
        if msg is not None:
            self.cur_chunk = msg

    def send_out_chunk(self):
        if self.send('chunks',self.out_chunk):
            self.cur_chunk = None
            self.out_chunk = None

    def process(self):
        if self.cur_chunk is None:
            self.recv_chunk()
        else:
            if self.out_chunk is None:
                self.handle_current_chunk()
            else:
                self.send_out_chunk()

    def handle_current_chunk(self):
        '''
        Send back any buffers we've received, and generate some output
        Obligatory for ChunkedProcessors
        '''

        c = self.cur_chunk['content']
        period_idx = c['period_idx']
        chunk_idx = c['chunk_idx']
        buf = c['buffer']
        dims = c['dims']
        valid = c['valid_keys']

        bgroup = self.map_buffer(buf)
        bdata = bgroup.map_dims(dims)

        # Basically need to ensure that we're getting states from the sender rather
        # than trying to generate them locally...
        if period_idx > 0:
            if not self.recycling:
                for k in self.state_keys:
                    init_k = 'init_' + k
                    if init_k in self.exe_graph.process_graph:
                        del self.exe_graph.process_graph[init_k]
                    self.exe_graph.input_graph[init_k] = dict(exe=nodes.ConstNode(None))
                self.recycling = True

        for k in valid:
            self.exe_graph.input_graph[k]['exe'].value = bdata[k]       

        extent = self.chunks[chunk_idx]
        period = self.periods[period_idx]
        coords = mt.gen_coordset(period,extent)
        graph_results = self.exe_graph.get_data_flat(coords,extent.mask)

        state_buf_id, state_buf = self.get_buffer_safe('states')
        output_buf_id, output_buf = self.get_buffer_safe('outputs')

        #Run the model!
        #data = self.process_data(in_data,period_idx,chunk_idx)
        model_results = self.runner.run_over_dimensions(graph_results,dims)

        self.reclaim_buffer(buf)

        states_np = state_buf.map_dims(dims)

        for k in self.state_keys:
            states_np[k][...] = model_results['final_states'][k]

        state_msg = message('states')
        c = state_msg['content']
        c['chunk_idx'] = chunk_idx
        c['period_idx'] = period_idx
        c['buffer'] = state_buf_id

        #self.send('state_return',state_msg)
        self.qout['state_return'].put(state_msg)


        output_np = output_buf.map_dims(dims)

        for k in model_results.keys():
            if k != 'final_states':
                output_np[k][...] = model_results[k]

        self.out_chunk = message('chunk')
        c = self.out_chunk['content']
        c['period_idx'] = period_idx
        c['chunk_idx'] = chunk_idx
        c['dims'] = dims
        c['buffer'] = output_buf_id

        self.send_out_chunk()

