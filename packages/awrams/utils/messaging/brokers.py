from awrams.utils.messaging.robust import PollingChild
from time import sleep


class FanInChunkBroker(PollingChild):
    '''
    Multi-input, single output FIFO polling
    '''
    def __init__(self,qin,qout,nchildren):
        PollingChild.__init__(self,qin,qout)
        self.nchildren = nchildren
        
    def run_setup(self):
        self.results_msg = None
        self.curchild = -1

    def pollchildren(self):
        for i in range(self.nchildren):
            self.curchild = (self.curchild+1)%self.nchildren
            try:
                msg = self.qin[self.curchild].get_nowait()
                self.latest_from = self.curchild
                return msg
            except:
                pass

    def process(self):
        if self.results_msg is None:
            self.results_msg = self.pollchildren()
        if self.results_msg is not None:
            self.handle_results(self.latest_from)
        else:
            sleep(0.001)

    def handle_results(self,source):
        if not self.send('out',self.results_msg): ### try to send latest result
            return False
        self.send('workers',source) ### report which kid out came from
        self.results_msg = None


class OrderedFanInChunkBroker(PollingChild):
    '''
    Multi-input, single output FIFO polling,
    All chunks from a single period are enqueued
    contiguously
    '''
    def __init__(self,qin,qout,nchildren,nchunks):
        '''

        '''
        PollingChild.__init__(self,qin,qout)
        self.nchildren = nchildren
        self.chunks_per_period = nchunks
        
    def run_setup(self):
        self.results_msg = None
        self.backlog = []

        self.cur_chunk_count = 0
        self.cur_period_idx = 0

        self.curchild = -1

    def pollchildren(self):
        for i in range(self.nchildren):
            self.curchild = (self.curchild+1)%self.nchildren
            try:
                msg = self.qin[self.curchild].get_nowait()
                self.latest_from = self.curchild
                return msg
            except:
                pass

    def process(self):
        if self.results_msg is None:
            self.results_msg = self.process_backlog()
            if self.results_msg is None:
                self.results_msg = self.pollchildren()
        if self.results_msg is not None:
            self.handle_results(self.latest_from,**self.results_msg['content'])
        else:
            sleep(0.001)

    def handle_results(self,source,chunk_idx,period_idx,data):
        valid = self.validate_period(period_idx)

        if not valid:
            self.backlog.append(self.results_msg)
            self.results_msg = None
        else:
            #Could timeout queuing results; wait and try again...
            if not self.send('out',self.results_msg):
                return False
            self.send('workers',source)
            self.update_indices(period_idx,chunk_idx)
            self.results_msg = None

    def update_indices(self,period_idx,chunk_idx):
        self.cur_chunk_count += 1

        if self.cur_chunk_count == self.chunks_per_period:
            self.cur_chunk_count = 0
            self.cur_period_idx += 1

    def validate_period(self,period_idx):
        #+++ Maybe add enforce_ordering flag?
        #eg flat files don't need ordered chunks...
        return period_idx == self.cur_period_idx

    def process_backlog(self):
        for item in self.backlog:
            pidx = item['content']['period_idx']
            if self.validate_period(pidx):
                self.backlog.remove(item)
                return item
        return None

class FanOutChunkBroker(PollingChild):
    '''
    Multi-out, single input broker
    '''
    def __init__(self,qin,qout):
        PollingChild.__init__(self,qin,qout)

    def run_setup(self):
        self.cur_task = None
        self.cur_worker = None

    def process(self):
        '''
        Match incoming chunks with the next available worker
        '''
        if self.cur_task is None:
            self.cur_task = self.recv('chunks')
        if self.cur_worker is None:
            self.cur_worker = self.recv('workers')

        if self.cur_task is not None:
            if self.cur_worker is not None:
                if self.send(self.cur_worker,self.cur_task):
                    self.cur_task = None
                    self.cur_worker = None

