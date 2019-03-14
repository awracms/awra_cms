import multiprocessing as mp
from awrams.utils.messaging import message
from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('mp_parent')


class MultiprocessingParent(object):
    """
    Parent for classes that need to manage multiple, concurrent workers
    using Python multiprocessing
    """
    def __init__(self):
        self.control_q = mp.Queue()

        self.child_procs = {}
        self.acknowledgements = {}

    def add_child_proc(self,process,msg_q):
        '''
        Register a child process to ensure correct termination
        when it has finished working; wait on this message before terminating
        '''
        if not process.is_alive():
            process.start()

        self.child_procs[process.pid] = {'process': process, 'msg_q': msg_q}

    def terminate_children(self):
        '''
        Terminate the simulation; close any open subprocesses
        '''

        #+++
        #Input_bridge still using special case ZMQ code
        #should be converted to regular python mp process

        #self.input_bridge.terminate()

        for process in list(self.child_procs.values()):
            process['msg_q'].put(message('terminate'))

        while len(self.child_procs) > 0:
            msg = self.control_q.get()
            self.handle_child_message(msg)

    def child_exited(self,child_pid):
        if not child_pid in list(self.child_procs.keys()):
            return
            
        child = self.child_procs[child_pid]['process']
        child.join()
        self.child_procs.pop(child_pid)
        return child.exitcode

    def poll_children(self):
        '''
        Receive message(s) from child processes
        '''
        while not self.control_q.empty():
            msg = self.control_q.get()
            self.handle_child_message(msg)

    def wait_on_acknowledgements(self,message_being_acknowledged,ids,timeout=None):
        '''
        Don't return until all expected acknowledgments have returned
        '''
        finished = False
        while not finished:
            try:
                msg = self.control_q.get(timeout=timeout)
            except mp.queues.Empty:
                raise Exception('Timed out waiting for acknowledgment: %s' % message_being_acknowledged)
            self.handle_child_message(msg)
            relevant_acknowledgements = self.acknowledgements.get(message_being_acknowledged,[])
            finished = (set(relevant_acknowledgements) == set(ids))
        self.acknowledgements[message_being_acknowledged] = []

    def handle_child_message(self,m):
        subject = m['subject']
        content = m['content']
        if subject=='exception':
            self._handle_exception(content['exception'],content['traceback'])
            self.child_exited(content['pid'])
        elif subject=='terminated':
            self.child_exited(content['pid'])
        elif subject=="ack":
            self.handle_acknowledgement(content['acknowledgement'],content['id'])
        else:
            print(m)

    def handle_acknowledgement(self,acknowledgement,identifier):
        self.acknowledgements[acknowledgement] = self.acknowledgements.get(acknowledgement,[])
        self.acknowledgements[acknowledgement].append(identifier)

    def profile_results(self,total_time,measurements,pid):
        '''
        Stub profile_results handler
        '''
        pass
        #print('profile_results from %s' % pid)
