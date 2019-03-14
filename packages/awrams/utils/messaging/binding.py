import multiprocessing as mp

from awrams.utils.messaging.general import get_traceback, message
from awrams.utils.messaging.buffers import *
from awrams.utils.profiler import Profiler
from decorator import decorator, FunctionMaker
from functools import partial
import inspect
import types
#import new # deprecated in python3

class MessageHandler:
    def _handle_message(self,m):
        subj = m['subject']
        try:
            target_fn = getattr(self,subj)
        except AttributeError:
            self._handle_unknown_message(m)
        return target_fn(**(m['content']))

    def _handle_unknown_message(self,m):
        raise Exception("Unknown message", m)

class MPChild(MessageHandler,mp.Process):
    def __init__(self):
        mp.Process.__init__(self)
        #self.control_p = control_pipe
        self.pr = Profiler()

    def _handle_exception(self,e):
        m = message('child_exception', pid=self.pid, exception=e,traceback=get_traceback())
        self._send_msg(m)

    def run_setup(self):
        '''
        Perform (post-launch) initialisation
        '''
        pass

    def cleanup(self):
        '''
        Perform any pre-termination cleanup
        '''
        pass

    def run(self):
        self.pr.begin_profiling()
        self.active = True
        try:
            self.run_setup()
            while self.active:
                msg = self._recv_msg() # (['request',variable,row_idx])
                self._result = self._handle_message(msg)
                #actions = msg.get('actions')
                #if actions is not None:
                #    for action in actions:
                #        self._send_msg(message(subject=action['subject'], content=_result),target=action['target'])
                    #self._send_msg('ack',acknowledgement=ack['message'],id=ack['id'])
        except Exception as e:
            self._handle_exception(e)
            raise
        finally:
            self.cleanup()
            self.pr.end_profiling()
            self._send_msg(message('child_profile_results',pid=self.pid,profiler=self.pr))
            self._send_msg(message('child_terminated',pid=self.pid))

    def terminate(self):
        self.active = False


class PipeChild(MPChild):
    def __init__(self,control_pipe):
        MPChild.__init__(self)
        self.control_p = control_pipe

    def _recv_msg(self):
        return self.control_p.recv()

    def _send_msg(self,msg,target):
        self.control_p.send(msg)
        #self.control_p.send_bytes(msgpack.dumps(msg))

def build_queues(queue_names=None):
    if queue_names is None:
        queue_names = ['input','control']
    return dict([[k,mp.Queue()] for k in queue_names])

class QueueChild(MPChild):
    def __init__(self,pipes):
        MPChild.__init__(self)
        self.in_q = pipes['input']
        self.control_q = pipes['control']
        self.pipes = pipes

    def _poll_input(self):
        return not self.in_q.empty()

    def _recv_msg(self):
        return self.in_q.get()

    def _send_msg(self,msg,target='control'):
        self.pipes[target].put(msg)
        #self.control_q.put(msg)

def _profile_method(wrapped,self,*args, **kwargs):
    self.pr.start(wrapped.__name__)
    #return self._in(wrapped.__get__(self, cls), *args, **kwargs)
    wrapped(self,*args,**kwargs)
    self.pr.stop(wrapped.__name__)

def profile_method(wrapped):
    return decorator(_profile_method,wrapped)

def send_result(subject,target='control'):
    def _send_result(wrapped,self,*args, **kwargs):
        result = wrapped(self,*args,**kwargs)
        self._send_msg(message(subject,**result),target=target)
    return decorator(_send_result)

#def binding_apply(binder, func):
#    return FunctionMaker.create(func,'return wrapped_fn(%(signature)s)',dict(wrapped_fn=binder(func)))

def make_binding(func):
    argspec = inspect.getargspec(func)
    argnames = argspec.args[1:]
    def new_fn(self,fname,argnames,*args,**kwargs):
        a = {}
        for i in range(len(args)):
            a[argnames[i]] = args[i]
        a.update(**kwargs)
        self._send(msg=message(subject=fname,**a))
    from functools import partialmethod
    k_wrap = partialmethod(new_fn,func.__name__,argnames)
    return k_wrap

class PipeSender:
    def __init__(self,pipe):
        self._pipe = pipe

    def _send(self,msg):
        self._pipe.send(msg)

class QueueSender:
    def __init__(self,queues):
        self._queues = queues

    def _send(self,msg):
        self._queues['input'].put(msg)

    def _recv(self):
        return self._queues['control'].get()

def bound_proxy(target,io_class=QueueSender):
    class WrapperClass(io_class):
        def __init__(self,pipes):
            io_class.__init__(self,pipes)

        def terminate(self):
            self._send(message('terminate'))


    members = inspect.getmembers(target,predicate=lambda member: inspect.ismethod(member) or inspect.isfunction(member))
    for k,v in members:
        if k[0] != '_':
            #bound = binding_apply(make_binding,v)
            bound = make_binding(v)
            setattr(WrapperClass,k,bound)#types.MethodType(bound, WrapperClass))

    return WrapperClass

def wrap_as_process(in_class,p_class,mappings=None):
    if mappings is None:
        mappings = {}

    class WrappedClass(in_class,p_class):
        def __init__(self,pipes,*args,**kwargs):
            p_class.__init__(self,pipes)
            in_class.__init__(self,*args,**kwargs)

    for k,v in mappings.items():
        subject = v.get('subject')
        target = v.get('target')
        if target is None:
            target = 'control'
        setattr(WrappedClass,k,send_result(subject,target)(getattr(WrappedClass,k)))

    return WrappedClass



def instantiate_pipe_proxy(in_class,*args,**kwargs):
    Proxy = bound_proxy(in_class,PipeSender)
    Process = wrap_as_process(in_class,PipeChild)
    p0,p1 = mp.Pipe()
    process = Process(p1,*args,**kwargs)
    process.start()
    proxy = Proxy(p0)
    return proxy,process

def instantiate_q_proxy(in_class,queues=None,mappings=None,*args,**kwargs):
    if mappings is None:
        mappings = {}
    Proxy = bound_proxy(in_class,QueueSender)
    Process = wrap_as_process(in_class,QueueChild,mappings)

    if queues is None:
        q = mp.Queue()
        control_q = mp.Queue()
        queues = dict(input=q,control=control_q)
    #
    #process = Process(dict(input=q,control=control_q),*args,**kwargs)
    process = Process(queues,*args,**kwargs)
    process.start()
    #proxy = Proxy(queues['input'])
    proxy = Proxy(queues)
    return proxy,process

class MultiprocessingParent(MessageHandler):
    """
    Parent for classes that need to manage multiple, concurrent workers
    using Python multiprocessing
    """
    def __init__(self):
        self.control_q = mp.Queue()

        self.child_procs = {}
        self.acknowledgements = {}

    def child_exception(self,pid,exception,traceback):
        raise exception

    def add_child_proc(self,process,msg_q,name=None):
        '''
        Register a child process to ensure correct termination
        when it has finished working; wait on this message before terminating
        '''
        if not process.is_alive():
            process.start()

        name = str(process.pid) if name is None else name

        self.child_procs[process.pid] = {'process': process, 'msg_q': msg_q, 'name': name}

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
            self._handle_message(msg)

    def child_terminated(self,pid):
        if not pid in list(self.child_procs.keys()):
            return

        child = self.child_procs[pid]['process']
        child.join()
        self.child_procs.pop(pid)
        return child.exitcode

    def poll_children(self):
        '''
        Receive message(s) from child processes
        '''
        while not self.control_q.empty():
            msg = self.control_q.get()
            self._handle_message(msg)

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
            self._handle_message(msg)
            relevant_acknowledgements = self.acknowledgements.get(message_being_acknowledged,[])
            finished = (set(relevant_acknowledgements) == set(ids))
        self.acknowledgements[message_being_acknowledged] = []

    def handle_acknowledgement(self,acknowledgement,identifier):
        self.acknowledgements[acknowledgement] = self.acknowledgements.get(acknowledgement,[])
        self.acknowledgements[acknowledgement].append(identifier)

    def child_profile_results(self,pid,profiler):
        '''
        Stub profile_results handler
        '''
        pass
        #print('profile_results from %s' % pid)
        #print(profiler.repr_stats())
