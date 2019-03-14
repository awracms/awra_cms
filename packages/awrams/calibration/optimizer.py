import multiprocessing as mp
from awrams.utils.messaging import Message
from awrams.utils.nodegraph.nodes import funcspec_to_callable
from collections import OrderedDict
import numpy as np
import threading
import time
import sys
from queue import Empty, Queue

class OptimizerProcess(mp.Process):
    def __init__(self,pspace,opt_spec,report_freq=5):
        super().__init__()
        self.pspace = pspace
        self.opt_spec = opt_spec
        self.finished = mp.Event()
        self.task_out = mp.Queue()
        self.obj_in = mp.Queue()
        self.control_in = mp.Queue()
        self.report_freq = report_freq

    def run(self):
        import os
        all_cores = set(range(mp.cpu_count()))

        # No sched_setaffinity in Windows, we handle this in MSMPI
        if os.name is not 'nt':
            os.sched_setaffinity(0,all_cores)

        def get_max_evaluators():
            optimizer = self.opt_spec.opt_cls(self.pspace,None,**self.opt_spec.opt_args)
            return optimizer.get_max_evaluators()

        eval_f = RemoteEvaluatorFactory(self.task_out,self.obj_in,get_max_evaluators())

        queues = dict(control_to_opt=Queue(),opt_to_control=Queue())

        def run_opt(opt_class,opt_args,queues):
            opt = opt_class(self.pspace,eval_f,**opt_args)
            opt.set_threaded(queues)
            opt.run()

        tt = threading.Thread(target=run_opt,args=[self.opt_spec.opt_cls,self.opt_spec.opt_args,queues])
        tt.start()

        status = Message('launching')

        new_status = True

        prev_time = time.time()

        while status.subject != 'complete':
            time_now = time.time()
            if time_now - prev_time > self.report_freq:
                if new_status:
                    print('\n',status)
                    new_status = False
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                prev_time = time_now
            time.sleep(0.1)
            queues['control_to_opt'].put('request_update')
            try:
                status = queues['opt_to_control'].get_nowait()
                new_status = True
            except Empty:
                pass
            try:
                msg = self.control_in.get_nowait()
                if msg.subject == 'terminate':
                    status.subject = 'complete'
            except Empty:
                pass
        print(status)

        self.finished.set()

class Optimizer:
    def __init__(self):
        self._threaded = False
        self.status = Message("Launching")
        
    def check_status(self):
        if self._threaded:
            try:
                msg = self.queues['control_to_opt'].get_nowait()
            except Empty:
                return
            if msg == 'request_update':
                self.update_status()
            elif msg == 'terminate':
                raise Exception("Early Termination Requested")
        else:
            self.update_status()
            
    def set_threaded(self,queues):
        self.queues = queues
        self._threaded = True
        self.update_status = self._update_status_threaded
            
    def _update_status_threaded(self):
        self.queues['opt_to_control'].put(self.status)
        
    def update_status(self):
        print(self.status)

    def get_max_evaluators(self):
        '''
        Return the maximum number of evaluators that may be spawned by this optimizer
        In general this will be the number of parallel evaluators in a shuffled or compartmentalized optimizer,
        plus 1 if there is evalution of the initial population
        Serial optimizers need only 1 (the default value)
        '''
        return 1


def recv_message_t(inq,out_queues):
    while(True):
        msg = inq.get()
        out_queues[msg['source_id']].put(msg)

class RemoteEvaluator:
    def __init__(self,inq,outq,identifier):
        self.inq = inq
        self.outq = outq
        self.identifier = identifier
        self.n_eval = 0
 
    def evaluate(self,params,meta=None):
        task_id = "%s_%s" % (self.identifier,self.n_eval)
        self.outq.put(dict(source_id=self.identifier,task_id=task_id,params=params,meta=meta))
        self.n_eval += 1
        result = self.inq.get()
        return result['objf_val']
    
    def evaluate_population(self,paramsets,meta=None):
        results = OrderedDict()
        for i,p in enumerate(paramsets):
            task_id = "%s_%s" % (self.identifier,self.n_eval)
            results[task_id] = None
            if meta is not None:
                tmeta = meta[i]
            else:
                tmeta = None
            self.outq.put(dict(source_id=self.identifier,task_id=task_id,params=p,meta=tmeta))
            self.n_eval += 1
        for p in paramsets:
            res = self.inq.get()
            results[res['task_id']] = res['objf_val']
        return np.array(list(results.values()))

class RemoteEvaluatorFactory:
    def __init__(self,task_out_q,objf_in_q,max_nworkers):
        self.task_out_q = task_out_q
        self.objf_in_q = objf_in_q
        self.cur_id = 0
        self.max_nworkers = max_nworkers
        self.worker_queues = [Queue() for i in range(max_nworkers)]

    def new(self):
        if self.cur_id == self.max_nworkers:
            raise Exception("Maximum evaluators instantiated")
        if self.cur_id == 0:
            self._rthread = threading.Thread(target=recv_message_t,args=[self.objf_in_q,self.worker_queues])
            self._rthread.start()
        evaluator = RemoteEvaluator(self.worker_queues[self.cur_id],self.task_out_q,self.cur_id)
        self.cur_id += 1
        return evaluator


