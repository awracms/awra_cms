from io import StringIO
import traceback
import numpy as np
#import zmq
import errno
import uuid
import logging
from awrams.utils.metatypes import ObjectDict as o
import subprocess

class Chunk:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.shape = (1,self.y.stop - self.y.start)

    def __repr__(self):
        return str((self.x,self.y))

    def contains(self,cell):
        if cell[0] == self.x:
            if cell[1] >= self.y.start and cell[1] < self.y.stop:
                return True
        return False

    def idx(self,cell):
        '''

        Warning - this doesn't check for validity, only passes back what it thinks is a local index
        '''
        return cell[1]-self.y.start

NULL_CHUNK = Chunk(-1,slice(-1,-1))

def gen_ipc_handle(prefix='awra'):
    suffix = uuid.uuid4().hex
    return "ipc:///tmp/" + prefix + '_' + suffix

def send_array(socket, A, flags=0, copy=False, track=True):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )

    socket.send_pyobj(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=False, track=True):
    """recv a numpy array"""
    md = socket.recv_pyobj(flags=flags)

    msg = socket.recv(flags=flags, copy=copy, track=track)

    buf = msg
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def message(subject,**kwargs):
    m = dict(subject=subject,content=dict())
    m['content'].update(kwargs)
    return m

def get_traceback(include_locals=False):
    import sys

    sio = StringIO()
    traceback.print_exc(file=sio)
    tb_text = sio.getvalue()

    if include_locals:
        tb = sys.exc_info()[2]
        while 1:
            if not tb.tb_next:
                break
            tb = tb.tb_next
        frame = tb.tb_frame

        tb_text += "Locals:\n"
        for key, value in list(frame.f_locals.items()):
            if key.startswith('__'):
                tb_text += '%s (skipped)\n'%(key)
                continue

            try:
                lstr = "  %s = %s" % (key,value)
            except:
                lstr = "  %s (unprintable)" % (key)
            tb_text += lstr + '\n'
    
    return tb_text


'''
Support classes and methods
'''

class ZMQLogger(logging.Handler):
    def __init__(self, socket, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        self.socket = socket

    def emit(self,record):
        '''
        Send a message to the socket
        '''
        try:
            self.socket.send_pyobj(message('log_message',content=record.getMessage(),level=record.levelno))
        except zmq.ZMQError as e:
            if e.errno == errno.EINTR:
            #+++ System interrupt, retry
                self.emit(record)
            else:
                raise

class MPLogger:
    def __init__(self,queue):
        self.queue = queue

    def write(self,msg):
        self.queue.put(message('log_message',content=msg))

    def flush(self):
        pass

class QueingLogHandler(logging.Handler):
    def __init__(self, queue, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        self.queue = queue

    def emit(self,record):
        self.queue.put(message('log_message',content=record.getMessage(),level=record.levelno))

class QueuedLogCollector(object):
    def __init__(self,queue):
        self.reset()
        self.queue = queue

    def reset(self):
        self.messages = {
            'debug': [],
            'info': [],
            'warning': [],
            'error': [],
            'critical': [],
        }

    def harvest(self):
        while not self.queue.empty():
            record = self.queue.get()
            self.messages[record.levelname.lower()].append(record.getMessage())


def configure_logging_to_zmq_client(channel):
    return _configure_client_logging(ZMQLogger(channel),True)

def configure_logging_to_mp_client(queue):
    return _configure_client_logging(QueingLogHandler(queue))

def _configure_client_logging(handler,format=False):
    from awrams.utils.settings import LOGFORMAT #pylint: disable=no-name-in-module
    import logging
    import awrams.utils.awrams_log
    logger = awrams.utils.awrams_log.establish_logging()
    #client_logger = log_writer
    #handler = logging.StreamHandler(client_logger)
    #if format:
    #    handler.setFormatter(logging.Formatter(LOGFORMAT))
    logger.addHandler(handler)
    return handler


def term(msg):
    '''
    Terminator handler for Managed listeners
    '''
    return 1

def term_print(msg):
    print(msg)
    return 1

class Message:
    def __init__(self,subject,content=None,**kwargs):
        self.subject = subject
        if content is None:
            content = dict(**kwargs)
        self.content = content
        
    def __getitem__(self,k):
        return self.content[k]
    
    def __setitem__(self,k,v):
        self.content[k] = v

    def __repr__(self):
        return self.subject + "\n" + str(self.content)