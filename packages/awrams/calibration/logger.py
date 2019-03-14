import h5py
import pandas as pd
import numpy as np
import time
import sys
from multiprocessing import Process, Queue

from awrams.utils.parameters import dataframe_to_h5

class CalibrationLoggerProcess(Process):
    def __init__(self,parameters,local_ids,filename,lschema,gschema,param_df,opt_meta=None):
        super().__init__()
        self.msg_in = Queue()
        self.parameters = parameters
        self.local_ids = local_ids
        self.filename = filename
        self.lschema = lschema
        self.gschema = gschema
        self.param_df = param_df
        self.opt_meta = opt_meta

    def run(self):
        self.logger = CalibrationLogger(self.parameters,self.local_ids,self.filename,self.lschema,self.gschema,self.param_df,self.opt_meta)
        self.logger.run_setup()

        print("Logger setup complete")

        try:

            while True:
                msg = self.msg_in.get()

                if msg.subject == 'log_results':
                    self.logger.log_results(**msg.content)
                elif msg.subject == 'terminate':
                    print("Logger terminate received")
                    break
                else:
                    raise Exception("Unknown logger message %s" %msg)
        except KeyboardInterrupt:
            print("Logger interrupt")
            pass
        except Exception as e:
            print('Exception in logger')
            print(e)
        finally:
            print("Logger finally")
            sys.stdout.flush()
            self.logger.close()

class CalibrationLogger:
    def __init__(self,parameters,local_ids,filename,lschema,gschema,param_df,opt_meta = None):
        self.local_ids = local_ids
        self.lschema = lschema
        self.gschema = gschema
        self.filename = filename
        self.parameters = parameters
        self.param_df = param_df
        self.opt_meta = opt_meta

        
        self.plen = len(self.parameters)
        self.llen = len(self.local_ids)

    def run_setup(self):
        self.fh = fh = h5py.File(self.filename,'w')
        
        self.fh.attrs['iterations'] = 0
        
        p_dim = fh.create_dataset('parameter_name',shape=(len(self.parameters),),dtype=h5py.special_dtype(vlen=str))
        for i, p in enumerate(self.parameters):
            p_dim[i] = p
        
        p_ds = fh.create_dataset('parameter_values',shape=(0,len(self.parameters)),maxshape=(None,len(self.parameters)),dtype='f8')
        p_ds.dims.create_scale(p_dim,'parameter')
        p_ds.dims[0].label = 'iteration'
        p_ds.dims[1].attach_scale(p_dim)

        #gs_ds = fh.create_dataset('global_score',shape=(0,),maxshape=(None,),dtype='f8')
        #gs_ds.dims[0].label = 'iteration'

        g_iparams = fh.create_group('initial_parameters')

        dataframe_to_h5(self.param_df,g_iparams)

        g_results = fh.create_group('global_scores')
        for name in self.gschema:
            ds = g_results.create_dataset(name,shape=(0,),maxshape=(None,),dtype='f8')
            ds.dims[0].label = 'iteration'
    
        l_results = fh.create_group('local_scores')
        
        l_id = l_results.create_dataset('local_id',shape=(len(self.local_ids),),dtype=h5py.special_dtype(vlen=str))
        for i, l in enumerate(self.local_ids):
            l_id[i] = l
        
        for name in self.lschema:
            ds = l_results.create_dataset(name,shape=(0,len(self.local_ids)),maxshape=(None,len(self.local_ids)),dtype='f8')
            ds.dims.create_scale(l_id,'local_id')
            ds.dims[0].label = 'iteration'
            ds.dims[1].attach_scale(l_id)  
            ds.dims[1].label = 'local_id'

        tid_ds = fh.create_dataset('task_id',shape=(0,),maxshape=(None,),dtype=h5py.special_dtype(vlen=str))
        tid_ds.dims[0].label = 'iteration'

        task_meta = fh.create_group('task_meta')

        if self.opt_meta is not None:
            print(self.opt_meta)
            for k,t in self.opt_meta:
                tm_ds = task_meta.create_dataset(k,shape=(0,),maxshape=(None,),dtype=get_h5dtype(t))
                ds.dims[0].label = 'iteration'

        fh.flush()


    def log_results(self,parameters,global_scores,local_scores,task_id,meta=None):
        #Parameters must support pandas series indexing 
        #ie parameters[param_names] for correct dict sorting
        self.fh.attrs['iterations'] += 1
        i_size = self.fh.attrs['iterations']
        i_idx = i_size-1

        self.fh['parameter_values'].resize((i_size,self.plen))  
        self.fh['parameter_values'][i_idx,:] = parameters
        
        #self.fh['global_score'].resize((i_size,))
        #self.fh['global_score'][i_idx] = global_score

        for name in self.gschema:
            ds = self.fh['global_scores'][name]
            ds.resize((i_size,))
            ds[i_idx] = global_scores[name]
        
        for name in self.lschema:
            ds = self.fh['local_scores'][name]
            ds.resize((i_size,self.llen))
            ds[i_idx,:] = local_scores[name]

        self.fh['task_id'].resize((i_size,))
        self.fh['task_id'][i_idx] = task_id

        if self.opt_meta is not None:
            for i,k in enumerate(self.opt_meta):
                ds = self.fh['task_meta'][k[0]]
                ds.resize((i_size,))
                ds[i_idx] = meta[i]

        self.fh.flush()

    def close(self):
        print("Logger file close")
        self.fh.close()

def get_h5dtype(python_type):
    tdict = {
        str: h5py.special_dtype(vlen=str),
        int: 'i8',
        float: 'f8'
    }

    h5t = tdict.get(python_type)
    if h5t is None:
        h5t = python_type

    return h5t

if __name__ == '__main__':
    parameters = ['p' + str(x) for x in range(23)]
    local_ids = ['catch_' + str(x) for x in range(300)]
    filename = 'cal_log.h5'
    lschema = ['qtot_nse']
    logger = CalibrationLogger(parameters,local_ids,filename,lschema)
    logger.run_setup()

    # parameters : flat
    # global_score : scalar (refactor to flat_multi)
    # local_scores : dict of flat

    s = time.time()

    for x in range(1000):

        params = np.random.normal(size=len(parameters))
        global_score = np.random.normal()
        local_scores = dict(qtot_nse=np.random.normal(size=(len(local_ids))))

        logger.log_results(params,global_score,local_scores)

    logger.close()

    e = time.time()

    print(e-s)