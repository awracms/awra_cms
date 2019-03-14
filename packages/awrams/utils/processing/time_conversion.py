import multiprocessing as mp

try:
    mp.set_start_method('forkserver')

    mp.set_forkserver_preload(['numpy','pandas','WIP.robust','Support.Interaction.datetools','netCDF4'])

except:
    pass

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('time_conversion')

def resample_data(in_path,in_pattern,variable,period,out_path,to_freq,method,mode='w',enforce_mask=True,extent=None,use_weights=False):
    '''
    method is 'sum' or 'mean'
    if no extent is supplied then the full (unmasked) input will be used
    'use_weights' should be set for unequally binned conversions (monthly->annual means, for example)
    '''
    from glob import glob
    import time
    import numpy as np

    from awrams.utils.messaging import reader as nr
    from awrams.utils.messaging import writer as nw
    from awrams.utils.messaging.brokers import OrderedFanInChunkBroker, FanOutChunkBroker
    from awrams.utils.messaging.general import message
    from awrams.utils.messaging.buffers import create_managed_buffers
    from awrams.utils.processing.chunk_resampler import ChunkedTimeResampler
    from awrams.utils.extents import subdivide_extent
    from awrams.utils import datetools as dt
    from awrams.utils import mapping_types as mt
    from awrams.utils.io import data_mapping as dm

    start = time.time()

    NWORKERS = 2
    read_ahead = 3
    writemax = 3
    BLOCKSIZE = 128
    nbuffers = (NWORKERS*2)+read_ahead+writemax

    # Receives all messages from clients

    '''
    Build the 'standard queues'
    This should be wrapped up somewhere else for 
    various topologies...
    '''

    control_master = mp.Queue()

    worker_q = mp.Queue()
    for i in range(NWORKERS):
        worker_q.put(i)

    #Reader Queues
    chunk_out_r = mp.Queue(read_ahead)
    reader_in = dict(control=mp.Queue())
    reader_out = dict(control=control_master,chunks=chunk_out_r)

    #Writer Queues
    chunk_in_w = mp.Queue(writemax)
    writer_in = dict(control=mp.Queue(),chunks=chunk_in_w)
    writer_out = dict(control=control_master)

    #FanIn queues
    fanout_in = dict(control=mp.Queue(),chunks=chunk_out_r,workers=worker_q)
    fanout_out = dict(control=control_master)

    fanin_in = dict(control=mp.Queue())
    fanin_out = dict(control=control_master,out=chunk_in_w,workers=worker_q)
    
    #Worker Queues
    work_inq = []
    work_outq= []

    for i in range(NWORKERS):
        work_inq.append(mp.Queue())
        fanout_out[i] = work_inq[-1]

        work_outq.append(mp.Queue())
        fanin_in[i] = work_outq[-1]

    '''
    End standard queues...
    '''

    infiles = glob(in_path+'/'+in_pattern)
    if len(infiles) > 1:
        ff = dm.filter_years(period)
    else:
        ff = None

    sfm = dm.SplitFileManager.open_existing(in_path,in_pattern,variable,ff=ff)
    in_freq = sfm.get_frequency()

    split_periods = [period]
    if hasattr(in_freq,'freqstr'):
        if in_freq.freqstr == 'D':
            #Force splitting so that flat files don't end up getting loaded entirely into memory!
            #Also a bit of a hack to deal with PeriodIndex/DTI issues...
            split_periods = dt.split_period(dt.resample_dti(period,'d',as_period=False),'a')

    in_periods = [dt.resample_dti(p,in_freq) for p in split_periods]
    in_pmap = sfm.get_period_map_multi(in_periods)

    out_periods = []
    for p in in_periods:
        out_periods.append(dt.resample_dti(p,to_freq))

    if extent is None:
        extent = sfm.ref_ds.get_extent(True)
        if extent.mask.size == 1:
            extent.mask = (np.ones(extent.shape)*extent.mask).astype(np.bool)

    sub_extents = subdivide_extent(extent,BLOCKSIZE)
    chunks = [nr.Chunk(*s.indices) for s in sub_extents]

    out_period = dt.resample_dti(period,to_freq)
    out_cs = mt.gen_coordset(out_period,extent)

    v = mt.Variable.from_ncvar(sfm.ref_ds.awra_var)
    in_dtype = sfm.ref_ds.awra_var.dtype

    sfm.close_all()

    use_weights = False

    if method == 'mean':
        if dt.validate_timeframe(in_freq) == 'MONTHLY':
            use_weights = True

    '''
    Need a way of formalising multiple buffer pools for different classes of
    work..
    '''

    max_inplen = max([len(p) for p in in_periods])
    bufshape = (max_inplen,BLOCKSIZE,BLOCKSIZE)

    shared_buffers = {}
    shared_buffers['main'] = create_managed_buffers(nbuffers,bufshape,build=False)

    mvar = mt.MappedVariable(v,out_cs,in_dtype)
    sfm = dm.FlatFileManager(out_path,mvar)

    CLOBBER = mode=='w'

    sfm.create_files(False,CLOBBER,chunksize=(1,BLOCKSIZE,BLOCKSIZE))

    outfile_maps = {v.name:dict(nc_var=v.name,period_map=sfm.get_period_map_multi(out_periods))}
    infile_maps = {v.name:dict(nc_var=v.name,period_map=in_pmap)}

    reader = nr.StreamingReader(reader_in,reader_out,shared_buffers,infile_maps,chunks,in_periods)
    writer = nw.MultifileChunkWriter(writer_in,writer_out,shared_buffers,outfile_maps,sub_extents,out_periods,enforce_mask=enforce_mask)

    fanout = FanOutChunkBroker(fanout_in,fanout_out)
    fanin = OrderedFanInChunkBroker(fanin_in,fanin_out,NWORKERS,len(chunks))

    fanout.start()
    fanin.start()

    workers = []
    w_control = []
    for i in range(NWORKERS):
        w_in = dict(control=mp.Queue(),chunks=work_inq[i])
        w_out = dict(control=control_master,chunks=work_outq[i])
        w = ChunkedTimeResampler(w_in,w_out,shared_buffers,sub_extents,in_periods,to_freq,method,enforce_mask=enforce_mask,use_weights=use_weights)
        workers.append(w)
        w_control.append(w_in['control'])
        w.start()

    writer.start()
    reader.start()

    writer.join()

    fanout_in['control'].put(message('terminate'))
    fanin_in['control'].put(message('terminate'))

    for i in range(NWORKERS):
        w_control[i].put(message('terminate'))

    for x in range(4):
        control_master.get()

    for i in range(NWORKERS):
        workers[i].join()
        control_master.get()

    reader.join()
    fanout.join()
    fanin.join()

    end = time.time()
    logger.info("elapsed time: %ss", end-start)
