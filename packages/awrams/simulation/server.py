import time
from awrams.models.model import Model
from .clustered import build_sim_pickle, launch_sim_from_pickle
import shutil
import os
from awrams.utils import config_manager
from awrams.cluster.support import build_mpi_call_str, build_pbs_header, \
                                   build_full_pbs_file, get_pbs_header_options,\
                                   RemoteJobSpec

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('server')


class SimulationServer:
    def __init__(self,model,sys_settings = None):
        
        if not isinstance(model,Model):
            raise TypeError("model must be of type awrams.models.model.Model")
        ### defaults
        
        if sys_settings is None:
            sys_settings = config_manager.get_system_profile().get_settings()

        self.sys_settings = sys_settings

        sim_settings = sys_settings['SIMULATION']

        self.spatial_chunk = sim_settings['SPATIAL_CHUNK']
        self.time_chunk = sim_settings['TIME_CHUNK']
        self.min_cells_per_worker = sim_settings['MIN_CELLS_PER_WORKER']
        self.num_workers = os.cpu_count()
        self.task_buffers = sim_settings['TASK_BUFFERS']

        # This value is deprecated as of AWRACMS 1.2, but kept for old scripts that may attempt
        # to set it.  Setting read_ahead will have no effect, and will be removed in later versions
        self.read_ahead = 1

        self.model = model

        self.logger = logger


    def run(self,input_map,output_map,period,extent,clobber=False): #periods,chunks):
        '''
        Should be the basis for new-style sim server
        Currently no file output, but runs inputgraph/model quite happily...
        '''
        
        start = time.time()

        #For very small extents, we might not be able to distribute enough
        #work with our current settings; reduce num workers to account for this
        expected_min = self.num_workers * self.min_cells_per_worker
        actual_cell_count = extent.cell_count
        
        if extent.cell_count < expected_min:
            self.num_workers = extent.cell_count // self.min_cells_per_worker
        if self.num_workers == 0:
            self.num_workers = 1
            self.min_cells_per_worker = extent.cell_count

        self.logger.info("Building simulation specification file")

        build_sim_pickle(self.model,input_map,output_map,period,extent, \
                         self.num_workers,'./server_sim.pkl',self.spatial_chunk, \
                         self.time_chunk, self.min_cells_per_worker)

        if clobber:
            cleaned_paths = []

            for k,v in output_map.items():
                if 'io' in v.properties and v.properties['io'] == 'w':
                    if 'path' in v.args:
                        cur_path = v.args['path']
                        if cur_path not in cleaned_paths:
                            self.logger.info("Removing files in %s" % cur_path)
                            shutil.rmtree(cur_path,True)
                            cleaned_paths.append(cur_path)

        self.logger.info("Running simulation")

        self.model.rebuild_for_input_map(input_map, force=False)

        launch_sim_from_pickle('./server_sim.pkl')

        self.logger.info("elapsed time: %.2f",time.time() - start)

    def get_remote_job(self, job_name, remote_path, node_count, walltime, \
                       input_map, output_map, period, extent, clobber=False):

        pickle_filename = '%s.pkl' %job_name
        pbs_filename = '%s.pbs' % job_name

        cores_per_node = self.sys_settings['REMOTE_SETTINGS']['PBS_SETTINGS']['CORES_PER_NODE']
    
        sim_settings = self.sys_settings['SIMULATION']

        spatial_chunk = sim_settings['SPATIAL_CHUNK']
        time_chunk = sim_settings['TIME_CHUNK']
        min_cells_per_worker = sim_settings['MIN_CELLS_PER_WORKER']
        num_workers = cores_per_node * node_count
        task_buffers = sim_settings['TASK_BUFFERS']

        sim_spec = build_sim_pickle(self.model, input_map, output_map, period, extent, \
                                    num_workers, pickle_filename ,spatial_chunk, \
                                    time_chunk, min_cells_per_worker)

        n_workers = len(sim_spec.node_map)

        task_specs = ['--mca mpi_warn_on_fork 0 -n {n_workers} python3 -m \
                      awrams.cluster.mpi_node_entry {pickle_filename}']

        common_opts = self.sys_settings['MPI_COMMON_OPTIONS']
        format_opts = dict(n_workers = n_workers, pickle_filename = pickle_filename)
        
        task_call_str = build_mpi_call_str(task_specs, common_opts, format_opts)
        
        pbs_header = build_pbs_header(**get_pbs_header_options(self.sys_settings,job_name,walltime,node_count))
        activation = self.sys_settings['REMOTE_SETTINGS']['ACTIVATION']

        if clobber:
            #+++
            logger.warning("Clobber not yet supported for remote jobs")
        
        build_full_pbs_file(pbs_header, activation, remote_path, task_call_str, pbs_filename)

        return RemoteJobSpec(remote_path, pbs_filename, [pickle_filename])
    