"""Cluster Support

Support functions for running build calibration jobs for clustered runs (targetting MPI/PBS)

"""

import os
import sys
import pickle
from awrams.calibration.allocation import allocate_catchments_to_nodes
from awrams.utils.nodegraph.nodes import callable_to_funcspec
from awrams.cluster.support import build_mpi_call_str, build_full_pbs_file, \
                                   build_pbs_header, get_pbs_header_options, \
                                   RemoteJobSpec


def build_pbs_from_cal_spec(cal_spec, walltime, pickle_fn, pbs_fn, max_nodes,
                            core_min=1, max_over=1.02, job_name='awrams_cal',
                            job_queue=None, project=None):
    """Build PBS and pickle files from the supplied calibration spec dict
    
    Args:
        cal_spec (dict): Calibration spec as built in example notebooks
        walltime (str): walltime for PBS script, "HH:MM:SS"
        pickle_fn (str): Output filename for generated pickle file
        pbs_fn (str): Output filename for generated PBS file
        max_nodes (int): Maximum number of (machine) nodes to allocate
        core_min (int, optional): Minimum number of cells per core
        max_over (float, optional): Scaling factor for maximum cells/node
        job_name (str, optional): PBS job name
        job_queue (str, optional): PBS job queue
        project (str, optional): PBS project
    """
    full_spec = build_pickle_from_spec(cal_spec, max_nodes, pickle_fn, core_min, max_over)

    n_nodes = full_spec['n_workers']

    build_pbs_file(n_nodes, walltime, pickle_fn, pbs_fn, job_name, job_queue, project)


def build_remote_pbs_from_cal_spec(cal_spec, walltime, pickle_fn, pbs_fn,
                                   remote_path, job_name, max_nodes, cal_config,
                                   core_min=1, max_over=1.02):
    """Build PBS and pickle files from the supplied calibration spec dict, for a remote run
    
    Args:
        cal_spec (dict): Calibration spec as built in example notebooks
        walltime (str): walltime for PBS script, "HH:MM:SS"
        pickle_fn (str): Output filename for generated pickle file
        pbs_fn (str): Output filename for generated PBS file
        remote_path (str): Path for the remote pickle and pbs files
        job_name (str, optional): PBS job name
        max_nodes (int): Maximum number of (machine) nodes to allocate
        cal_config (dict): Calibration config dict (from ./awrams/<PROFILE>/calibration.py)
        core_min (int, optional): Minimum number of cells per core
        max_over (float, optional): Scaling factor for maximum cells/node
    """
    full_spec = build_pickle_from_spec(cal_spec, cal_config.CORES_PER_NODE, max_nodes, 
                                       pickle_fn, core_min, max_over)
    
    n_nodes = full_spec['n_workers']

    remote_pickle_fn = os.path.join(remote_path, pickle_fn)

    build_pbs_file(n_nodes, walltime, remote_pickle_fn,
                   pbs_fn, job_name, cal_config=cal_config)


def build_pickle_from_spec(cal_spec, cores_per_node, max_nodes, pickle_fn, 
                           core_min=1, max_over=1.02):
    """Build a fully realised pickle file from the supplied cal_spec
    
    Args:
        cal_spec (dict): Calibration spec as built in example notebooks
        cores_per_node (int): Number of cores to use on each (machine) node
        max_nodes (int): Maximum number of (machine) nodes to allocate
        pickle_fn (str): Output filename for generated pickle file
        core_min (int, optional): Minimum number of cells per core
        max_over (float, optional): Scaling factor for maximum cells/node
    
    Returns:
        dict: Calibration spec dict updated with node allocation information
    """
    node_alloc, catch_node_map = allocate_catchments_to_nodes(
        cal_spec['extent_map'], max_nodes, cores_per_node, max_over=max_over)

    n_workers = len(node_alloc)

    cal_spec = cal_spec.copy()

    for k in ['prerun_action', 'postrun_action']:
        calldef = cal_spec.get(k)
        if calldef is not None:
            if isinstance(calldef, dict):
                pass
            else:
                cal_spec[k] = callable_to_funcspec(calldef)

    cal_spec['node_alloc'] = node_alloc
    cal_spec['catch_node_map'] = catch_node_map
    cal_spec['n_workers'] = n_workers
    cal_spec['n_sub_workers'] = cores_per_node

    with open(pickle_fn, 'wb') as pkl_out:
        pickle.dump(cal_spec, pkl_out)

    return cal_spec


def build_pbs_file(n_nodes, walltime, pickle_fn, pbs_fn, job_name, job_queue=None, project=None,
                   cores_per_node=None, mem_per_node=None, activation=None, cal_config=None):
    """Build a PBS job file to run a calibration job using a pregenerated pickle

    All parameters must be supplied if cal_config is None, but if cal_config is supplied, then
    these values are used to partially override its contents
    
    Args:
        n_nodes (int): Number of (machine) nodes
        walltime (str): walltime for PBS script, "HH:MM:SS"
        pickle_fn (str): Output filename for generated pickle file
        pbs_fn (str): Output filename for generated PBS file
        job_name (str, optional): PBS job name
        job_queue (str, optional): PBS job queue
        project (str, optional): PBS project
        cores_per_node (int, optional): Number of cores to use on each (machine) node
        mem_per_node ((int,str), optional): Memory to request per node (count,units)
        activation (str, optional): Command to run activation script (normally 'source <SCRIPT>')
        cal_config (dict, optional): Calibration config dict (from ./awrams/<PROFILE>/calibration.py)
    
    Raises:
        Exception: Fails if cal_spec is not supplied and all optional parameters are None
    """
    fh = open(pbs_fn, 'w')

    try:
        if cores_per_node is None:
            cores_per_node = cal_config.CORES_PER_NODE
        if mem_per_node is None:
            mem_per_node = cal_config.MEM_PER_NODE
        if job_queue is None:
            job_queue = cal_config.JOB_QUEUE
        if project is None:
            project = cal_config.PROJECT
        if activation is None:
            activation = cal_config.ACTIVATION
    except:
        raise Exception(
            "Must specify cal_config if other options are not passed")

    n_cores = int(n_nodes)*cores_per_node
    mem_scale, mem_units = mem_per_node
    mem_string = str(int(n_nodes*mem_scale))+mem_units

    base_script = \
        """
    #!/bin/bash
    #PBS -q {job_queue}
    #PBS -P {project}
    #PBS -N {job_name}
    #PBS -l walltime={walltime}
    #PBS -l ncpus={n_cores}
    #PBS -l mem={mem_string}

    {activation}

    python3 -m awrams.calibration.launch_calibration {pickle_fn}

    """.format(**locals())

    fh.write(base_script)
    fh.close()

def cal_spec_to_remote_job(cal_spec, job_name, remote_path, node_count, walltime, sys_settings):
    
    pickle_filename = '%s.pkl' % job_name
    pbs_filename = '%s.pbs' % job_name
    
    cores_per_node = sys_settings['REMOTE_SETTINGS']['PBS_SETTINGS']['CORES_PER_NODE']
    
    cal_spec = build_pickle_from_spec(cal_spec,cores_per_node,node_count,pickle_filename)
    
    # Update node_count in case it has been reduced from bin-packing
    node_count = cal_spec['n_workers']
    
    # Is there anywhere better for this to live?
    task_specs = ['-n 1 --map-by ppr:1:node python3 -m awrams.calibration.server {pickle_filename}', \
                  '-n {n_workers} --map-by ppr:1:node python3 -m awrams.calibration.node {pickle_filename}']
    
    common_opts = sys_settings['MPI_COMMON_OPTIONS']
    format_opts = dict(pickle_filename = pickle_filename, n_workers = node_count)
    
    task_call_str = build_mpi_call_str(task_specs,common_opts,format_opts)
    
    pbs_header = build_pbs_header(**get_pbs_header_options(sys_settings,job_name,walltime,node_count))
    activation = sys_settings['REMOTE_SETTINGS']['ACTIVATION']
    
    build_full_pbs_file(pbs_header, activation, remote_path, task_call_str, pbs_filename)
    
    return RemoteJobSpec(remote_path, pbs_filename, [pickle_filename])