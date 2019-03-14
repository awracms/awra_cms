from numbers import Number
import os
from awrams.utils.remote import PBSJob, establish_remote_session, clone_to_remote
from awrams.utils.general import join_paths

def build_mpi_call_str(task_specs,common_options,format_opts=None,runner='mpiexec'):
    """Summary
    
    Args:
        task_specs (list): List of strings, each containing an MPI task description, eg
                           e.g: ['-n {nrunners} python3 awrams.module.example {pickle_file}']
        common_options (list): List of MPI options that are expanded to for all tasks
        format_opts (dict, optional): Dictionary for formatting options within task_specs
        runner (str, optional): Name of MPI executable
    
    Returns:
        str: Full runnable MPI call string
    """
    if format_opts is None:
        format_opts = {}
    call_str = runner + ' '
    for t_str in task_specs:
        call_str = call_str + ''.join([o + ' ' for o in common_options]) + t_str + ' : '
    return call_str.format(**format_opts)

def get_pbs_node_requirements(sys_settings,node_count):
    """Get the cpu and memory requirements for a given number of nodes
    
    Args:
        sys_settings (dict): System settings dict, as supplied from config_manager
        node_count (int): Number of whole nodes on target system
    
    Returns:
        dict: ncpus and mem for target number of nodes
    """
    ncpus = node_count * sys_settings['REMOTE_SETTINGS']['PBS_SETTINGS']['CORES_PER_NODE']
    mem_per_node = sys_settings['REMOTE_SETTINGS']['PBS_SETTINGS']['MEM_PER_NODE']
    mem = '%s%s' % (int(node_count * mem_per_node[0]), mem_per_node[1])
    return dict(ncpus=ncpus,mem=mem)

def build_pbs_header(job_queue,project,job_name,walltime,ncpus,mem,**kwargs):
    pbs_base_str = """#!/bin/bash
                   #PBS -q {job_queue}
                   #PBS -P {project}
                   #PBS -N {job_name}
                   #PBS -l walltime={walltime}
                   #PBS -l ncpus={ncpus}
                   #PBS -l mem={mem}"""
    out_str = pbs_base_str.format(job_queue=job_queue,project=project,job_name=job_name,walltime=walltime,ncpus=ncpus,mem=mem)
    out_str = ''.join([s.lstrip(' ')+'\n' for s in out_str.splitlines()])
    
    for k,v in kwargs.items():
        out_str = out_str + '#PBS %s\n' % v
    
    return out_str

def get_pbs_header_options(sys_settings,job_name,walltime,node_count=None,ncpus=None,mem=None):
    pbs_settings = sys_settings.REMOTE_SETTINGS.PBS_SETTINGS
    
    header_options = {
        'job_queue': pbs_settings['JOB_QUEUE'],
        'project': pbs_settings['PROJECT'],
        'job_name': job_name,
        'walltime': walltime
    }

    if node_count is None:
        if ncpus is None or mem is None:
            raise Exception("Must supply either [node_count] or [ncpus and mem]")
        else:
            hw_req = dict(ncpus=ncpus,mem=mem)
    else:
        if (ncpus is not None or mem is not None):
            raise Exception("Must supply either [node_count] or [ncpus and mem]; not both")
        else:
            hw_req = get_pbs_node_requirements(sys_settings, node_count)
    
    header_options.update(hw_req)

    if pbs_settings['NOTIFICATION']['NOTIFY'] == True:
        email = pbs_settings['NOTIFICATION']['EMAIL']
        notify_opts = pbs_settings['NOTIFICATION']['NOTIFY_OPTS']
        notify_dict = dict(email = '-M %s' % email, notify_opts = '-m %s' % notify_opts)
        header_options.update(notify_dict)
    
    return header_options

def build_custom_pbs_file(job_name, remote_path, walltime, sys_settings, \
                          task_call_str, node_count = None, ncpus = None, \
                          mem = None):

    pbs_filename = '%s.pbs' % job_name
    
    pbs_header_options = get_pbs_header_options(sys_settings,job_name,walltime,node_count,ncpus,mem)
    
    pbs_header = build_pbs_header(**pbs_header_options)
    
    activation = sys_settings['REMOTE_SETTINGS']['ACTIVATION']

    return build_full_pbs_file(pbs_header, activation, remote_path, task_call_str, pbs_filename)


def build_full_pbs_file(pbs_header, activation, remote_path, task_call_str, pbs_filename):
    
    cd_str = 'cd {working_dir}\n'.format(working_dir=remote_path)
    
    full_pbs_str = pbs_header + activation + cd_str + '\n' + task_call_str + '\n'
    with open(pbs_filename,'w',newline='\n') as pbs_fh:
        pbs_fh.write(full_pbs_str)
    
    return full_pbs_str


def send_required_files(session,required_files,remote_path,make_path=True):
    if make_path:
        output,error = session('mkdir -p %s' % remote_path)
        if len(error):
            raise Exception(error)
    for f in required_files:
        session.put(f,join_paths(remote_path,f))

#-----
# PBS dependancies support code 
#-----

def normalise_dependency(dependency):
    if isinstance(dependency,PBSJob):
        return [dependency.pbs_name]
    elif isinstance(dependency,Number):
        return [str(dependency)]
    elif isinstance(dependency,str):
        return [dependency]
    elif isinstance(dependency,list):
        return [normalise_dependency(d) for d in dependency]
    else:
        raise Exception("Dependency is of unknown type %s" % type(dependency))

def dependencies_to_qsub(dependencies,depend_type='afterok'):
    if dependencies is None:
        return ''
    else:
        dependencies = normalise_dependency(dependencies)
        dstr = depend_type + ''.join([':' + d for d in dependencies])
        return '-W depend=%s ' % dstr

#-----
# RemotePBSManager
# Handles interactions with remote PBS session (job submission, tracking etc)
#-----


class RemotePBSManager:
    def __init__(self, sys_settings, session = None):
        self.sys_settings = sys_settings

        if session is None:
            host_settings = sys_settings['REMOTE_SETTINGS']['HOST_SETTINGS']
            session = establish_remote_session(host_settings['HOSTNAME'], host_settings['USERNAME'])

        self.session = session
        
    def submit_job(self,remote_path,pbs_file,job_files=[],dependencies=None,depend_type='afterok'):
        """
        https://opus.nci.org.au/display/Help/How+to+use+job+dependencies
        """
        send_required_files(self.session,job_files + [pbs_file],remote_path)
        
        depend_str = dependencies_to_qsub(dependencies,depend_type)
        
        remote_pbs_file = os.path.split(pbs_file)[1]
        
        output,error = self.session('cd %s; qsub %s%s' % (remote_path, depend_str, remote_pbs_file))
        if len(error):
            raise Exception(error)
        else:
            return PBSJob(self.session,output[0].split('.')[0], remote_path)

    def submit_job_from_spec(self,job_spec,dependencies=None,depend_type='afterok'):
        
        return self.submit_job(job_spec.remote_path, job_spec.pbs_file, \
                               job_spec.required_files, dependencies, depend_type)

    def qstat(self):
        return self.session('qstat -u %s' % self.session.username)

    def sync_user_files(self):
        awrams_base_remote = self.sys_settings['DATA_PATHS']['AWRAMS_BASE']
        awrams_base_local = os.environ['AWRAMS_BASE_PATH']
        for target in ['config','code']:
            rpath = join_paths(awrams_base_remote,target)
            lpath = join_paths(awrams_base_local,target)
            clone_to_remote(self.session,lpath,rpath,True)

    def sync_dev_repo(self,local_repo_path):
        remote_path = self.sys_settings['config_options']['REPO_PATH']
        #explicit = 
        exclusions = ['.git','__pycache__','.nc','.h5']
        clone_to_remote(self.session,local_repo_path,remote_path,True,exclusions=exclusions)


class RemoteJobSpec:
    def __init__(self,remote_path,pbs_file,required_files=[]):
        self.remote_path = remote_path
        self.pbs_file = pbs_file
        self.required_files = required_files

    def __repr__(self):
        return 'RemoteJobSpec:\nremote_path: %s\npbs_file: %s\nrequired_files: %s\n' % \
                (self.remote_path, self.pbs_file, self.required_files)



# for target in ['config','code']:
#     rpath = os.path.join(awrams_base_remote,target)
#     lpath = os.path.join(awrams_base_local,target)
#     remote.clone_to_raijin(session,rpath,True,lpath,'*')