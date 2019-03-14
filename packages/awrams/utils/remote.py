import paramiko
import os
from awrams.utils.general import join_paths
from getpass import getpass
import re
import tarfile
import copy
from glob import glob

class Interactor:
    def __init__(self,ssh_client,username):
        self.client = ssh_client
        self.username = username

    def __call__(self,command):
        stdin, stdout, stderr = self.client.exec_command(command)
        err = [k.strip() for k in stderr.readlines()]
        return [k.strip() for k in stdout.readlines()], err

    def put(self,localfile,remotepath=None):
        if remotepath is None:
            remotepath = os.path.split(localfile)[-1]
        sftp = self.client.open_sftp()
        sftp.put(localfile, remotepath)
        sftp.close()

    def get(self,remotefile,localpath=None):
        filename = os.path.split(remotefile)[-1]
        if localpath is None:
            localpath = filename
        elif os.path.isdir(localpath):
            localpath = join_paths(localpath,filename)
        sftp = self.client.open_sftp()
        sftp.get(remotefile,localpath)
        sftp.close()

def establish_remote_session(host, username=None):
    client = paramiko.SSHClient()

    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if username is None:
        username=input("Username:")

    client.connect(host, username=username,password=getpass('Password for %s:' % username))
    return Interactor(client, username)

def clone_to_raijin(session,remote_path,force=False,local_path=None,explicit=None,exclusions=None):
    '''
    No error checking! This is for convenience only...
    '''

    cwd = os.getcwd()

    if local_path is None:
        local_path=os.environ['AWRAPATH']

    os.chdir(local_path)

    if exclusions is None:
        exclusions = [
            ".git",
            "__pycache__",
            ".ipynb",
            ".nc",
            "notebooks"
        ]

    def filter_function(tarinfo):
        if (
            (os.path.basename(tarinfo.name) in exclusions)
            or (os.path.dirname(tarinfo.name) in exclusions)
            or (os.path.splitext(tarinfo.name)[1] in exclusions)
        ):
            return None
        else:
            return tarinfo

    def make_tarfile(output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            to_add = []
            for k in explicit:
                to_add = to_add + glob(k)

            for k in to_add:
                tar.add(k, filter=filter_function)

    if explicit is None:
        explicit = ['*']

    make_tarfile('out.tgz', local_path)

    if force:
        session('rm -rf %s' % remote_path)

    session('mkdir -p %s' % remote_path)
    try:
        session.put('out.tgz',remote_path + '/out.tgz')
    except:
        print(remote_path)
        raise

    session('cd %s; tar -xf out.tgz; rm out.tgz' % remote_path)
    os.remove("out.tgz")

    session('cd %s; mv raijin_activate.sh activate' % remote_path)
    session('cd %s/Config/; mv raijin_host_defaults.py host_defaults.py' % remote_path)

    os.chdir(cwd)

def clone_to_remote(session,local_path,remote_path,force=False,explicit=None,exclusions=None):
    '''
    No error checking! This is for convenience only...
    '''

    cwd = os.getcwd()

    os.chdir(local_path)

    if exclusions is None:
        exclusions = []

    if explicit is None:
        explicit = ['*']

    def filter_function(tarinfo):
        if (
            (os.path.basename(tarinfo.name) in exclusions)
            or (os.path.dirname(tarinfo.name) in exclusions)
            or (os.path.splitext(tarinfo.name)[1] in exclusions)
        ):
            return None
        else:
            return tarinfo

    def make_tarfile(output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            to_add = []
            for k in explicit:
                to_add = to_add + glob(k)

            for k in to_add:
                tar.add(k, filter=filter_function)

    make_tarfile('out.tgz', local_path)

    if force:
        session('rm -rf %s' % remote_path)

    session('mkdir -p %s' % remote_path)
    try:
        session.put('out.tgz',remote_path + '/out.tgz')
    except:
        print(remote_path)
        raise

    session('cd %s; tar -xf out.tgz; rm out.tgz' % remote_path)
    os.remove("out.tgz")

    os.chdir(cwd)

class PBSJob:
    def __init__(self, client, pbs_name, remote_path = None):
        self.client = client
        self.pbs_name = pbs_name
        self.remote_path = remote_path

    def get_status(self, full_status=False):
        status_str, error_str = self.client('qstat %s' % self.pbs_name)
        if len(error_str) > 0:
            if re.match('.*Job has finished.*',str(error_str)) is not None:
                return 'Finished'
        else:
            if full_status:
                return status_str
            else:
                sstr = status_str[2]
                status_code = [k for k in sstr.split(' ') if len(k) > 0][-2]
                return status_code

    def list_files(self,sub_path = None, detailed=False):
        if self.remote_path is None:
            raise Exception("Remote path unknown, cannot get file list")
        else:
            if sub_path is not None:
                full_path = join_paths(self.remote_path, sub_path)
            else:
                full_path = self.remote_path
            if detailed:
                call_str = 'ls -al %s' % full_path
            else:
                call_str = 'ls %s' % full_path
            res,err = self.client(call_str)
            if len(err):
                raise Exception(err)
            else:
                return res

    def get_files(self, filenames):
        if self.remote_path is None:
            raise Exception("Remote path unknown, cannot get files")
        else:
            if isinstance(filenames, str):
                filenames = [filenames]
            for f in filenames:
                self.client.get('%s/%s' % (self.remote_path,f))

    def get_output(self, tail_n = None):
        return self._qcat('o', tail_n)

    def get_errors(self, tail_n = None):
        return self._qcat('e', tail_n)

    def _qcat(self, mode = 'o', tail_n = None):
        qcat_str = 'qcat -%s %s' % (mode, self.pbs_name)
        if tail_n is not None:
            tail_str = ' | tail -n %s' % tail_n
        else:
            tail_str = ''
        res,err = self.client(qcat_str + tail_str)

        finished = False

        if len(res):
            if res[0] == 'PBS error: Job has finished':
                finished = True
        elif len(err):
            if 'No such file' in err[0]:
                finished = True
        if finished:
            res, err = self.client('find %s/*.%s%s' % (self.remote_path,mode,self.pbs_name))
            filename = res[0]
            res, err = self.client('cat %s%s' % (filename,tail_str))
        
        return res, err

    def cancel(self):
        return self.client('qdel %s' % self.pbs_name)


class CalibrationSession:
    def __init__(self,cal_settings=None,client=None):
        if cal_settings is None:
            cal_settings = get_profile_settings('calibration')
        if client is None:
            client = establish_remote_session(cal_settings.USERNAME,cal_settings.REMOTE_HOST)
        self.client = client
        self.settings = cal_settings

    def submit_job(self,cal_spec,job_name,walltime,max_nodes):
        PKL_FNAME = '%s.pkl'%job_name
        PBS_FNAME = '%s.pbs'%job_name
        RESULTS_FNAME = '%s.h5'%job_name
        JOB_PATH = self.settings.JOB_PATH

        cal_spec['logfile'] = '%s/%s.h5' % (JOB_PATH,job_name)

        build_remote_pbs_from_cal_spec(cal_spec,walltime,PKL_FNAME,PBS_FNAME,JOB_PATH,job_name,max_nodes,\
                                       self.settings)
        self.client.put(PKL_FNAME,join_paths(JOB_PATH,PKL_FNAME))
        self.client.put(PBS_FNAME,join_paths(JOB_PATH,PBS_FNAME))
        qo,qe = self.client('cd %s; qsub %s'%(JOB_PATH,PBS_FNAME))
        if len(qe) > 0:
            raise Exception(qe)
        pbs_id = qo[0]
        caljob = PBSJob(self.client,pbs_id)
        return caljob

    def qstat(self):
        return self.client('qstat -u %s' % self.settings.USERNAME)

    def list_completed(self):
        o,e = self.client('ls %s/*.h5'%self.settings.JOB_PATH)
        return [k.split('.')[-2].split('/')[-1] for k in o]

    def get_results(self,job_name,open_on_return=True):
        results_fname = '%s.h5' % job_name
        self.client.get('%s/%s' % (self.settings.JOB_PATH,results_fname))
        if open_on_return:
            return CalibrationResults(results_fname)

    def get_job(self,pbs_id):
        return PBSJob(self.client,pbs_id)
