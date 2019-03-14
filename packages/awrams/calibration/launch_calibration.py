"""Entry point for launching calibration runs
Uses pickle files as generated in awrams.calibration.cluster
"""
import argparse
import os
import sys
import pickle
from subprocess import Popen, PIPE, STDOUT, signal
import sys


def run_from_pickle(pickle_file):
    """
    Launch an MPI calibration job from the specified picklefile 
    (usually built from awrals.calibration.cluster.build_pickle_from_spec)
    
    Args:
        pickle_file (str): Input pregenerated pickle file
    
    Returns:
        int: Return code of the executed mpirun command
    """

    cspec = pickle.load(open(pickle_file, 'rb'))
    n_workers = cspec['n_workers']

    if os.name is 'nt':
        call_str = ('mpiexec  -al 0 -n 1 '
                    'python -m awrams.calibration.server {pickle_file} : '
                    '-al 0 -n {n_workers} '
                    'python -m awrams.calibration.node').format(**locals())

        proc = Popen(call_str, stdout=PIPE, stderr=STDOUT, shell=True)
    else:
        call_str = ('mpirun --oversubscribe --allow-run-as-root --mca plm_rsh_agent '
                    'false -x TMPDIR=/dev/shm/ -n 1 --map-by ppr:1:node '
                    'python3 -m awrams.calibration.server {pickle_file} : '
                    '-x TMPDIR=/dev/shm/ -n {n_workers} --map-by ppr:1:node '
                    'python3 -m awrams.calibration.node').format(**locals())        

        proc = Popen(call_str, stdout=PIPE, stderr=STDOUT, shell=True, 
                     preexec_fn = os.setsid)

    cur_out = ' '

    try:

        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()

    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()

    return_code = proc.wait()

    return return_code


def get_nodelist():
    """Get the hostnames of each available (machine) node
    
    Returns:
        list: List of hostnames available in this PBS environment
    """
    return list(set([n.strip() for n in open(os.environ['PBS_NODEFILE']).readlines()]))


def build_nodestring(nodelist):
    """Convert get_nodelist output to string
    
    Args:
        nodelist (list): List of hostnames available (see get_nodelist())
    
    Returns:
        str: String representation of nodelist
    """
    return ''.join([n+',' for n in nodelist])[:-1]


if __name__ == '__main__':
    """
    Command line module entry point for launching calibration jobs
    """

    parser = argparse.ArgumentParser(description='Launch a clustered job')
    parser.add_argument('pickle_file', type=str,
                        help='filename of pickled cal_spec')

    args = parser.parse_args()

    pickle_file = args.pickle_file

    run_from_pickle(pickle_file)
