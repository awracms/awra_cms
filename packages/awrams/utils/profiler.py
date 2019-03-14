"""
Tools for runtime profiling of the system

NestedProfiler is the current 'best in class' portion of this module
with regards to CPU utilisiation.
MemoryInstrumentation is Linux-only and has the usual issues with
Linux memory measurement (reserved vs 'actual' usage etc)
"""


from time import time
from numpy import floor #pylint: disable=no-name-in-module
from collections import OrderedDict

def do(func, times):
    '''
    Execute func a number of times
    '''
    for i in range(times):
        func()

def seconds_to_hms(time_s):
    '''
    Convert a float (number of seconds) to a HMS string
    '''
    hours = int(floor(time_s/3600.0))
    rem = time_s - (3600.0*hours)
    mins = int(floor(rem/60.0))
    seconds = rem - (60.0 * mins)
    return (hours,mins,seconds)

def hms_to_str(hms):
    '''
    Convert an HoursMinutesSeconds tuple to a string
    '''
    if hms[0] > 0:
        return "%sh%sm%.1f" % (hms[0],hms[1],hms[2])
    elif hms[1] > 0:
        return "%sm%.2fs" % (hms[1],hms[2])
    else:
        return "%.4fs" % hms[2]

def profile_stats(eval_str):
    '''
    Profile a function, return stats
    '''
    import cProfile
    import pstats
    cProfile.run(eval_str,'run_stats.txt')
    ps = pstats.Stats('run_stats.txt')
    ps.sort_stats('tottime')
    return CallStats(ps)

def plot_stats(ps,thresh=0.02,label_mode='filename'):
    '''
    Plot a pstats.Stats object
    thresh: minimum pct of time for inclusion in graph
    label_mode: 'filename','full'
    '''
    import matplotlib as mpl
    out = {}
    for k,v in list(ps.stats.items()):
        #if len(v[4].keys()) > 0:
        #if out.has_key(str(v[4].keys()[0])):
        #    out[str(v[4].keys()[0])] += v[2]
        #else:
        #    out[str(v[4].keys()[0])] = v[2]
        out[str(k)] = v[2]
    tot_time = sum(out.values())
    true_out = {}
    other_time = 0.
    for k,v in list(out.items()):
        if v/tot_time > thresh:
            true_out[k] = v
        else:
            other_time += v
    true_out['other'] = other_time

    in_labels = list(true_out.keys())

    if label_mode == 'filename':
        labels = []
        for label in in_labels:
            labels.append("'" + label.split('/')[-1])
    else:
        labels = in_labels

    fig = mpl.pylab.figure(figsize=(8,8))
    mpl.pylab.pie(list(true_out.values()),labels=labels,autopct='%.1f')

    return true_out


class Profiler():
    '''
    Very simple profiler; records blocks spent in marked sections of code
    '''

    def __init__(self):
        self.measurements = {}

    def begin_profiling(self):
        '''
        Being a profiling run
        '''
        self.measurements = {}
        self.start_time = time()

    def start(self,identifier):
        '''
        Start a profiling a particular block of code
        '''
        if identifier not in self.measurements:
            self.measurements[identifier] = 0.
        self.cur_start = time()

    def stop(self,identifier):
        '''
        Stop profiling a particular block of code
        '''
        now = time()
        cur_time = now - self.cur_start
        self.measurements[identifier] += cur_time

    def end_profiling(self):
        '''
        End a profiling run, calculate stats
        '''
        self.end_time = time()
        self.total_time = self.end_time - self.start_time

    def occupancy(self):
        '''
        Return the percentage of time spent in blocks other than 'waiting'
        '''
        if 'waiting' in self.measurements:
            return 100.0 * (1.0 - self.measurements['waiting']/self.total_time)
        else:
            return 100.0

    def repr_stats(self):
        '''
        Return stats as a human readable string
        '''
        repr_list = []
        accounted_time = 0.
        for k,v in list(self.measurements.items()):
            accounted_time += v
            pct_time = v/self.total_time
            repr_list.append([pct_time,k])
        repr_list.append([1.0-(accounted_time/self.total_time),'other'])
        repr_list.sort(reverse=True)
        out_str = 'Total Runtime: %s\n' % hms_to_str(seconds_to_hms(self.total_time))
        out_str += 'Breakdown:\n'
        for item in repr_list:
            out_str += '%s : %.2f%%\n' % (item[1],item[0]*100.)
        return out_str

class MeasurementGroup:
    def __init__(self):
        self.children = {}
        self.time = 0.
        self._start = None
        self.active = False

    def start(self):
        if self.active is False:
            self._start = time()
            self.active = True
        else:
            raise Exception("Start already called on MeasurementGroup")

    def stop(self):
        if self.active is True:
            self.time += time() - self._start
            self.active = False
        else:
            raise Exception("Stop already called on MeasurementGroup")

    def __getitem__(self,k):
        return self.children[k]

    def __OLD_repr__(self):
        return self.__report__()

    def __report__(self,level=0):
        out = ''
        if level == 0:
            out = 'Total time: %.2fs\n' % self.time
        accounted_time = 0.
        for k,v in self.children.items():
            accounted_time += v.time
            out = out + level*'  ' + '%s: %.2f%%\n' % (k,v.time/self.time*100.0)
            out = out + v.__report__(level+1)
        if len(self.children) > 0:
            out = out + level*'  ' + 'other: %.2f%%\n' % ((self.time-accounted_time)/self.time*100.0)
        return out

class NestedProfiler():
    '''
    Profiler with heirarchical grouping
    '''

    def __init__(self):
        self.root = MeasurementGroup()
        self.stack = []  # Stateful group for submeasurements; +++ maybe use deque?
        self.cur_group = self.root

    def begin_profiling(self):
        '''
        Being a profiling run
        '''
        self.root.start()

    def __getitem__(self,k):
        return self.root[k]

    def start(self,identifier):
        '''
        Start a profiling a particular block of code
        '''
        mg = self.cur_group.children.get(identifier)
        if mg is None:
            mg = MeasurementGroup()
            self.cur_group.children[identifier] = mg

        mg.start()
        self.stack.append(self.cur_group)
        self.cur_group = mg

    def stop(self):
        '''
        Stop profiling the current group
        '''
        self.cur_group.stop()
        self.cur_group = self.stack.pop()

    def end_profiling(self):
        '''
        End a profiling run, calculate stats
        '''
        while self.cur_group != self.root:
            self.stop()
        self.root.stop()

    def report(self):
        return self.root.__repr__()#.__report__()

    def __repr__(self):
        return self.report()

class DummyProfiler:
    def __init__(self):
        pass

    def begin_profiling(self):
        pass

    def start(self,identifier):
        pass

    def stop(self,identifier):
        pass

    def end_profiling(self):
        pass

    def repr_stats(self):
        'Profiling disabled'


"""
Memory usage
"""

import os

_mem_scale = {'kB': 1./1024.0, 'mB': 1.,
          'KB': 1./1024.0, 'MB': 1.}

def _VmB(pid, VmKey):
    proc_status = '/proc/%d/status' % pid
    try:
        t = open(proc_status)
        v = t.read()
        t.close()
    except:
        # +++Should really raise an exception, but for all intents
        # and purposes, this is more useful...
        return 0.0
        #raise Exception("No proc info for pid %s" % pid)
    try:
        i = v.index(VmKey)
    except:
        #most likely zombie process
        return 0.0
    v = v[i:].split(None, 3)
    if len(v) < 3:
        raise Exception("Unknown memory usage format")
    return float(v[1]) * _mem_scale[v[2]]


def vmem(pid=os.getpid()):
    '''Return memory usage in megabytes.
    '''
    return _VmB(pid,'VmSize:')


def resident(pid=os.getpid()):
    '''Return resident memory usage in megabytes.
    '''
    return _VmB(pid,'VmRSS:')


def stacksize(pid=os.getpid()):
    '''Return stack size in megabytes.
    '''
    return _VmB(pid,'VmStk:')

class MemoryInstrumentation:
    '''
    Provides a dictionary of name/pid pairs and
    reports on their memory usage
    '''
    def __init__(self):
        self.processes = OrderedDict()

    def __setitem__(self,k,v):
        self.processes[k] = v

    def __getitem__(self,k):
        return resident(self.processes[k])

    def report(self):
        usage = {}
        total_mem = 0.
        for v in list(self.processes.values()):
            cur_usage = resident(v)
            usage[v] = cur_usage
            total_mem += cur_usage

        out_str = 'Total memory: %.2fmb' % total_mem
        if (len(self.processes) > 1):
            for k, v in list(self.processes.items()):
                out_str += '\n%s (%s): %.2fmb' % (k, v, usage[v])

        return out_str

    def __repr__(self):
        return self.report()

'''
Heirarchical callee graph from pstats
'''
from awrams.utils.metatypes import ObjectDict as o

class CallStat(object):
    '''
    Initialise from a pstats stat object
    '''
    def __init__(self,ps):
        self.key = ps[0]
        self.function = o(fn = ps[0][0], line = ps[0][1], func_name = ps[0][2])
        stats = ps[1]
        self.ncalls = stats[0]
        self.tot_time = stats[2]
        self.cum_time = stats[3]
        self.callers = stats[4]

    def __repr__(self):
        return "%s : %s" % (self.key, self.cum_time)

    def plot(self):
        plot_callees(self)

class CallStats(object):
    def __init__(self,pstats):
        stats = []
        for ps in list(pstats.stats.items()):
            stat = CallStat(ps)
            stats.append(stat)

        self.stats = sorted(stats,key=lambda k: k.cum_time,reverse=True)

        self._build_callees()

    def query(self,q):
        results = []
        for stat in self.stats:
            if q in stat.key[0] or q in stat.key[2]:
                results.append(stat)
        if len(results) > 1:
            return results
        elif len(results) == 1:
            return results[0]
        else:
            return None

    def _find_callees(self,stat):
        callees = []
        for s in self.stats:
            for pos_caller in list(s.callers.items()):
                if pos_caller[0] == stat.key:
                    callees.append([s.key,pos_caller[1]])
        return sorted(callees,key=lambda k: k[1][3],reverse=True)

    def _build_callees(self):
        for stat in self.stats:
            stat.callees = self._find_callees(stat)

    def __repr__(self):
        out = ''
        for stat in self.stats:
            out = out + ('%s: %s\n' % (stat.key, stat.cum_time))
        return out

    def plot_totals(self,thresh=0.02):
        '''
        Plot total times taken by called functions
        '''
        import matplotlib as mpl
        out = {}
        for stat in self.stats:
            #if len(v[4].keys()) > 0:
            #if out.has_key(str(v[4].keys()[0])):
            #    out[str(v[4].keys()[0])] += v[2]
            #else:
            #    out[str(v[4].keys()[0])] = v[2]
            out[stat.key] = stat.tot_time
        tot_time = sum(out.values())
        true_out = {}
        other_time = 0.
        for k,v in list(out.items()):
            if v/tot_time > thresh:
                true_out[k] = v
            else:
                other_time += v
        true_out['other'] = other_time

        in_labels = list(true_out.keys())

        fig = mpl.pylab.figure(figsize=(8,8))
        mpl.pylab.pie(list(true_out.values()),labels=in_labels,autopct='%.1f')

        print(true_out)


def print_callees(stat):
    print(stat.key)
    print(stat.cum_time)
    callees = sorted(stat.callees,key=lambda k: k[1][3],reverse=True)
    for callee in callees:
        print(callee[0], callee[1][3])

def plot_callees(stat,thresh=0.02):
    callees = sorted(stat.callees,key=lambda k: k[1][3],reverse=True)

    import matplotlib as mpl

    out = {}
    for callee in callees:
        #if len(v[4].keys()) > 0:
        #if out.has_key(str(v[4].keys()[0])):
        #    out[str(v[4].keys()[0])] += v[2]
        #else:
        #    out[str(v[4].keys()[0])] = v[2]
        out[str(callee[0])] = callee[1][3]

    tot_time = stat.cum_time
    true_out = {}
    other_time = 0.
    for k,v in list(out.items()):
        if v/tot_time > thresh:
            true_out[k] = v
        else:
            other_time += v
    true_out['other'] = other_time

    in_labels = list(true_out.keys())

    fig = mpl.pylab.figure(figsize=(8,8))
    mpl.pylab.pie(list(true_out.values()),labels=in_labels,autopct='%.1f')

    print_callees(stat)

def plot_stats(ps,thresh=0.02,label_mode='filename'):
    '''
    Plot a pstats.Stats object
    thresh: minimum pct of time for inclusion in graph
    label_mode: 'filename','full'
    '''
    import matplotlib as mpl
    out = {}
    for k,v in list(ps.stats.items()):
        #if len(v[4].keys()) > 0:
        #if out.has_key(str(v[4].keys()[0])):
        #    out[str(v[4].keys()[0])] += v[2]
        #else:
        #    out[str(v[4].keys()[0])] = v[2]
        out[str(k)] = v[2]
    tot_time = sum(out.values())
    true_out = {}
    other_time = 0.
    for k,v in list(out.items()):
        if v/tot_time > thresh:
            true_out[k] = v
        else:
            other_time += v
    true_out['other'] = other_time

    in_labels = list(true_out.keys())

    if label_mode == 'filename':
        labels = []
        for label in in_labels:
            labels.append("'" + label.split('/')[-1])
    else:
        labels = in_labels

    fig = mpl.pylab.figure(figsize=(8,8))
    mpl.pylab.pie(list(true_out.values()),labels=labels,autopct='%.1f')

    return true_out
