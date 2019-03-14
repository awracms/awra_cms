# Initialises multiprocessing libs to use forkserver
# Import this file _before_ doing anything fork-sensitive
# E.g HDF libraries, MPI, ZMQ...

import multiprocessing as _mp

try:
	_mp.set_start_method('forkserver')
	_p = mp.Process()
	_p.start()
	_p.join()

	del(_p)
except:
	pass