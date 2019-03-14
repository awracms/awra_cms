'''
A set of non-blocking MPI primitives, where lacking (or inconvenient) in
mpi4py
'''

import time
import pickle

def _nofunc():
	return

def mpi_bcast_send(comm,buf,root=0,wt=0.,ifunc=_nofunc):
	r = comm.Ibcast(buf,root)
	mpi_wait(r,wt,ifunc)

def mpi_bcast_recv(comm,buf=None,root=0,wt=0.,ifunc=_nofunc):
	if buf is None:
		buf = bytearray(2**20)
	r = comm.Ibcast(buf,root)
	mpi_wait(r,wt,ifunc)
	return pickle.loads(buf)

def mpi_recv(comm,buf=None,source=-1,tag=-1,wt=0.,ifunc=_nofunc):
	if buf is None:
		buf = bytearray(16384)
	r = comm.irecv(buf,source,tag)
	mpi_wait(r,wt,ifunc)
	return pickle.loads(buf)

def mpi_wait(r,wt=0.,ifunc=_nofunc):
	while r.Test() is not True:
		ifunc()
		time.sleep(wt)