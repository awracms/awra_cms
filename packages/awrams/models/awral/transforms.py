import numpy as np
from awrams.utils.nodegraph.nodes import InputNode, ProcessNode, GraphNode, ForcingNode
from . import solar

def interlayer_k(ktop,kbottom):
    return np.minimum(150.0, np.maximum(1.0, ktop/kbottom))

def k_rout(k_rout_scale,k_rout_int,mean_pet):
    return k_rout_scale* mean_pet + k_rout_int

def pe(tmin):
    return 610.8 * np.exp(17.27 * tmin / (237.3 + tmin))

def fday():
    return GraphNode('fday',FDayNode,'forcing')

def u2t(windspeed_spatial,fday):
    return GraphNode('windspeed',WindspeedNode,'forcing',[windspeed_spatial,fday])

def radcskyt():
    return GraphNode('radcskyt',RadClearSkyNode,'forcing')

class FDayNode(ForcingNode):
    dtype = np.float64

    def get_data(self,coords):
        fday = np.empty(coords.shape,dtype=self.dtype)
        for i, lat in enumerate(coords.latitude):
            f_lat = solar.fday(lat,coords.time[0],coords.time[-1])
            fday[:,i,:] = f_lat.repeat(coords.shape[2]).reshape(coords.shape[0],coords.shape[2])
        return fday

class WindspeedNode(ProcessNode):
    def process(self,inputs):
        wspeed_grid = inputs[0]
        fday = inputs[1]
        return wspeed_grid * ((1 - (1 - fday) * 0.25) / fday)

class RadClearSkyNode(ForcingNode):
    dtype = np.float64

    def get_data(self,coords):
        rcskyt = np.empty(coords.shape,dtype=self.dtype)
        for i, lat in enumerate(coords.latitude):
            r_lat = solar.rad_clear_sky(lat,coords.time[0],coords.time[-1])
            rcskyt[:,i,:] = r_lat.repeat(coords.shape[2]).reshape(coords.shape[0],coords.shape[2])
        return rcskyt