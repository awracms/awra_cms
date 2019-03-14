import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator

def get_pctiles(data,pctile_keys=list(range(0,100)),axis=0):
    pctiles = np.percentile(data,pctile_keys,axis=axis)
    return pctiles
    
def assign_to_deciles(data,pctile_values,pctile_keys):
    '''
    Map data to a quantized percentile range
    pctiles: ndarray 
    '''
    out = np.zeros_like(data)
    for i in range(pctile_values.shape[0]):
        mask = data>pctile_values[i]
        out[mask] = pctile_keys[i]
    return out

def mean_fill(data,mask,iterations=3,sd=1.0):
    out = data.copy()
    out[mask==True] = data[mask==False].mean()
    for i in range(iterations):
        out[mask==True] = ndimage.filters.gaussian_filter(out,sd)[mask==True]
    return np.ma.MaskedArray(data=out,mask=mask)

def water_cmap(n=10):
    '''
    Generate a segmented Red-White-Blue colourmap
    '''
    amap = mpl.colors.LinearSegmentedColormap.from_list('lsmap',[[1.0,0,0],[1.0,1.0,1.0],[0,0,1.0]],n)
    amap.set_bad((1,1,1))
    amap.set_over((0,0,1))
    amap.set_under((1,0,0))
    return amap

def borders_from_mask(mask):
    '''
    Return True for all True pixels with at least one vertical or horizontal (False) neighbour
    '''
    nmask = np.zeros_like(mask,dtype=np.bool)

    invmask = ~mask
    
    for x in range(invmask.shape[0]):
        for y in range(invmask.shape[1]):
            if invmask[x,y] == False:
                neighbours = 0
                if x>0:
                    neighbours += invmask[x-1,y]
                if x<invmask.shape[0]-1:
                    neighbours += invmask[x+1,y]
                if y>0:
                    neighbours += invmask[x,y-1]
                if y<invmask.shape[1]-1:   
                    neighbours += invmask[x,y+1]
                if neighbours > 0:
                    nmask[x,y] = True
    
    return nmask

def plot_decile_map(data,pctile_data,pctile_keys,decile_keys,title='',sigma=0.0,scale=1.5):
    nmask = borders_from_mask(data.mask)
    
    decilic_data = assign_to_deciles(data,pctile_data,pctile_keys)

    if sigma > 0.0:
        filled_out = mean_fill(decilic_data,data.mask,30,sigma)
        filtered = ndimage.filters.gaussian_filter(filled_out,sigma) 
    else:
        filtered = decilic_data

    img_data = np.ma.MaskedArray(data=filtered,mask=data.mask.copy())
    img_data.mask[nmask == True] = False
    img_data[nmask == True] = 999

    #+++ Hardcoded figsize
    fig = mpl.pylab.plt.figure(figsize=(scale*data.shape[1]/100.,scale*data.shape[0]/100.),dpi=100)
    ax = fig.gca()
    
    cmap=water_cmap(len(decile_keys)-1)
    bounds = decile_keys
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(img_data,cmap=cmap,norm=norm)

    plt.title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb2 = mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,ticks=bounds, spacing='proportional') 

def plot_decile(data,decile_keys,title='',sigma=0.0,scale=1.5,nmask=None):
    if nmask is None:
        nmask = borders_from_mask(data.mask)

    decilic_data = data.copy()

    if sigma > 0.0:
        filled_out = mean_fill(decilic_data,data.mask,30,sigma)
        filtered = ndimage.filters.gaussian_filter(filled_out,sigma) 
    else:
        filtered = decilic_data

    img_data = np.ma.MaskedArray(data=filtered,mask=data.mask.copy())

    #+++ Hardcoded figsize
    fig = plt.figure(figsize=(scale*data.shape[1]/100.,scale*data.shape[0]/100.),dpi=100)
    ax = fig.gca()
    
    cmap=water_cmap(len(decile_keys)-1)
    bounds = decile_keys
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(img_data,cmap=cmap,norm=norm)

    lsc = mpl.colors.LinearSegmentedColormap.from_list('maskmap',[(0.,0,0.,0.),(1.,1.,1.,0.)],2)
    lsc.set_over((0,0,0,1.))
    ax.imshow(nmask,cmap=lsc,vmin=0,vmax=0.5)

    plt.title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb2 = mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,ticks=bounds, spacing='proportional') 

def nearest_neighbour_fill(data):
    '''
    Return an array whose masked values are filled by the nearest unmasked neighbour
    '''

    out_data = data.copy()
    mask = data.mask

    validpts = np.where(mask==False)
    validpts = np.array(list(zip(validpts[0],validpts[1])))
    values = data[mask==False].flatten()
    invalidpts = np.where(mask==True)
    itp = NearestNDInterpolator(validpts,values)

    out_data[mask==True] = itp(invalidpts[0],invalidpts[1])

    return out_data