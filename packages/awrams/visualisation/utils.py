import base64
from io import BytesIO
import os

import matplotlib as mpl
from matplotlib import tight_layout
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mpl_col
import matplotlib.pyplot as plt
import numpy as np
import osgeo.gdal as gd
import pandas as pd

import awrams.utils.catchments as cat
import awrams.utils.datetools as dt
from awrams.utils.extents import from_boundary_coords
import awrams.utils.io.data_mapping as dm


def show_shape(shp_file, match=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    sdb = cat.ShapefileDB(shp_file)

    if match is None:
        shapes = sdb._records
    else:
        shapes = []
        for shape in sdb._records:
            for k in match.keys():
                if shape[k] == match[k]:
                    shapes.append(shape)

    def get_points(gr, c=0):
        _gr = gr.GetGeometryRef(c)
        if _gr.GetGeometryName() == 'LINEARRING':
            return _gr.GetPoints()
        else:
            try:
                return get_points(_gr)
            except:
                return

    for shape in shapes:
        gr = shape.GetGeometryRef()
        for c in range(gr.GetGeometryCount()):
            pts = get_points(gr,c)
            if pts:
                x = []
                y = []
                for pt in pts:
                    x.append(pt[0])
                    y.append(pt[1])
                ax.plot(x,y,color='m')

def display_gdal(fn,mask_value=None,slice=slice(None)):
    g = gd.Open(fn)
    a = g.GetRasterBand(1).ReadAsArray()
    print(a.min(),a.max(),a.shape)
    if mask_value is not None:
        a = np.ma.masked_less_equal(a, mask_value)
        print(a.min(),a.max(),a.shape)
    fig = plt.figure(figsize=(16,20))
    ax = fig.add_subplot(1,1,1)

    ax.imshow(a[slice],interpolation='none')
    return a

def display_ncslice(fn,idx,extent=None,mask_value=None,cmap=None,norm=None,fig=None,ax=None,**kwds):
    md = dm.managed_dataset(fn)
    v = md.get_mapping_var()
    shape = md.variables[v.name].shape
    md.close()
    p,f = os.path.split(fn)
    sfm = dm.SplitFileManager.open_existing(p,f,v)

    def get_data(idx,extent,sfm):
        if not extent:
            a = sfm[np.s_[idx,:]]
            return a,'slice',None
        else:
            try:
                s = [idx]
                s.extend(x for x in extent)
                a = sfm[np.s_[s]]
                return a,'slice',None
            except:
                gb = from_boundary_coords(*extent,compute_areas=False)
                print(gb.x_size,gb.y_size)
                a = sfm[np.s_[idx,slice(gb.x_min,gb.x_max),slice(gb.y_min,gb.y_max)]]
                extent = (gb.lon_min,gb.lon_max,gb.lat_min,gb.lat_max)
                return a,'geobounds',extent

    if type(idx) == int:
        if idx < 0:
            idx += shape[0]
        a,extent_type,extent = get_data(idx,extent,sfm)

    else: # idx is dti
        _idx = sfm.ref_ds.idx_for_period(idx)
        a,extent_type,extent = get_data(int(_idx[0]),extent,sfm)

    a = np.ma.masked_invalid(a)

    if mask_value is not None:
        a = np.ma.masked_equal(a, mask_value)
    print(a.min(),a.max(),a.mean(),a.shape)

    if fig is None:
        fig = plt.figure(figsize=(16,20))
    if ax is None:
        ax = fig.add_subplot(1,1,1)

    if cmap:
        kwds['cmap'] = cmap
    if norm:
        kwds['norm'] = norm
    if extent_type == 'geobounds':
        cax = ax.imshow(a,interpolation='none',extent=extent, **kwds)
    else:
        cax = ax.imshow(a,interpolation='none', **kwds)

    sfm.close_all()
    return fig,ax,cax

def plot_cell_timeseries(file_map,slice,ax=None,figsize=(20,10)):
    if not ax:
        ax=plt.figure(figsize=figsize).gca()
    for s in file_map:
        md = dm.managed_dataset(file_map[s])

        d = md.awra_var[slice]

        df = pd.DataFrame(d)
        df.plot(ax=ax)

    return ax

def get_histeq(data, nob=10, clip_pc=None, colors='Blues'):
    """
    get value boundaries for equalised stretch
    :param data: sorted 1D array of valid data
    :param nob:  number of boundaries
    :param clip_pc: percent saturation for upper and lower values
    :param colors:  valid matplotlib colour palette
    :return: norm,cmap
    """

    if clip_pc is None:
        inc = int(data.shape[0] / nob)
        bnd = data[::inc].tolist()

    else:
        pc = int(data.shape[0] / 100 * clip_pc)
        inc = int((data.shape[0] - 2 * pc) / nob)
        idx = [0]
        idx.extend([i for i in range(pc,data.shape[0] - pc,inc)])
        idx.append(-pc)
        idx.append(-1)
        bnd = data[idx].tolist()

    return get_cmap(bnd,colors)

def df_to_mpl(df,ax=None):
    """
    Plot a pandas dataframe as a matplotlib table
    :param df:
    :param ax:
    :return:
    """

    row_labels = list(df.index)
    col_labels = list(df.columns)
    cell_data = [["%.2f" % x for x in v] for k, v in df.iterrows()]

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    t = mpl.table.table(ax=ax,loc='center right',cellText=cell_data,colLabels = col_labels,rowLabels = row_labels)
    t.set_fontsize(10)
    return t

def plot_to_html(plot_func,**kwargs):
    """
    Encode the output of a matplotlib call to a base64 img for HTML display
    """

    bio = BytesIO()
    plot_func(**kwargs)

    plt.savefig(bio,bbox_inches='tight')
    plt.close()

    bio.flush()
    bio.seek(0)
    encoded_str = base64.b64encode(bio.read())
    img_str = """<img src="data:image/png;base64,""" + encoded_str.decode() + '"/>'
    return img_str

def get_cmap(bnd,colors='Blues'):
    norm = mpl_col.BoundaryNorm(bnd, len(bnd))
    cmap = plt.get_cmap(colors,lut=len(bnd)-1)
    return norm,cmap


def plot_pdf(filename,plot_cmd,**kwargs):
    pdf = PdfPages(filename)
    plot_cmd(**kwargs)
    pdf.savefig()
    plt.close()
    pdf.close()

def plot_array_table(arr,ax,scale=(1.0,1.0),**kwargs):
    '''
    Plot a numpy array as a table
    '''
    ax.axis('off')
    table = ax.table(cellText=arr,
                    cellLoc='center',
                    loc='center')
    table.scale(*scale)

    return table

def series_to_array(series):
    '''
    Convert a pandas series to a numpy array
    '''
    tarr = np.empty(shape=(len(series),2),dtype=object)
    tarr[:,1] = series.values
    tarr[:,0] = series.index
    return tarr    

def df_to_array(df,float_fmt=lambda x : '%.2f' % x):
    '''
    Convert a pandas dataframe to a numpy array
    '''
    tarr = np.empty(shape=np.array(df.values.shape)+(1.,1.),dtype=object)
    tarr[1:,1:] = df.values
    tarr[1:,0] = df.index
    tarr[0,1:] = df.columns
    tarr[0,0] = ''

    for x in range(tarr.shape[0]):
        for y in range(tarr.shape[1]):
            if isinstance(tarr[x,y],float):
                tarr[x,y] = float_fmt(tarr[x,y])

    return tarr

'''
Table Specific Functions
'''

def scale_width(table,shape):
    max_w = {}

    cells = table.get_celld()
    
    fig = table.get_figure()

    for col in range(shape[1]):
        max_w[col] = 0.
        for row in range(shape[0]):
            c = cells[(row,col)]
            cur_w = c.get_required_width(tight_layout.get_renderer(fig))
            if cur_w > max_w[col]:
                max_w[col] = cur_w
    
    for col in range(shape[1]):
        for row in range(shape[0]):
            c = cells[(row,col)]
            c.set_width(max_w[col])

MULTI_COL = {'index': 'left','columns': 'center','cells': 'right'}
SERIES = {'index': 'left', 'cells': 'right'}

def align_table(table,shape,rules=MULTI_COL):
    cells = table.get_celld()
    
    for row in range(shape[0]):
        c = cells[row,0]
        c._loc = rules['index']
        
    for row in range(shape[0]):
        for col in range(1,shape[1]):
            c = cells[row,col]
            c._loc = rules['cells']
            
    if 'columns' in rules:
        for col in range(1,shape[1]):
            c = cells[0,col]
            c._loc = rules['columns']
    
    
def get_wh(table,shape):
    '''
    Return the width and height (proportionally) of the specified table
    '''
    cells = table.get_celld()
    
    w,h = 0.,0.
    
    for row in range(shape[0]):
        c = cells[row,0]
        h += c._height
    for col in range(0,shape[1]):
        c = cells[0,col]
        w += c._width
        
    return w,h
