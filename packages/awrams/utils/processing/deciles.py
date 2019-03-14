import numpy as np

def get_percentiles_of(data,ref_data,pctiles=None):
    '''
    Given a reference set of sorted data <ref_data>, return the percentile scores of <data>
    '''

    data = data.copy()

    if hasattr(data,'mask'):
        in_mask = data.mask
        data = data.data
    else:
        in_mask = np.zeros(data.shape[-2:],dtype=bool)

    if hasattr(ref_data,'mask'):
        ref_data = ref_data.data

    CLIPL = data<ref_data[0]
    CLIPH = data>ref_data[-1]
    data[CLIPL] = ref_data[0][CLIPL]
    data[CLIPH] = ref_data[-1][CLIPH]

    NPCTILES = ref_data.shape[0]
    if pctiles is None:
        pctiles = np.linspace(0,1.,NPCTILES)

    indices = np.indices(data.shape)

    LIDX = NPCTILES-(data <= ref_data).sum(0)
    LIDX = np.where(LIDX == ref_data.shape[0],ref_data.shape[0] - 1,LIDX)
    UIDX = (data >= ref_data).sum(0) - 1
    LIDX,UIDX = np.minimum(LIDX,UIDX),np.maximum(LIDX,UIDX)

    if indices.ndim == 2:
        upper = ref_data[UIDX,indices[0]]
        lower = ref_data[LIDX,indices[0]]
    elif indices.ndim == 3:
        upper = ref_data[UIDX,indices[0],indices[1]]
        lower = ref_data[LIDX,indices[0],indices[1]]

    lurange = upper-lower
    abs_diffs = data-lower
    rel_diffs = abs_diffs/lurange
    rel_diffs[lurange==0.]=0.5
    final=pctiles[UIDX]*rel_diffs+pctiles[LIDX]*(1.0-rel_diffs)

    final[CLIPL.reshape(final.shape)] = 0.
    final[CLIPH.reshape(final.shape)] = 1.

    out_arr = np.ma.masked_array(data=final)    
    out_arr.mask=in_mask
    return out_arr
