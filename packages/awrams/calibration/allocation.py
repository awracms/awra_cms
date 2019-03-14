from collections import OrderedDict
import numpy as np

def sortbins(bins):
    return OrderedDict(sorted(bins.items(),key=lambda x: x[1]['ncells'],reverse=True))

def sortalloc(alloc):
    return OrderedDict(sorted(alloc.items(),key=lambda x: sum(x[1]['cell_counts']),reverse=False))


def allocate_catchments_to_nodes(extent_map,max_nodes,n_cores,core_min=1,max_over=1.02):
    n_cores = n_cores * core_min

    n_cells = sum([e.cell_count for e in extent_map.values()])
    max_valid = int(np.ceil(n_cells / n_cores))

    if max_nodes > max_valid:
        max_nodes = max_valid

    n_nodes = max_nodes + 1
    cur_mn = 0
    while cur_mn < n_cores:
        n_nodes -= 1
        node_alloc, catch_node_map, _, _ = _allocate_catchments_to_nodes(extent_map,n_nodes,max_over)
        cur_mn = min([sum(node['cell_counts']) for node in node_alloc.values()])
        if n_nodes == 1:
            break
    return node_alloc,catch_node_map

def _allocate_catchments_to_nodes(extent_map,nnodes,max_over=1.02):
    '''
    Allocate catchments as specified in extent_map, over nnodes, splitting if necessary
    '''
    
    extent_map = OrderedDict(sorted(extent_map.items(),key=lambda x: x[1].cell_count))
    
    total_cells = sum([e.cell_count for e in extent_map.values()])
    min_cells = min([e.cell_count for e in extent_map.values()])
    max_cells = max([e.cell_count for e in extent_map.values()])
    ncatch = len(extent_map)

    target = int(np.ceil(total_cells/nnodes))
                             
    ccounts = [e.cell_count for e in extent_map.values()]

    # catchment data
    cvals = OrderedDict(zip([k for k in extent_map],[dict(start_cell=0,ncells=c) for c in ccounts]))

    rem = cvals.copy()

    node_alloc = OrderedDict([(n,dict(catchments=[],cell_counts=[])) for n in range(nnodes)])

    catch_node_map = dict([(e,[]) for e in extent_map])

    splits = []

    true_max = int(np.round(target*max_over))

    while len(rem) > 0:
        rem = sortbins(rem)
        catch_id, cur_catch = rem.popitem(False)

        node_id, node = node_alloc.popitem(False)

        pdict = dict(owns = cur_catch['start_cell'] == 0)

        if sum(node['cell_counts']) + cur_catch['ncells'] <= true_max:
            pdict['remote'] = not pdict['owns']

            node['catchments'].append((catch_id,cur_catch,pdict))
            node['cell_counts'].append(cur_catch['ncells']) # cell_count
        else:
            avail = target - sum(node['cell_counts'])
              
            split_catch = cur_catch.copy()

            cur_catch['ncells'] = avail        

            split_catch['start_cell'] = split_catch['start_cell'] + avail
            split_catch['ncells'] = split_catch['ncells'] - avail

            pdict['remote'] = True

            node['catchments'].append((catch_id,cur_catch,pdict))
            node['cell_counts'].append(avail)

            rem[catch_id] = split_catch

            splits.append(catch_id)

        catch_node_map[catch_id].append(node_id)

        node_alloc[node_id] = node

        node_alloc = sortalloc(node_alloc)

    tots = [sum(n['cell_counts']) for n in node_alloc.values()]
    return node_alloc, catch_node_map, splits, (tots[-1]-tots[0])/np.mean(tots)

def allocate_cells_to_workers(node_alloc,extent_map,nworkers):

    from awrams.utils.extents import split_extent
    
    subex_full = OrderedDict()
    
    for cid,specs,pdict in node_alloc['catchments']:
        if pdict['remote']:
            subex_full[cid] = split_extent(extent_map[cid],specs['ncells'],specs['start_cell']),specs['start_cell']
        else:
            subex_full[cid] = extent_map[cid],0
            
    total_cells = sum(node_alloc['cell_counts'])
    
    cpw = int(np.floor(total_cells/nworkers))
    excess = total_cells - cpw*nworkers
    
    worker_cell_avail = np.ones(nworkers,dtype=int) * cpw
    worker_cell_avail[0:excess] += 1
        
    cvals = [dict(cid=k,start_cell=c[1],ncells=c[0].cell_count) for k,c in subex_full.items()]
    worker_alloc = OrderedDict([(n,dict(catchments=[],cell_counts=[])) for n in range(nworkers)])
    
    cur_worker = 0
    
    while(len(cvals) > 0):
        catchment = cvals.pop(False)
        worker = worker_alloc[cur_worker]
        avail = worker_cell_avail[cur_worker] - sum(worker['cell_counts'])
        if catchment['ncells'] <= avail:
            worker['catchments'].append(catchment)
            worker['cell_counts'].append(catchment['ncells'])
        else:
            worker['catchments'].append(dict(cid=catchment['cid'],start_cell=catchment['start_cell'],ncells=avail))
            worker['cell_counts'].append(avail)
            cvals.insert(0,dict(cid=catchment['cid'],start_cell=catchment['start_cell'] + avail,ncells=catchment['ncells']-avail))
        if sum(worker['cell_counts']) == worker_cell_avail[cur_worker]:
            cur_worker += 1
    
    sizes = [sum(w['cell_counts']) for w in worker_alloc.values()]
    offsets = [0] + list(np.cumsum([sizes]))[:-1]
    
    for i, wval in worker_alloc.items():
        wval['size'] = sizes[i]
        wval['offset'] = offsets[i]

    worker_alloc = [w for w in worker_alloc.values() if sum(w['cell_counts']) > 0]
    
    return worker_alloc
        

def build_node_splits(node_alloc,catch_node_map,catch_id,scale=1):
    '''
    Determine the split sizes and offsets for the specified catchment
    '''
    catch_nodes = catch_node_map[catch_id]
    split_sizes = []
    split_offsets = []
    cur_offset = 0
    for n in catch_nodes:
        node_catch = node_alloc[n]['catchments']
        ncells = [c[1]['ncells'] for c in node_catch if c[0] == catch_id][0]
        split_sizes.append(ncells)
        split_offsets.append(cur_offset)
        cur_offset += ncells
        
    return np.array(split_sizes)*scale,np.array(split_offsets)*scale