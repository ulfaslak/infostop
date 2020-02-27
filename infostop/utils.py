import infomap
import numpy as np
from math import radians
from sklearn.neighbors import BallTree
import time

## DEBUG
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed



def euclidean(points_a, points_b):
    """ 
    Calculate the euclidian distance bewteen points_a and points_b.
    """
    return np.linalg.norm((points_a - points_b), axis=0)

def haversine(points_a, points_b):
    """ 
    Calculate the great-circle distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.
    """
    def _split_columns(array):
        return array[0], array[1]
    lat1, lon1 = _split_columns(np.radians(points_a))
    lat2, lon2 = _split_columns(np.radians(points_b))
    lat = lat2 - lat1
    lon = lon2 - lon1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2
    h = 2 * 6371e3 * np.arcsin(np.sqrt(d))
    return h  # in meters

@timeit
def build_network(coords, r2, distance_metric='haversine'):
    """Build a network from a set of points and a threshold distance.
    
    Parameters
    ----------
        coords : array-like (N, 2)
        r2 : float
            Threshold distance
        distance_metric : str
            Either 'haversine' or None
    
    Returns
    -------
        nodes : list of ints
            Correspond to the list of nodes
        edges : list of tuples
            An edge between two nodes exist if they are closer than r2
        singleton nodes : list of ints
            Nodes that have no connections, e.g. have been visited once
    """
    
    # If the metric is haversine update points (to radians) and r2 accordingly.
    if distance_metric == 'haversine':
        coords = np.radians(coords)
        r2 = r2 / 6371000

    # Build and query the tree
    tree = BallTree(coords, metric=distance_metric)

    return tree.query_radius(coords, r=r2, return_distance=False)

def group_time_distance(coords, r_C, min_staying_time, max_staying_time, distance_func):
    """Group temporally adjacent points if they are closer than r_C.
    
    (NOT IN USE: REPLACED BY `cpputils.get_stationary_events`)

    Parameters
    ----------
        coords : array-like (shape=(N, 2) or shape=(N,3))
        r_C : number (critical radius)
        min_staying_time : int
        max_staying_time : int
        distance_metric : str
    
    Returns
    -------
        groups : list-of-list
            Each list is a group of points
    """
    groups = []
    
    # No time information
    if coords.shape[1] == 2:
        current_group = coords[0].reshape(1, 2)
        for coord in coords[1:]:
            
            # Compute distance to current group
            dist = distance_func(np.median(current_group, axis=0), coord)
        
            # Put in current group
            if dist <= r_C:
                current_group = np.vstack([current_group, coord])
            
            # Or start new group if dist is too large
            else:
                groups.append(current_group)
                current_group = coord.reshape(1, 2)
        
        # Append the last group
        groups.append(current_group)

    # With time information
    else:
        current_group = coords[0].reshape(1, 3)
        for coord in coords[1:]:
            
            # Compute distance to current group
            dist = distance_func(np.median(current_group[:, :2], axis=0), coord[:2])
            time = current_group[-1, 2] - coord[2]
        
            # Put in current group
            if dist <= r_C and time <= max_staying_time:
                current_group = np.vstack([current_group, coord])
            
            # Or start new group if dist is too large or time criteria are not met
            else:
                if current_group.shape[0] > 1 and current_group[-1, 2] - current_group[0, 2] < min_staying_time:
                    groups.extend(current_group.reshape(-1, 1, 3))
                else:
                    groups.append(current_group)
                    
                current_group = coord.reshape(1, 3)
                
        # Append the last group
        if current_group.shape[0] > 1 and current_group[-1, 2] - current_group[0, 2] < min_staying_time:
            groups.extend(current_group.reshape(-1, 1, 3))
        else:
            groups.append(current_group)
            
    return groups

@timeit
def reduce_groups(groups, min_size=2):
    """Convert groups of multiple points (stationary location events) to median-of-group points.
    
    (NOT IN USE: REPLACED BY `cpputils.get_stationary_events`)
    
    Parameters
    ----------
        groups : list-of-list
            Each list is a group of points
        min_size : int
            Minimum size of group to consider it stationary (default: 1)
            
    Returns
    -------
        stat_coords : array-like (M, 2)
            Medioids of stationary groups
        event_map : list
            Maps event index to input-data index. Used for mapping label ids onto each (lat, lon) point.
    """
    stat_coords = np.empty(shape=(0, 2))
    event_map = []
    i = 0
    for g in groups:
        if g.shape[0] >= min_size:
            stat_coords = np.vstack([stat_coords, np.median(g[:, :2], axis=0).reshape(1, -1)])
            event_map.extend([i] * len(g))
            i += 1
        else:
            event_map.extend([-1] * len(g))
     
    return stat_coords, np.array(event_map)


def infomap_communities(node_idx_neighbors):
    """Two-level partition of single-layer network with Infomap.
    
    Parameters
    ----------
        node_index_neighbors : array of arrays
            Example: `array([array([0]), array([1]), array([2]), ..., array([9997]),
       array([9998]), array([9999])], dtype=object)

    Returns
    -------
        out : dict (node-community hash map)
    """
    # Infomap wants ranked indices
    name_map = {}
    name_map_inverted = {}
    singleton_nodes = []
    infomap_idx = 0
    for n, neighbors in enumerate(node_idx_neighbors):  # Loop over nodes
        if len(neighbors) > 1:
            name_map_inverted[infomap_idx] = n
            name_map[n] = infomap_idx
            infomap_idx += 1
        else:
            singleton_nodes.append(n)

    # Raise exception is network is too sparse.
    if len(name_map) == 0:
        raise Exception("No edges added because `r2` < the smallest distance between any two points.")

    # Initiate two-level Infomap
    infomapSimple = infomap.Infomap("--two-level")
    network = infomapSimple.network()
    
    # Add nodes
    n_edges = 0
    for n, neighbors in enumerate(node_idx_neighbors):
        if len(neighbors) > 1:
            network.addNode(name_map[n]).disown()
            n_edges += 1

    print(n_edges)

    # Add links
    for node, neighbors in enumerate(node_idx_neighbors):
        for neighbor in neighbors:
            if neighbor > node:
                network.addLink(name_map[node], name_map[neighbor], 1)
    
    # Run infomap
    infomapSimple.run()
    
    # Convert to node-community dict format
    partition = dict([
        (name_map_inverted[infomap_idx], module)
        for infomap_idx, module in infomapSimple.getModules().items()
    ])

    return partition, singleton_nodes


def compute_intervals(coords, coord_labels, max_time_between=86400, distance_metric="haversine"):
    """Compute stop and moves intervals from the list of labels.
    
    Parameters
    ----------
        coords : array-like (shape=(N, 2) or shape=(N,3))
        coord_labels: list of integers

    Returns
    -------
        intervals : array-like (shape=(N_intervals,4), location, start_time, end_time, latitude, longitude)
    
    """
    
    if coords.shape[1] == 2:
        times = np.array(list(range(0, len(coords))))
        coords = np.hstack([coords, times.reshape(-1,1)])
        
    trajectory = np.hstack([coords, coord_labels.reshape(-1,1)])
    
    final_trajectory = []
    
    #initialize values
    lat_prec, lon_prec, t_start, loc_prec = trajectory[0]  
    t_end = t_start 
    median_lat = [lat_prec]
    median_lon = [lon_prec]

    #Loop through trajectory
    for lat, lon, time, loc in trajectory[1:]:
        
        #if the location name has not changed update the end of the interval
        if (loc==loc_prec) and (time-t_end)<(max_time_between):
            t_end = time
            median_lat.append(lat)
            median_lon.append(lon)
            
            
        #if the location name has changed build the interval and reset values
        else:
            if loc_prec==-1:
                final_trajectory.append([loc_prec, t_start,  t_end, np.nan, np.nan])
            else:
                final_trajectory.append([loc_prec, t_start,  t_end, np.median(median_lat), np.median(median_lon)])
                
            t_start = time 
            t_end = time 
            median_lat = []
            median_lon = []
            
        
        #update current values
        loc_prec = loc
        lat_prec = lat
        lon_prec = lon
        
    #Add last group
    if loc_prec==-1:
        final_trajectory.append([loc_prec, t_start,  t_end, np.nan, np.nan])
    else:
        final_trajectory.append([loc_prec, t_start,  t_end, np.median(median_lat), np.median(median_lon)])

        
    return final_trajectory