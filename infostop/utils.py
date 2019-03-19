import infomap
import numpy as np

def haversine(points_a, points_b, radians=False):
    """ 
    Calculate the great-circle distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.

    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Using this because it is vectorized (stupid fast).
    """
    def _split_columns(array):
        if array.ndim == 1:
            return array[0], array[1] # just a single row
        else:
            return array[:,0], array[:,1]

    if radians:
        lat1, lon1 = _split_columns(points_a)
        lat2, lon2 = _split_columns(points_b)

    else:
    # convert all latitudes/longitudes from decimal degrees to radians
        lat1, lon1 = _split_columns(np.radians(points_a))
        lat2, lon2 = _split_columns(np.radians(points_b))

    # calculate haversine
    lat = lat2 - lat1
    lon = lon2 - lon1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2
    h = 2 * 6371e3 * np.arcsin(np.sqrt(d))
    return h  # in meters

def haversine_pdist(points, radians=False):
    """ 
    Calculate the great-circle distance bewteen each pair in a set of points.
    
    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Input
    -----
        points : array-like (shape=(N, 2))
            (lat, lon) in degree or radians (default is degree)

    Output
    ------
        result : array-like (shape=(N*(N-1)//2, ))
    """ 
    c = points.shape[0]
    result = np.zeros((c*(c-1)//2,), dtype=np.float64)
    vec_idx = 0
    if not radians:
        points = np.radians(points)
    for idx in range(0, c-1):
        ref = points[idx]
        temp = haversine(points[idx+1:c,:], ref, radians=True)
        result[vec_idx:vec_idx+temp.shape[0]] = temp
        vec_idx += temp.shape[0]
    return result

def group_time_distance(coords, r_C):
    """Group temporally adjacent points if they are closer than r_C.
    
    Input
    -----
        coords : array-like (shape=(N, 2))
        r_C : number (critical radius)
    
    Output
    ------
        groups : list-of-list
            Each list is a group of points
    """
    groups = []
    
    current_group = coords[0].reshape(-1, 2)
    for coord in coords[1:]:
        
        # Compute distance to current group
        dist = haversine(np.median(current_group, axis=0), coord)
    
        # Put in current group
        if dist <= r_C:
            current_group = np.vstack([current_group, coord])
        # Or start new group if dist is too large
        else:
            groups.append(current_group)
            current_group = coord.reshape(-1, 2)

    # Add the last group
    groups.append(current_group)

    return groups

def get_stationary_medoids(groups, min_size=1):
    """Convert groups of multiple points (stationary location events) to median-of-group points.
    
    Input
    -----
        groups : list-of-list
            Each list is a group of points
        min_size : int
            Minimum size of group to consider it stationary (default: 1)
            
    Output
    ------
        stat_coords : array-like (M, 2)
            Medioids of stationary groups
        mediod_map : list
            Group labels
            
    """
    stat_coords = np.empty(shape=(0, 2))
    medoid_map = []
    i = 0
    for g in groups:
        if g.shape[0] > min_size:
            stat_coords = np.vstack([stat_coords, np.median(g, axis=0).reshape(1, -1)])
            medoid_map.extend([i] * len(g)); i += 1
        else:
            medoid_map.append(-1)
     
    return stat_coords, np.array(medoid_map)


def infomap_communities(edges):
    """Two-level partition of single-layer network with Infomap.
    
    Input
    -----
        edges : list of tuples
            Example: `("cat", "dog", 1)` or (0, 1)

    Output
    ------
        out : dict (node-community hash map)
    """
    # Represent node names as indices
    name_map = {}
    name_map_inverted = {}
    for id_, n in enumerate(set([n for e in edges for n in e[:2]])):  # Loop over nodes
        name_map_inverted[id_] = n
        name_map[n] = id_
       
    # Initiate two-level Infomap
    infomapSimple = infomap.Infomap("--two-level")
    network = infomapSimple.network()
    
    # Add links (weighted)
    if len(edges[0]) == 2:
        for n1, n2 in edges:
            network.addLink(name_map[n1], name_map[n2], 1)

    # Add links (unweighted)
    if len(edges[0]) == 3:
        for n1, n2, w in edges:
            network.addLink(name_map[n1], name_map[n2], w)
    
    # Run infomap
    infomapSimple.run()
    
    # Return in node-community dictionary format
    return dict([
        (name_map_inverted[k], v)
        for k, v in infomapSimple.getModules().items()
    ])
