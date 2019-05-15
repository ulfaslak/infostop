import infomap
import numpy as np


def euclidean(points_a, points_b, radians = False):
    """ 
    Calculate the euclidian distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.
    """
    
    distance = np.linalg.norm((points_a - points_b).reshape(-1,2),axis = 1)
    
    if points_a.ndim ==1:
        return distance[0]
    else:
        return distance


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
        
    

def general_pdist(points, distance_function = haversine):
    """ 
    Calculate the distance bewteen each pair in a set of points given a distance function.
    
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
        
    for idx in range(0, c-1):
        ref = points[idx]
        temp = distance_function(points[idx+1:c,:], ref, radians = False)
        #to be taken care of
        result[vec_idx:vec_idx+temp.shape[0]] = temp
        vec_idx += temp.shape[0]
    return result

def group_time_distance(coords, r_C, min_staying_time, max_staying_time, distance_function):
    """Group temporally adjacent points if they are closer than r_C.
    
    Input
    -----
        coords : array-like (shape=(N, 2) or shape=(N,3))
        r_C : number (critical radius)
        min_staying_time : int
        max_staying_time : int
        distance_function : function (used to compute distances)
    
    Output
    ------
        groups : list-of-list
            Each list is a group of points
    """
    groups = []
    
    if coords.shape[1] == 2:
        current_group = coords[0].reshape(1, 2)
        for coord in coords[1:]:
            
            # Compute distance to current group
            dist = distance_function(np.median(current_group, axis=0), coord)
        
            # Put in current group
            if dist <= r_C:
                current_group = np.vstack([current_group, coord])
            
            # Or start new group if dist is too large
            else:
                groups.append(current_group)
                current_group = coord.reshape(1, 2)

    else:
        current_group = coords[0].reshape(1, 3)
        for coord in coords[1:]:
            
            # Compute distance to current group
            dist = distance_function(np.median(current_group[:, :2], axis=0), coord[:2])
            time = current_group[-1, 2] - coord[2]
        
            # Put in current group
            if dist <= r_C and time <= max_staying_time:
                current_group = np.vstack([current_group, coord])
            
            # Or start new group if dist is too large or time criteria are not met
            else:
                if current_group.shape[0] == 1:
                    groups.append(current_group)
                elif current_group[-1, 2] - current_group[0, 2] > min_staying_time:
                    groups.append(current_group)
                else:
                    groups.extend(current_group.reshape(-1, 1, 3))
                current_group = coord.reshape(1, 3)

    # Add the last group
    groups.append(current_group)
    return groups

def get_stationary_events(groups, min_size=2):
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


def infomap_communities(nodes, edges):
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
    for id_, n in enumerate(nodes):  # Loop over nodes
        name_map_inverted[id_] = n
        name_map[n] = id_
        
    # Initiate two-level Infomap
    infomapSimple = infomap.Infomap("--two-level")
    network = infomapSimple.network()
    
    # Add nodes
    for n in nodes:
        network.addNode(name_map[n]).disown()

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

def distance_matrix(stop_events, distance_function=haversine):
    """Compute distance matrix between list of points.

    Input
    -----
        stop_events : array-like (shape=(N, 2))
        distance_function : function (used to compute distances)

    Output
    ------
        D : array-like (shape=(N, N))
    """
    c = stop_events.shape[0]
    D = np.zeros((c, c)) * np.nan
    D[np.triu_indices(c, 1)] = general_pdist(stop_events, distance_function)
    return D

def compute_intervals(coords, coord_labels, max_time_between, distance_function=haversine):
    
    
    """Compute stop and moves intervals from the list of labels.
    
    Input
    -----
        coords : array-like (shape=(N, 2) or shape=(N,3))
        coord_labels: list of integers

    Output
    ------
        intervals : array-like (shape=(N_intervals,4), location, start_time, end_time, latitude, longitude)
    
    """
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
        if (loc==loc_prec):
            t_end = time
            median_lat.append(lat)
            median_lon.append(lon)
            
            
        #if the location name has changed build the interval and reset values
        else:
            t_end = min([t_end+max_time_between,time])
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
