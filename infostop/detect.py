import numpy as np
import warnings
from infostop import utils
import cpputils
from tqdm import tqdm


def label_trace(coords, r1=10, r2=10, label_singleton=False, min_staying_time=300, max_time_between=86400, min_size=2, tolerance=0, distance_metric='haversine'):
    """Infer stop-location labels from mobility trace. Dynamic points are labeled -1.

    The method entils the following steps:
        1.  Detect which points are stationary and store only the median (lat, lon) of
            each stationarity event. A point belongs to a stationarity event if it is 
            less than `r1` meters away from the median of the time-previous collection
            of stationary points.
        2.  Construct a network that links nodes (event medians) that are within `r2` m.
        3.  Cluster this network using two-level Infomap.
        4.  Put the labels back info a vector that matches the input data in size.
    
    Parameters
    ----------
        coords : array-like (shape (N, 2)/(N, 3)) or list of arrays
            Columns 0 and 1 are reserved for lat and lon. Column 2 is reserved for time (any unit consistent with
            `min_staying_time` and `max_time_between`). If the input type is a list of arrays, each array is assumed
            to be the trace of a single user, in which case the obtained stop locations are shared by all users in
            the population.
        r1 : number
            Max distance between time-consecutive points to label them as stationary
        r2 : number
            Max distance between stationary points to form an edge.
        label_singleton: bool
            If True, give stationary locations that was only visited once their own
            label. If False, label them as outliers (-1)
        min_staying_time : int
            The shortest duration that can constitute a stop. Only used if timestamp column
            is provided
        max_time_between : int
            The longest duration that can constitute a stop. Only used if timestamp column
            is provided
        min_size : int
            Minimum size of group to consider it stationary (default: 2)
        tolerance : float ()
            The minimal difference allowed between points before they are considered the same points.
            Highly useful for downsampling data and dramatically reducing runtime. Higher values yields
            higher downsampling. For GPS data, it is not recommended to increase it beyond 1e-6.
        distance_metric: str
            Either 'haversine' (for GPS data) or 'euclidean'


    Returns
    -------
        out : array-like (N, )
            Array of labels matching input in length. Non-stationary locations and
            outliers (locations visited only once if `label_singleton == False`) are
            labeled as -1. Detected stop locations are labeled from 0 and up, and
            typically locations with more observations have lower indices.
    """

    # Format input
    multiuser = True
    if type(coords) != list:
        coords = [coords]
        multiuser = False
        
    # ASSERTIONS
    # ----------
    try:
        assert distance_metric in ['euclidean', 'haversine']
    except AssertionError:
        raise AssertionError("`distance_metric` should be either 'euclidean' or 'haversine'")
        
    for u, coords_u in enumerate(coords):
        error_insert = "" if not multiuser else f"User {u}: "
        try:
            assert coords_u.shape[1] in [2, 3]
        except AssertionError:
            raise AssertionError("%sNumber of columns must be 2 or 3" % error_insert)
        if coords_u.shape[1] == 3:
            try:
                assert np.all(coords_u[:-1, 2] <= coords_u[1:, 2])
            except AssertionError:
                raise AssertionError("%sTimestamps must be ordered" % error_insert)

        if distance_metric == 'haversine':
            try:
                assert np.min(coords_u[:, 0]) > -90
                assert np.max(coords_u[:, 0]) < 90
            except AssertionError:
                raise AssertionError("%sColumn 0 (latitude) must have values between -90 and 90" % error_insert)
            try:
                assert np.min(coords_u[:, 1]) > -180
                assert np.max(coords_u[:, 1]) < 180
            except AssertionError:
                raise AssertionError("%sColumn 1 (longitude) must have values between -180 and 180" % error_insert)


    # Time-group points
    stop_events, event_maps = [], []
    for coords_u in tqdm(coords):  ## DEBUG
        stop_events_u, event_map_u = cpputils.get_stationary_events(
            coords_u, r1, min_size, min_staying_time, max_time_between, distance_metric
        )
        stop_events.append(stop_events_u)
        event_maps.append(event_map_u)
    
    # Downsample
    coords = np.vstack(stop_events)
    if tolerance > 0:
        coords = np.around(coords / tolerance) * tolerance

    # Only keep unique coordinates for clustering
    coords, inverse_indices = np.unique(
        coords,
        return_inverse=True, axis=0
    )

    # Create network
    node_idx_neighbors = utils.build_network(coords, r2, distance_metric)
        
    # Create network and run infomap
    labels = label_network(node_idx_neighbors, label_singleton)

    # Add back the labels that were taken out when getting unique coords
    labels = labels[inverse_indices]
    
    # Label all the input points and return that label vector
    labels += [-1] # hack: make the last item -1, so when you index -1 you get -1 (HA!)
    coord_labels = []
    for j, event_map_u in enumerate(event_maps):
        i_min = sum([len(stop_events[j_]) for j_ in range(j)])
        coord_labels.append(
            np.array([labels[i + i_min] for i in event_map_u])
        )
    
    if multiuser:
        return coord_labels
    else:
        return coord_labels[0]

def label_static_points(coords, r2=10, label_singleton=True, distance_metric='haversine', method='ball_tree'):
    """Infer stop-location labels from static points.

    The method entils the following steps:
        1.  Construct a network that links nodes (event medians) that are within `r2`.
        2.  Cluster this network using two-level Infomap.

    Parameters
    ----------
        coords : array-like (N, 2)
        r2 : number
            Max distance between stationary points to form an edge.
        label_singleton: bool
            If True, give stationary locations that was only visited once their own
            label. If False, label them as outliers (-1)
        distance_metric: str
            Either 'haversine' (for GPS data) or 'euclidean'
            
    Returns
    -------
        out : array-like (N, )
            Array of labels matching input in length. Detected stop locations are labeled from 0
            and up, and typically locations with more observations have lower indices. If
            `label_singleton=False`, coordinated with no neighbors within distance `r2` are
            labeled -1.
    """

    # ASSERTIONS
    # ----------
            
    if distance_metric == "haversine":
        try:
            assert coords.shape[1] == 2
        except AssertionError:
            raise AssertionError("Number of columns must be 2")
        try:
            assert np.min(coords[:, 0]) > -90
            assert np.max(coords[:, 0]) < 90
        except AssertionError:
            raise AssertionError("Column 0 (latitude) must have values between -90 and 90")
        try:
            assert np.min(coords[:, 1]) > -180
            assert np.max(coords[:, 1]) < 180
        except AssertionError:
            raise AssertionError("Column 1 (longitude) must have values between -180 and 180")

    # Create network
    node_idx_neighbors = utils.build_network(coords, r2, distance_metric)

    # Create network and run infomap
    return label_network(node_idx_neighbors, label_singleton)

@utils.timeit
def label_network(node_idx_neighbors, label_singleton=True):
    """Infer infomap clusters from distance matrix and link distance threshold.
    
    Parameters
    ----------
        nodes: array 
            Nodes in the network
        edges: array
            Edges in the network (two nodes are connected if distance<r2)
        singleton_nodes: array
            Non connected nodes.
        label_singleton: bool
            If True, give stationary locations that was only visited once their own
            label. If False, label them as outliers (-1)
            
    Returns
    -------
        out : array-like (N, )
            Array of labels matching input in length. Detected stop locations are labeled from 0
            and up, and typically locations with more observations have lower indices. If
            `label_singleton=False`, coordinates with no neighbors within distance `r2` are
            labeled -1.
    """
        
    # Infer the partition with infomap. Partiton looks like `{node: community, ...}`
    partition, singleton_nodes = utils.infomap_communities(node_idx_neighbors)
    
    # Add new labels to each singleton point (stop that was further than r2 from
    # any other point and thus was not represented in the network)
    if label_singleton:
        max_label = max(partition.values())
        partition.update(dict(zip(
            singleton_nodes,
            range(max_label+1, max_label+1+len(singleton_nodes))
        )))
    
    # Cast the partition as a vector of labels like `[0, 1, 0, 3, 0, 0, 2, ...]`
    return np.array([
        partition[n] if n in partition else -1
        for n in range(len(node_idx_neighbors))
    ])

def get_stationary_events(coords, r1, min_size, min_staying_time, max_time_between, distance_metric):
    """Reduce location trace to the sequence of stationary events.

    (NOT IN USE: REPLACED BY `cpputils.get_stationary_events`)

    Parameters
    ----------
        coords : array-like (shape=(N, 2))
        r1 : number (critical radius)
        min_size : int
        min_staying_time : int
        max_time_between : int
        distance_metric : str

    Returns
    -------
        stop_events : np.array (<N, 2)
        event_map : list (N, )
            Maps index to input-data indices.
    """
    groups = utils.group_time_distance(coords, r1, min_staying_time, max_time_between, distance_metric)
    stop_events, event_map = utils.reduce_groups(groups, min_size)
    return stop_events, event_map