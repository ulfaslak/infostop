import numpy as np
from sklearn.neighbors import BallTree
from scipy.spatial.qhull import QhullError
from infomap import Infomap
from scipy.spatial import ConvexHull
from tqdm import tqdm

def pass_func(input, **kwargs):
    return input

def query_neighbors(coords, r2, distance_metric='haversine', weighted=False):
    """Build a network from a set of points and a threshold distance.
    
    Parameters
    ----------
        coords : array-like (N, 2)
        r2 : float
            Threshold distance.
        distance_metric : str
            Either 'haversine' or None.
    
    Returns
    -------
        nodes : list of ints
            Correspond to the list of nodes
        edges : list of tuples
            An edge between two nodes exist if they are closer than r2.
        singleton nodes : list of ints
            Nodes that have no connections, e.g. have been visited once.
    """
    
    # If the metric is haversine update points (to radians) and r2 accordingly.
    if distance_metric == 'haversine':
        coords = np.radians(coords)
        r2 = r2 / 6371000

    # Init tree
    tree = BallTree(coords, metric=distance_metric)

    # Query
    return tree.query_radius(coords, r=r2, return_distance=weighted)


def infomap_communities(node_idx_neighbors, node_idx_distances, counts, weight_exponent, distance_metric, verbose):
    """Two-level partition of single-layer network with Infomap.
    
    Parameters
    ----------
        node_index_neighbors : array of arrays
            Example: `array([array([0]), array([1]), array([2]), ..., array([9997]),
       array([9998]), array([9999])], dtype=object)`.

    Returns
    -------
        out : dict (node-community hash map).
    """
    # Tracking 
    if verbose: progress = tqdm
    else:       progress = pass_func

    # Initiate  two-level Infomap
    network = Infomap("--two-level" + (" --silent" if not verbose else ""))

    # Add nodes (and reindex nodes because Infomap wants ranked indices)
    if verbose: print("    ... adding nodes:")
    name_map, name_map_inverse = {}, {}
    singleton_nodes = []
    infomap_idx = 0
    for n, neighbors in progress(enumerate(node_idx_neighbors), total=len(node_idx_neighbors)):
        if len(neighbors) > 1:
            network.addNode(infomap_idx)
            name_map_inverse[infomap_idx] = n
            name_map[n] = infomap_idx
            infomap_idx += 1
        else:
            singleton_nodes.append(n)

    if verbose:
        print(f"    --> added {len(name_map)} nodes (found {len(singleton_nodes)} singleton nodes)")

    
    # Add links
    if verbose:
        n_edges = 0
        print("    ... adding edges")

    if node_idx_distances is None:
        for node, neighbors in progress(enumerate(node_idx_neighbors), total=len(node_idx_neighbors)):
            for neighbor in neighbors[neighbors > node]:
                network.addLink(name_map[node], name_map[neighbor], max(counts[node], counts[neighbor]))
                if verbose: n_edges += 1
    else:
        for node, (neighbors, distances) in progress(enumerate(zip(node_idx_neighbors, node_idx_distances)), total=len(node_idx_neighbors)):
            for neighbor, distance in zip(neighbors[neighbors > node], distances[neighbors > node]):
                if distance_metric == "haversine":
                    distance *= 6371000
                network.addLink(name_map[node], name_map[neighbor], max(counts[node], counts[neighbor]) * distance**(-weight_exponent))
                if verbose: n_edges += 1
    
    if verbose:
        print(f"    --> added {n_edges} edges")

    # Run infomap
    if verbose: print("    ... running Infomap...", end=" ")
    if len(name_map) > 0:
        network.run()
    	# Convert to node-community dict format
        partition = dict([
            (name_map_inverse[infomap_idx], module)
            for infomap_idx, module in network.modules
        ])
        if verbose: print("done")
    else:
        partition = {}


    if verbose:
        print(f"Found {len(set(partition.values()))-1} stop locations")

    return partition, singleton_nodes


def label_network(node_idx_neighbors, node_idx_distances, counts, weight_exponent, label_singleton, distance_metric, verbose):
    """Infer infomap clusters from distance matrix and link distance threshold.
    
    Parameters
    ----------
        nodes: array 
            Nodes in the network.
        edges: array
            Edges in the network (two nodes are connected if distance<r2).
        singleton_nodes: array
            Non connected nodes.
        label_singleton: bool
            If True, give stationary locations that was only visited once their own
            label. If False, label them as outliers (-1).
            
    Returns
    -------
        out : array-like (N, )
            Array of labels matching input in length. Detected stop locations are labeled from 0
            and up, and typically locations with more observations have lower indices. If
            `label_singleton=False`, coordinates with no neighbors within distance `r2` are
            labeled -1.
    """ 
    # Infer the partition with infomap. Partiton looks like `{node: community, ...}`
    partition, singleton_nodes = infomap_communities(node_idx_neighbors, node_idx_distances, counts, weight_exponent, distance_metric, verbose)
    
    # Add new labels to each singleton point (stop that was further than r2 from
    # any other point and thus was not represented in the network)
    if label_singleton:
        max_label = max(partition.values(), default=-1)
        partition.update(dict(zip(
            singleton_nodes,
            range(max_label+1, max_label+1+len(singleton_nodes))
        )))
    
    # Cast the partition as a vector of labels like `[0, 1, 0, 3, 0, 0, 2, ...]`
    return np.array([
        partition[n] if n in partition else -1
        for n in range(len(node_idx_neighbors))
    ])

def max_pdist(points):
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
    def _l2(points_a, points_b):
        return np.linalg.norm((points_a - points_b).reshape(-1,2),axis = 1)

    c = points.shape[0]
    result = np.zeros((c*(c-1)//2,), dtype=np.float64)
    vec_idx = 0
    for idx in range(0, c-1):
        ref = points[idx]
        temp = _l2(points[idx+1:c, :], ref)
        #to be taken care of
        result[vec_idx:vec_idx+temp.shape[0]] = temp
        vec_idx += temp.shape[0]
    return max(result)

def convex_hull(points, to_return='points'):
    """Return the convex hull of a collection of points."""
    try:
        hull = ConvexHull(points)
        return points[hull.vertices, :]
    except QhullError:
        c = points.mean(0)
        if points.shape[0] == 1:
            l = 5e-5
        else:
            l = max_pdist(points)
        return np.vstack([
            c + np.array([-l/2, -l/2]),  # bottom left
            c + np.array([l/2, -l/2]),   # bottom right
            c + np.array([l/2, l/2]),    # top right
            c + np.array([-l/2, l/2]),    # top right
        ])
