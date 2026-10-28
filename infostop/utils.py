import numpy as np
from sklearn.neighbors import BallTree

try:
    from scipy.spatial import QhullError
except ImportError:  # SciPy < 1.8
    from scipy.spatial.qhull import QhullError
from infomap import Infomap
from scipy.spatial import ConvexHull
from tqdm import tqdm


def pass_func(input, **kwargs):
    return input


def median_lat_lon(coords):
    """Coordinate-wise median of (lat, lon) points in degrees, also across the antimeridian.

    It is `np.median(coords, axis=0)`, unless the longitudes span more than 180 degrees.
    Then the points are taken to lie on both sides of longitude +-180: the negative
    longitudes are moved up by 360 for the median, and the median is moved back into
    (-180, 180].

    Input
    -----
        coords : array-like (shape=(N, 2))
            Latitudes in column 0 and longitudes in column 1, in degrees.

    Output
    ------
        median : numpy.array (shape=(2, ))
    """
    coords = np.asarray(coords)
    median = np.median(coords, axis=0)
    if len(coords) == 0:
        return median
    lon = coords[:, 1]
    if lon.max() - lon.min() > 180:
        median[1] = np.median(np.where(lon < 0, lon + 360, lon))
        if median[1] > 180:
            median[1] -= 360
    return median


def query_neighbors(coords, r2, distance_metric="haversine", weighted=False):
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
    if distance_metric == "haversine":
        coords = np.radians(coords)
        r2 = r2 / 6371000

    # Init tree
    tree = BallTree(coords, metric=distance_metric)

    # Query
    return tree.query_radius(coords, r=r2, return_distance=weighted)


def infomap_communities(
    node_idx_neighbors,
    node_idx_distances,
    counts,
    weight_exponent,
    distance_metric,
    verbose,
    num_trials=1,
):
    """Two-level partition of single-layer network with Infomap.

    Parameters
    ----------
        node_index_neighbors : array of arrays
            Example: `array([array([0]), array([1]), array([2]), ..., array([9997]),
       array([9998]), array([9999])], dtype=object)`.
        num_trials : int
            Number of Infomap runs from different random starts; the partition with the
            shortest codelength is kept. Defaults to 1 here; `Infostop` and
            `SpatialInfomap` default to 50.

    Returns
    -------
        out : dict (node-community hash map).
    """
    # Tracking
    if verbose:
        progress = tqdm
    else:
        progress = pass_func

    # Initiate  two-level Infomap
    network = Infomap(
        f"--two-level --num-trials {int(num_trials)}"
        + (" --silent" if not verbose else "")
    )

    # Add nodes (and reindex nodes because Infomap wants ranked indices)
    if verbose:
        print("    ... adding nodes:")
    name_map, name_map_inverse = {}, {}
    strength = {}  # weighted degree of each node, to number the modules by flow
    singleton_nodes = []
    infomap_idx = 0
    for n, neighbors in progress(
        enumerate(node_idx_neighbors), total=len(node_idx_neighbors)
    ):
        if len(neighbors) > 1:
            network.add_node(infomap_idx)
            name_map_inverse[infomap_idx] = n
            name_map[n] = infomap_idx
            infomap_idx += 1
        else:
            singleton_nodes.append(n)

    if verbose:
        print(
            f"    --> added {len(name_map)} nodes (found {len(singleton_nodes)} singleton nodes)"
        )

    # Add links
    if verbose:
        n_edges = 0
        print("    ... adding edges")

    if node_idx_distances is None:
        for node, neighbors in progress(
            enumerate(node_idx_neighbors), total=len(node_idx_neighbors)
        ):
            for neighbor in neighbors[neighbors > node]:
                weight = max(counts[node], counts[neighbor])
                network.add_link(name_map[node], name_map[neighbor], weight)
                strength[node] = strength.get(node, 0) + weight
                strength[neighbor] = strength.get(neighbor, 0) + weight
                if verbose:
                    n_edges += 1
    else:
        for node, (neighbors, distances) in progress(
            enumerate(zip(node_idx_neighbors, node_idx_distances)),
            total=len(node_idx_neighbors),
        ):
            for neighbor, distance in zip(
                neighbors[neighbors > node], distances[neighbors > node]
            ):
                if distance_metric == "haversine":
                    distance *= 6371000
                weight = max(counts[node], counts[neighbor]) * distance ** (
                    -weight_exponent
                )
                network.add_link(name_map[node], name_map[neighbor], weight)
                strength[node] = strength.get(node, 0) + weight
                strength[neighbor] = strength.get(neighbor, 0) + weight
                if verbose:
                    n_edges += 1

    if verbose:
        print(f"    --> added {n_edges} edges")

    # Run infomap
    if verbose:
        print("    ... running Infomap...", end=" ")
    if len(name_map) > 0:
        result = network.run()
        # Infomap >= 2.14 returns the result; later versions deprecate `network.modules`
        modules = network.modules if result is None else result.modules().items()
        # Convert to node-community dict format
        partition = dict(
            [(name_map_inverse[infomap_idx], module) for infomap_idx, module in modules]
        )
        partition = number_modules_by_flow(partition, strength)
        if verbose:
            print("done")
    else:
        partition = {}
        if verbose:
            print("skipped (no nodes)")

    return partition, singleton_nodes


def number_modules_by_flow(partition, strength):
    """Renumber modules 1, 2, ... by flow, largest first, keeping Infomap's order for ties.

    In an undirected network the flow of a module is proportional to the summed
    weighted degree of its nodes. Infomap numbers modules this way, except after
    several trials on some versions (2.9 to 2.15). Where Infomap already numbers by
    flow, this changes nothing. Flows are compared as rounded shares of the total,
    so that rounding noise in the sums does not break Infomap's order for ties.
    """
    flow = {}
    for node, module in partition.items():
        flow[module] = flow.get(module, 0) + strength[node]
    total = sum(flow.values()) or 1
    order = sorted(flow, key=lambda module: (-round(flow[module] / total, 12), module))
    number = {module: i + 1 for i, module in enumerate(order)}
    return {node: number[module] for node, module in partition.items()}


def label_network(
    node_idx_neighbors,
    node_idx_distances,
    counts,
    weight_exponent,
    label_singleton,
    distance_metric,
    verbose,
    num_trials=1,
):
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
            If True, give each node that has no neighbor within `r2` its own label. If False,
            label them as outliers (-1), unless their count is above 1.
        num_trials: int
            Number of Infomap runs; the partition with the shortest codelength is kept.
            Defaults to 1 here; `Infostop` and `SpatialInfomap` default to 50.

    Returns
    -------
        out : array-like (N, )
            Array of labels matching input in length. Detected stop locations get non-negative
            integer labels that carry no meaning beyond identity. Nodes with neighbors get
            Infomap's module ids, which start at 1. With `label_singleton=True`, nodes with no
            neighbor get the next labels after the largest module id, so if no node has a
            neighbor (and Infomap is not run), labels start at 0. If `label_singleton=False`,
            coordinates with no neighbors within distance `r2` and a count of 1 are labeled -1;
            those with a count above 1 get the next labels after the largest module id.
    """
    # Infer the partition with infomap. Partiton looks like `{node: community, ...}`
    partition, singleton_nodes = infomap_communities(
        node_idx_neighbors,
        node_idx_distances,
        counts,
        weight_exponent,
        distance_metric,
        verbose,
        num_trials,
    )

    # Without `label_singleton`, a singleton is still a stop location if its count is
    # above 1: several visits (or points) that share one coordinate
    if not label_singleton:
        singleton_nodes = [n for n in singleton_nodes if counts[n] > 1]

    # Add new labels to each singleton point (stop that was further than r2 from
    # any other point and thus was not represented in the network)
    max_label = max(partition.values(), default=-1)
    partition.update(
        dict(
            zip(
                singleton_nodes,
                range(max_label + 1, max_label + 1 + len(singleton_nodes)),
            )
        )
    )

    # Cast the partition as a vector of labels like `[0, 1, 0, 3, 0, 0, 2, ...]`
    labels = np.array(
        [partition[n] if n in partition else -1 for n in range(len(node_idx_neighbors))]
    )

    # Count the final stop locations: Infomap's modules and the labelled singletons
    if verbose:
        print(f"Found {len(np.unique(labels[labels != -1]))} stop locations")

    return labels


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
        return np.linalg.norm((points_a - points_b).reshape(-1, 2), axis=1)

    c = points.shape[0]
    result = np.zeros((c * (c - 1) // 2,), dtype=np.float64)
    vec_idx = 0
    for idx in range(0, c - 1):
        ref = points[idx]
        temp = _l2(points[idx + 1 : c, :], ref)
        # to be taken care of
        result[vec_idx : vec_idx + temp.shape[0]] = temp
        vec_idx += temp.shape[0]
    return max(result)


def convex_hull(points, to_return="points"):
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
        return np.vstack(
            [
                c + np.array([-l / 2, -l / 2]),  # bottom left
                c + np.array([l / 2, -l / 2]),  # bottom right
                c + np.array([l / 2, l / 2]),  # top right
                c + np.array([-l / 2, l / 2]),  # top right
            ]
        )
