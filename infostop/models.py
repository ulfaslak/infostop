import numpy as np
import cpputils
from infostop import utils
from tqdm import tqdm
 

class Infostop:
    """Infer stop-location labels from mobility trace. Dynamic points are labeled -1.

    Note: currently ONLY works for 2-dimensional spatial data. Data columns 0 and 1 are
    always intepreted as spatial locations while column 2 is interpreted as time.

    The method entils the following steps:
        1.  Sequential downsampling: Detect which points are stationary and store only
            the median of each stationarity event. A point belongs to a stationarity
            event if it is less than `r1` distance units away from the median of the
            time-previous collection of stationary points.
        2.  Spatial downsampling: Remove duplicate coordinates. Optionally round off
            coordinates before removing duplicates, to increase the downsampling.
        3.  For each coordinate, find its neighbors within `r2`.
        4.  Create Infomap network and optimize clusters in two-levels.
        5.  Reverse downsampling from step 2.
        6.  Reverse downsampling from step 1.
    
    Parameters
    ----------
        r1 : int/float
            Max distance between time-consecutive points to label them as stationary
        r2 : int/float
            Max distance between stationary points to form an edge.
        label_singleton: bool
            If True, give stationary locations that was only visited once their own label. If False, 
            label them as non-stationary (-1).
        min_staying_time : int
            The shortest duration that can constitute a stop.
        max_time_between : int
            The longest duration that can constitute a stop.
        min_size : int
            Minimum size of group to consider it stationary
        min_spacial_resolution : float
            The minimal difference allowed between points before they are considered the same points.
            Highly useful for spatially downsampling data and which dramatically reduces runtime. Higher
            values yields higher downsampling. For geo data, it is not recommended to increase it beyond
            1e-4. 1e-5, typically works well and has little impact on results.
        distance_metric : str
            Either 'haversine' (for geo data) or 'euclidean'
        weighted : bool
            Weight edges in the network representation by distance. Specifically, the weight used is
            `1 / distance(a, b) ** weight_exponent`.
        weight_exponent : float
            Exponent used when weighting edges in the network. Higher values increases the link strength
            between nearby locations yielding smaller stop locations. It is not recommended to use a negative
            `weight_exponent`, unless for a special purpose.
        verbose : bool
            Print output during the fitting procedure. Mostly for development and debugging. Option will
            be removed in future versions.

    Returns
    -------
        self : object

    Example
    -------
        >>> model = Infostop()
        >>> labels = model.fit_predict(traces)
    """
    def __init__(
        self,
        r1=10,
        r2=10,
        label_singleton=True,
        min_staying_time=300,
        max_time_between=86400,
        min_size=2,
        min_spacial_resolution=0,
        distance_metric='haversine',
        weighted=False,
        weight_exponent=1,
        verbose=False
    ):
        # Set input parameters as attributes
        self.r1 = r1
        self.r2 = r2
        self.label_singleton = label_singleton
        self.min_staying_time = min_staying_time
        self.max_time_between = max_time_between
        self.min_size = min_size
        self.min_spacial_resolution = min_spacial_resolution
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.weight_exponent = weight_exponent
        self.verbose = verbose
        self.data = None
        self.stat_coords = None
        self.stat_labels = None
        self.counts = None
        self.labels = None
        self.is_fitted = False

        # Run hyper parameter assertions
        self._hyperparam_assertions()


    def fit_predict(self, data):
        """Fit Infostop on one or more location sequnces, and return labels.

        Parameters
        ----------
            data : numpy.array (shape (N, 2)/(N, 3)) or list of such numpy.arrays
                Columns 0 and 1 are reserved for lat and lon. Column 2 is reserved for time (any unit consistent with
                `min_staying_time` and `max_time_between`). If the input type is a list of arrays, each array is assumed
                to be the trace of a single user, in which case the obtained stop locations are shared by all users in
                the population.

        Returns
        -------
            coord_labels : 1d numpy.array or list of such

        Example
        -------
            >>> model = Infostop()
            >>> labels = model.fit_predict(traces)
            >>> assert type(traces) == type(labels)
        """
        
        if self.verbose: progress = tqdm
        else:            progress = utils.pass_func

        # Infer multiuser mode
        self.multiuser = True
        if type(data) != list:
            self.data = [data]
            self.multiuser = False
            progress = utils.pass_func  # no need to log progress in (1) if there's just one user
        else:
            self.data = data
            if len(data) == 1:
                progress = utils.pass_func

        if self.verbose:
            print('Multiuser input:', self.multiuser)

        # Assert the input data
        self._data_assertions(self.data)

        # (1) Sequential downsampling: group time-adjacent points
        if self.verbose:
            avg_reduction = []
            print("Downsampling in time: keeping medians of stationary events")

        stop_events, event_maps = [], []
        for u, coords_u in progress(enumerate(self.data), total=len(self.data)):
            stop_events_u, event_map_u = cpputils.get_stationary_events(
                coords_u, self.r1, self.min_size, self.min_staying_time,
                self.max_time_between, self.distance_metric
            )

            if self.verbose:
                avg_reduction.append((1 - len(stop_events_u) / len(coords_u)) * 100)

            stop_events.append(stop_events_u)
            event_maps.append(event_map_u)

        if self.verbose:
            print("    --> %sreduction was %.1f%%" % ("average " if self.multiuser else "", np.mean(avg_reduction)))
        
        # Merge `stop_events` from different users into `stat_coords`
        try:
            self.stat_coords = np.vstack([se for se in stop_events if len(se) > 0])
        except ValueError:
            raise Exception("No stop events found. Check that `r1`, `min_staying_time` and `min_size` parameters are chosen correctly.")

        # (2) Downsample (dramatically reduces computation time)
        if self.min_spacial_resolution > 0:
            self.stat_coords = np.around(self.stat_coords / self.min_spacial_resolution) * self.min_spacial_resolution

        if self.verbose:
            num_stat_orig = len(self.stat_coords)
            print(f"Downsampling {num_stat_orig} total stop events to...", end=" ")

        # Only keep unique coordinates for clustering
        self.stat_coords, inverse_indices, self.counts = np.unique(
            self.stat_coords,
            return_inverse=True, return_counts=True, axis=0
        )

        if self.verbose:
            print(f"{len(self.stat_coords)}", end=" ")
            print("(%.1f%% duplicates)" % ((1 - len(self.stat_coords)/num_stat_orig)*100))

        # (3) Find neighbors within `r2` for each point
        if self.verbose:
            print("Finding neighbors...", end=" ")
        
        ball_tree_result = utils.query_neighbors(self.stat_coords, self.r2, self.distance_metric, self.weighted)
        
        if self.weighted:
            node_idx_neighbors, node_idx_distances = ball_tree_result
        else:
            node_idx_neighbors, node_idx_distances = ball_tree_result, None

        if self.verbose:
            print("done")
            
        # (4) Create network and run infomap
        if self.verbose: print("Creating network and clustering with Infomap...")
        self.stat_labels = utils.label_network(
            node_idx_neighbors, node_idx_distances, self.counts, self.weight_exponent,
            self.label_singleton, self.distance_metric, self.verbose
        )

        # (5) Reverse the downsampling in step (2)
        self.labels = self.stat_labels[inverse_indices]
        
        # (6) Reverse the downsampling in step (1)
        self.coord_labels = []
        for j, event_map_u in enumerate(event_maps):
            i0 = sum([len(stop_events[j_]) for j_ in range(j)])
            i1 = sum([len(stop_events[j_]) for j_ in range(j+1)])
            labels_u = np.hstack([self.labels[i0:i1], -1])
            self.coord_labels.append(labels_u[event_map_u])

        # Update model state and return labels
        self.is_fitted = True
        if self.multiuser:
            return self.coord_labels
        else:
            return self.coord_labels[0]


    def _hyperparam_assertions(self):
        assert self.r1 > 0, \
               "`r1` must be > 0"
        assert self.r2 > 0, \
               "`r2` must be > 0"
        assert type(self.label_singleton) is bool, \
               "`label_singleton` either `True` or `False`"
        assert self.min_staying_time > 0, \
               "`min_staying_time` must be > 0"
        assert self.max_time_between > 0, \
               "`max_time_between` must be > 0"
        assert self.max_time_between > self.min_staying_time, \
               "`max_time_between` must be > min_staying_time"
        assert self.min_size > 1, \
               "`min_size` must be > 1"
        assert 0 <= self.min_spacial_resolution <= 1, \
               "`min_spacial_resolution` must be within [0, 1]"
        assert self.distance_metric in ['euclidean', 'haversine'], \
               "`distance_metric` should be either 'euclidean' or 'haversine'"


    def _data_assertions(self, data):
        assert not np.any(np.isnan(np.vstack(data))), \
               f"There are {np.isnan(np.vstack(data))} NaN values in the input data."
        for u, coords_u in enumerate(data):
            error_insert = "" if not self.multiuser else f"User {u}: "
            assert coords_u.shape[1] in [2, 3], \
                   "%sNumber of columns must be 2 or 3" % error_insert
            if coords_u.shape[1] == 3:
                assert np.all(coords_u[:-1, 2] <= coords_u[1:, 2]), \
                       "%sTimestamps must be ordered" % error_insert
            if self.distance_metric == 'haversine':
                assert (np.min(coords_u[:, 0]) > -90 and np.max(coords_u[:, 0]) < 90), \
                       "%sLatitude (column 0) must have values between -90 and 90" % error_insert
                assert (np.min(coords_u[:, 1]) > -180 and np.max(coords_u[:, 1]) < 180), \
                       "%sLongitude (column 1) must have values between -180 and 180" % error_insert


class SpatialInfomap:
    """Cluster a collection of points using Infomap.

    The method entils the following steps:
        1.  Spatial downsampling: Remove duplicate coordinates. Optionally round off
            coordinates before removing duplicates, to increase the downsampling.
        2.  For each coordinate, find its neighbors within `r2`.
        3.  Create Infomap network and optimize clusters in two-levels.
        4.  Reverse downsampling from step 1.
    
    Parameters
    ----------
        r2 : int/float
            Max distance between stationary points to form an edge.
        label_singleton: bool
            If True, give stationary locations that was only visited once their own label. If False, 
            label them as non-stationary (-1).
        min_spacial_resolution : float
            The minimal difference allowed between points before they are considered the same points.
            Highly useful for spatially downsampling data and which dramatically reduces runtime. Higher
            values yields higher downsampling. For geo data, it is not recommended to increase it beyond
            1e-4. 1e-5, typically works well and has little impact on results.
        distance_metric : str
            Either 'haversine' (for geo data) or 'euclidean'
        weighted : bool
            Weight edges in the network representation by distance. Specifically, the weight used is
            `1 / distance(a, b) ** weight_exponent`.
        weight_exponent : float
            Exponent used when weighting edges in the network. Higher values increases the link strength
            between nearby locations yielding smaller stop locations. It is not recommended to use a negative
            `weight_exponent`, unless for a special purpose.
        verbose : bool
            Print output during the fitting procedure. Mostly for development and debugging. Option will
            be removed in future versions.

    Returns
    -------
        self : object

    Example
    -------
        >>> model = SpacialInfomap()
        >>> labels = model.fit_predict(coordinates)
    """
    def __init__(
        self,
        r2=10,
        label_singleton=True,
        min_spacial_resolution=0,
        distance_metric='haversine',
        weighted=False,
        weight_exponent=1,
        verbose=False
    ):
        # Set input parameters as attributes
        self.r2 = r2
        self.label_singleton = label_singleton
        self.min_spacial_resolution = min_spacial_resolution
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.weight_exponent = weight_exponent
        self.verbose = verbose
        self.data = None
        self.stat_coords = None
        self.stat_labels = None
        self.counts = None
        self.labels = None
        self.is_fitted = False

        # Run hyper parameter assertions
        self._hyperparam_assertions()


    def fit_predict(self, data):
        """Fit Infostop on one or more location sequnces, and return labels.

        Parameters
        ----------
            data : numpy.array (shape (N, 2)/(N, 3)) or list of such numpy.arrays
                Columns 0 and 1 are reserved for lat and lon. Column 2 is reserved for time (any unit consistent with
                `min_staying_time` and `max_time_between`). If the input type is a list of arrays, each array is assumed
                to be the trace of a single user, in which case the obtained stop locations are shared by all users in
                the population.

        Returns
        -------
            coord_labels : 1d numpy.array or list of such

        Example
        -------
            >>> model = Infostop()
            >>> labels = model.fit_predict(traces)
            >>> assert type(traces) == type(labels)
        """
        
        if self.verbose: progress = tqdm
        else:            progress = utils.pass_func

        self.data = data

        # Assert the input data
        self._data_assertions(self.data)
        
        # (1) Downsample (dramatically reduces computation time)
        self.stat_coords = data
        if self.min_spacial_resolution > 0:
            self.stat_coords = np.around(self.stat_coords / self.min_spacial_resolution) * self.min_spacial_resolution

        if self.verbose:
            num_stat_orig = len(self.stat_coords)
            print(f"Downsampling {num_stat_orig} total stop events to...", end=" ")

        # Only keep unique coordinates for clustering
        self.stat_coords, inverse_indices, self.counts = np.unique(
            self.stat_coords,
            return_inverse=True, return_counts=True, axis=0
        )

        if self.verbose:
            print(f"{len(self.stat_coords)}", end=" ")
            print("(%.1f%% duplicates)" % ((1 - len(self.stat_coords)/num_stat_orig)*100))

        # (2) Find neighbors within `r2` for each point
        if self.verbose:
            print("Finding neighbors...", end=" ")
        
        ball_tree_result = utils.query_neighbors(self.stat_coords, self.r2, self.distance_metric, self.weighted)
        
        if self.weighted:
            node_idx_neighbors, node_idx_distances = ball_tree_result
        else:
            node_idx_neighbors, node_idx_distances = ball_tree_result, None

        if self.verbose:
            print("done")
            
        # (3) Create network and run infomap
        if self.verbose: print("Creating network and clustering with Infomap...")
        self.stat_labels = utils.label_network(
            node_idx_neighbors, node_idx_distances, self.counts, self.weight_exponent,
            self.label_singleton, self.distance_metric, self.verbose
        )

        # (4) Reverse the downsampling in step (2)
        self.labels = self.stat_labels[inverse_indices]

        # Update model state and return labels
        self.is_fitted = True
        return self.labels


    def _hyperparam_assertions(self):
        assert self.r2 > 0, \
               "`r2` must be > 0"
        assert type(self.label_singleton) is bool, \
               "`label_singleton` either `True` or `False`"
        assert 0 <= self.min_spacial_resolution <= 1, \
               "`min_spacial_resolution` must be within [0, 1]"
        assert self.distance_metric in ['euclidean', 'haversine'], \
               "`distance_metric` should be either 'euclidean' or 'haversine'"


    def _data_assertions(self, data):
        assert not np.any(np.isnan(data)), \
               f"There are {np.isnan(np.vstack(data))} NaN values in the input data."
        assert data.shape[1] == 2, \
               "Number of columns must be 2"
        if self.distance_metric == 'haversine':
            assert (np.min(data[:, 0]) > -90 and np.max(data[:, 0]) < 90), \
                   "Latitude (column 0) must have values between -90 and 90"
            assert (np.min(data[:, 1]) > -180 and np.max(data[:, 1]) < 180), \
                   "Longitude (column 1) must have values between -180 and 180"