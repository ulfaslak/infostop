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
        self._r1 = r1
        self._r2 = r2
        self._label_singleton = label_singleton
        self._min_staying_time = min_staying_time
        self._max_time_between = max_time_between
        self._min_size = min_size
        self._min_spacial_resolution = min_spacial_resolution
        self._distance_metric = distance_metric
        self._weighted = weighted
        self._weight_exponent = weight_exponent
        self._verbose = verbose
        
        # Initialize internal variables
        self._data = None
        self._stat_coords = None
        self._stat_labels = None
        self._counts = None
        self._labels = None
        self._is_fitted = False

        # Initialize computable attributes
        self.label_medians = None
        # self.label_areas = None
        # self.label_count = None

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
        
        if self._verbose: progress = tqdm
        else:            progress = utils.pass_func

        # Infer multiuser mode
        self.multiuser = True
        if type(data) != list:
            self._data = [data]
            self.multiuser = False
            progress = utils.pass_func  # no need to log progress in (1) if there's just one user
        else:
            self._data = data
            if len(data) == 1:
                progress = utils.pass_func

        if self._verbose:
            print('Multiuser input:', self.multiuser)

        # Assert the input data
        self._data_assertions(self._data)

        # (1) Sequential downsampling: group time-adjacent points
        if self._verbose:
            avg_reduction = []
            print("Downsampling in time: keeping medians of stationary events")

        stop_events, event_maps = [], []
        for u, coords_u in progress(enumerate(self._data), total=len(self._data)):
            stop_events_u, event_map_u = cpputils.get_stationary_events(
                coords_u, self._r1, self._min_size, self._min_staying_time,
                self._max_time_between, self._distance_metric
            )

            if self._verbose:
                avg_reduction.append((1 - len(stop_events_u) / len(coords_u)) * 100)

            stop_events.append(stop_events_u)
            event_maps.append(event_map_u)

        if self._verbose:
            print("    --> %sreduction was %.1f%%" % ("average " if self.multiuser else "", np.mean(avg_reduction)))
        
        # Merge `stop_events` from different users into `stat_coords`
        try:
            self._stat_coords = np.vstack([se for se in stop_events if len(se) > 0])
        except ValueError:
            raise Exception("No stop events found. Check that `r1`, `min_staying_time` and `min_size` parameters are chosen correctly.")

        # (2) Downsample (dramatically reduces computation time)
        if self._min_spacial_resolution > 0:
            self._stat_coords = np.around(self._stat_coords / self._min_spacial_resolution) * self._min_spacial_resolution

        if self._verbose:
            num_stat_orig = len(self._stat_coords)
            print(f"Downsampling {num_stat_orig} total stop events to...", end=" ")

        # Only keep unique coordinates for clustering
        self._stat_coords, inverse_indices, self._counts = np.unique(
            self._stat_coords,
            return_inverse=True, return_counts=True, axis=0
        )

        if self._verbose:
            print(f"{len(self._stat_coords)}", end=" ")
            print("(%.1f%% duplicates)" % ((1 - len(self._stat_coords)/num_stat_orig)*100))

        # (3) Find neighbors within `r2` for each point
        if self._verbose:
            print("Finding neighbors...", end=" ")
        
        ball_tree_result = utils.query_neighbors(self._stat_coords, self._r2, self._distance_metric, self._weighted)
        
        if self._weighted:
            node_idx_neighbors, node_idx_distances = ball_tree_result
        else:
            node_idx_neighbors, node_idx_distances = ball_tree_result, None

        if self._verbose:
            print("done")
            
        # (4) Create network and run infomap
        if self._verbose: print("Creating network and clustering with Infomap...")
        self._stat_labels = utils.label_network(
            node_idx_neighbors, node_idx_distances, self._counts, self._weight_exponent,
            self._label_singleton, self._distance_metric, self._verbose
        )

        # (5) Reverse the downsampling in step (2)
        self._labels = self._stat_labels[inverse_indices]
        
        # (6) Reverse the downsampling in step (1)
        self.labels = []
        for j, event_map_u in enumerate(event_maps):
            i0 = sum([len(stop_events[j_]) for j_ in range(j)])
            i1 = sum([len(stop_events[j_]) for j_ in range(j+1)])
            labels_u = np.hstack([self._labels[i0:i1], -1])
            self.labels.append(labels_u[event_map_u])

        # Update model state and return labels
        self._is_fitted = True
        if self.multiuser:
            return self.labels
        else:
            return self.labels[0]


    def compute_label_medians(self):
        """Compute the median location of inferred labels.

        Returns
        -------
            label_medians : dict, {label: [lat, lon]}

        Example
        -------
            >>> label_medians = model.compute_label_medians()
        """
        # Assert model is fitted
        self._fitted_assertion()

        # Stack labels and coords
        labels_and_coords = np.hstack([self._stat_labels.reshape(-1, 1), self._stat_coords])

        # Remove outliers
        labels_and_coords = labels_and_coords[labels_and_coords[:, 0] != -1]

        # Get unique labels
        unique_labels = np.unique(labels_and_coords[:, 0]).astype(int)

        # For each unique label, take the median of its points
        unique_label_medians = [
            np.median(labels_and_coords[labels_and_coords[:, 0] == label, 1:], axis=0).tolist()
            for label in unique_labels
        ]

        # Zip into dictionary
        self.label_medians = dict(zip(unique_labels, unique_label_medians))

        return self.label_medians

    
    def compute_label_area(self):
        """Compute the area of inferred stop locations
        
        Returns
        -------
            label_areas : dict, {label: area}

        Example
        -------
            >>> label_areas = model.compute_label_areas()
        """
        # Assert model is fitted
        self._fitted_assertion()
        raise NotImplementedError


    def compute_label_counts(self):
        """Compute the count of inferred stop locations
        
        Returns
        -------
            label_counts : dict, {label: count}

        Example
        -------
            >>> label_counts = model.compute_label_counts()
        """
        # Assert model is fitted
        self._fitted_assertion()
        raise NotImplementedError
    

    def predict(self, data):
        """Predict labels of data given existing solution.
        
        # Returns
        # -------
        #     labels : dict, {label: count}

        # Example
        # -------
        #     >>> label_counts = model.compute_label_counts()
        """
        # Assert model is fitted
        self._fitted_assertion()
        raise NotImplementedError


    def _hyperparam_assertions(self):
        assert self._r1 > 0, \
               "`r1` must be > 0"
        assert self._r2 > 0, \
               "`r2` must be > 0"
        assert type(self._label_singleton) is bool, \
               "`label_singleton` either `True` or `False`"
        assert self._min_staying_time > 0, \
               "`min_staying_time` must be > 0"
        assert self._max_time_between > 0, \
               "`max_time_between` must be > 0"
        assert self._max_time_between > self._min_staying_time, \
               "`max_time_between` must be > min_staying_time"
        assert self._min_size > 1, \
               "`min_size` must be > 1"
        assert 0 <= self._min_spacial_resolution <= 1, \
               "`min_spacial_resolution` must be within [0, 1]"
        assert self._distance_metric in ['euclidean', 'haversine'], \
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
            if self._distance_metric == 'haversine':
                assert (np.min(coords_u[:, 0]) > -90 and np.max(coords_u[:, 0]) < 90), \
                       "%sLatitude (column 0) must have values between -90 and 90" % error_insert
                assert (np.min(coords_u[:, 1]) > -180 and np.max(coords_u[:, 1]) < 180), \
                       "%sLongitude (column 1) must have values between -180 and 180" % error_insert

    
    def _fitted_assertion(self):
        assert self._is_fitted, \
                "Model must be fitted before label medians can be computed."


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
        self._r2 = r2
        self._label_singleton = label_singleton
        self._min_spacial_resolution = min_spacial_resolution
        self._distance_metric = distance_metric
        self._weighted = weighted
        self._weight_exponent = weight_exponent
        self._verbose = verbose
        self._data = None
        self._stat_coords = None
        self._stat_labels = None
        self._counts = None
        self._labels = None
        self._is_fitted = False

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
        
        if self._verbose: progress = tqdm
        else:            progress = utils.pass_func

        self._data = data

        # Assert the input data
        self._data_assertions(self._data)
        
        # (1) Downsample (dramatically reduces computation time)
        self._stat_coords = data
        if self._min_spacial_resolution > 0:
            self._stat_coords = np.around(self._stat_coords / self._min_spacial_resolution) * self._min_spacial_resolution

        if self._verbose:
            num_stat_orig = len(self._stat_coords)
            print(f"Downsampling {num_stat_orig} total stop events to...", end=" ")

        # Only keep unique coordinates for clustering
        self._stat_coords, inverse_indices, self._counts = np.unique(
            self._stat_coords,
            return_inverse=True, return_counts=True, axis=0
        )

        if self._verbose:
            print(f"{len(self._stat_coords)}", end=" ")
            print("(%.1f%% duplicates)" % ((1 - len(self._stat_coords)/num_stat_orig)*100))

        # (2) Find neighbors within `r2` for each point
        if self._verbose:
            print("Finding neighbors...", end=" ")
        
        ball_tree_result = utils.query_neighbors(self._stat_coords, self._r2, self._distance_metric, self._weighted)
        
        if self._weighted:
            node_idx_neighbors, node_idx_distances = ball_tree_result
        else:
            node_idx_neighbors, node_idx_distances = ball_tree_result, None

        if self._verbose:
            print("done")
            
        # (3) Create network and run infomap
        if self._verbose: print("Creating network and clustering with Infomap...")
        self._stat_labels = utils.label_network(
            node_idx_neighbors, node_idx_distances, self._counts, self._weight_exponent,
            self._label_singleton, self._distance_metric, self._verbose
        )

        # (4) Reverse the downsampling in step (2)
        self._labels = self._stat_labels[inverse_indices]

        # Update model state and return labels
        self._is_fitted = True
        return self._labels


    def _hyperparam_assertions(self):
        assert self._r2 > 0, \
               "`r2` must be > 0"
        assert type(self._label_singleton) is bool, \
               "`label_singleton` either `True` or `False`"
        assert 0 <= self._min_spacial_resolution <= 1, \
               "`min_spacial_resolution` must be within [0, 1]"
        assert self._distance_metric in ['euclidean', 'haversine'], \
               "`distance_metric` should be either 'euclidean' or 'haversine'"


    def _data_assertions(self, data):
        assert not np.any(np.isnan(data)), \
               f"There are {np.isnan(np.vstack(data))} NaN values in the input data."
        assert data.shape[1] == 2, \
               "Number of columns must be 2"
        if self._distance_metric == 'haversine':
            assert (np.min(data[:, 0]) > -90 and np.max(data[:, 0]) < 90), \
                   "Latitude (column 0) must have values between -90 and 90"
            assert (np.min(data[:, 1]) > -180 and np.max(data[:, 1]) < 180), \
                   "Longitude (column 1) must have values between -180 and 180"