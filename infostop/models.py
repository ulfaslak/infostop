import numbers

import numpy as np
import cpputils
from infostop import utils
from tqdm import tqdm


class NoStopsFoundException(Exception):
    pass


def _check_num_trials(num_trials):
    """Raise unless `num_trials` is an integer >= 1 (NumPy integers allowed, booleans not).

    An explicit raise rather than `assert`, so that `python -O` cannot let a bad value
    through to Infomap: Infomap 1.0.6 hangs on -1.
    """
    if (
        not isinstance(num_trials, numbers.Integral)
        or isinstance(num_trials, bool)
        or num_trials < 1
    ):
        raise AssertionError("`num_trials` must be an integer >= 1")


class Infostop:
    """Infer stop-location labels from mobility trace. Dynamic points are labeled -1.

    Note: currently ONLY works for 2-dimensional spatial data. Data columns 0 and 1 are
    always interpreted as spatial locations while column 2 is interpreted as time.

    Units: with `distance_metric='haversine'` (the default), columns 0 and 1 are latitude and
    longitude in degrees, and the distances `r1` and `r2` are in meters. With 'euclidean', all
    distances are in the units of the coordinates. The time parameters `min_staying_time` and
    `max_time_between` are in the units of the time column; their defaults assume seconds.

    The method entails the following steps:
        1.  Sequential downsampling: Detect which points are stationary and store only
            the median of each stationarity event. A point belongs to the current
            stationarity event if it is at most `r1` away from the median of the
            time-previous points in the event (and, with a time column, at most
            `max_time_between` after the previous point).
        2.  Spatial downsampling: Remove duplicate coordinates. Optionally round off
            coordinates before removing duplicates, to increase the downsampling.
        3.  For each coordinate, find its neighbors within `r2`.
        4.  Create Infomap network and optimize clusters in two-levels.
        5.  Reverse downsampling from step 2.
        6.  Reverse downsampling from step 1.

    Parameters
    ----------
        r1 : int/float
            Max distance between a point and the median of the current stationarity event for the
            point to join the event. Meters for 'haversine', coordinate units for 'euclidean'.
        r2 : int/float
            Max distance between the medians of two stationarity events to link them with an edge.
            Meters for 'haversine', coordinate units for 'euclidean'.
        label_singleton: bool
            If True, give each stationary location (a distinct stationarity event median) that has no
            other stationary location within `r2` its own label. If False, label it as non-stationary
            (-1).
        min_staying_time : int/float
            The shortest duration that can constitute a stop, from the first to the last point of a
            stationarity event. In the units of the time column (the default, 300, is 5 minutes in
            seconds). Ignored if the data has no time column.
        max_time_between : int/float
            The longest time allowed between two consecutive points in the same stationarity event.
            A longer gap ends the event. In the units of the time column (the default, 86400, is 24
            hours in seconds). Ignored if the data has no time column.
        min_size : int
            Minimum number of points in a stationarity event.
        min_spacial_resolution : float
            The minimal difference allowed between points before they are considered the same points.
            Highly useful for spatially downsampling data and which dramatically reduces runtime. Higher
            values yields higher downsampling. In coordinate units: degrees for 'haversine', so 1e-5 is
            about 1 m. For geo data, it is not recommended to increase it beyond 1e-4. 1e-5, typically
            works well and has little impact on results. 0 (the default) turns rounding off.
        distance_metric : str
            Either 'haversine' (for geo data: latitude and longitude in degrees) or 'euclidean'.
        weighted : bool
            Weight edges in the network representation by distance. Each edge always has weight
            `max(count(a), count(b))`, where `count` is the number of stationarity events whose
            median has exactly these coordinates (after rounding by `min_spacial_resolution`). If
            True, that weight is divided by `distance(a, b) ** weight_exponent`, with the distance in
            meters for 'haversine' and in coordinate units for 'euclidean'.
        weight_exponent : float
            Exponent used when weighting edges in the network. Only used if `weighted=True`. Higher
            values increases the link strength between nearby locations yielding smaller stop
            locations. It is not recommended to use a negative `weight_exponent`, unless for a special
            purpose.
        verbose : bool
            Print output during the fitting procedure. Mostly for development and debugging. Option will
            be removed in future versions.
        num_trials : int
            Number of times Infomap clusters the network from a different random start. The partition
            with the shortest description (codelength) is kept. With 1 trial, labels on crowded stops
            (stops that overlap within `r2`) can differ between Infomap versions and between operating
            systems; the default 50 makes them agree for almost all inputs. Use 1 to get the labels that
            infostop gave before this option existed (on the same Infomap version and operating system).
            More trials take more time in the clustering step.

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
        distance_metric="haversine",
        weighted=False,
        weight_exponent=1,
        verbose=False,
        num_trials=50,
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
        self._num_trials = num_trials

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
            data : numpy.array (shape (N, 2)/(N, 3)) or list (or tuple) of such numpy.arrays
                Columns 0 and 1 are reserved for lat and lon. Column 2 is reserved for time (any unit consistent with
                `min_staying_time` and `max_time_between`). If the input type is a list or tuple of arrays, each array
                is assumed to be the trace of a single user, in which case the obtained stop locations are shared by all
                users in the population.

        Returns
        -------
            coord_labels : 1d numpy.array or list (or tuple, matching the input) of such

        Example
        -------
            >>> model = Infostop()
            >>> labels = model.fit_predict(traces)
            >>> assert type(traces) == type(labels)
        """

        if self._verbose:
            progress = tqdm
        else:
            progress = utils.pass_func

        # Infer multiuser mode
        self.multiuser = True
        if type(data) not in (list, tuple):
            self._data = [data]
            self.multiuser = False
            progress = (
                utils.pass_func
            )  # no need to log progress in (1) if there's just one user
        else:
            self._data = data
            if len(data) == 1:
                progress = utils.pass_func

        if self._verbose:
            print("Multiuser input:", self.multiuser)

        # Assert the input data
        self._data_assertions(self._data)

        # (1) Sequential downsampling: group time-adjacent points
        if self._verbose:
            avg_reduction = []
            print("Downsampling in time: keeping medians of stationary events")

        stop_events, event_maps = [], []
        for u, coords_u in progress(enumerate(self._data), total=len(self._data)):
            # A user with no points has no stop events and gets an empty label array
            if len(coords_u) == 0:
                stop_events.append([])
                event_maps.append(np.array([], dtype=int))
                continue

            stop_events_u, event_map_u = cpputils.get_stationary_events(
                coords_u,
                self._r1,
                self._min_size,
                self._min_staying_time,
                self._max_time_between,
                self._distance_metric,
            )

            if self._verbose:
                avg_reduction.append((1 - len(stop_events_u) / len(coords_u)) * 100)

            stop_events.append(stop_events_u)
            event_maps.append(event_map_u)

        if self._verbose and avg_reduction:
            print(
                "    --> %sreduction was %.1f%%"
                % ("average " if self.multiuser else "", np.mean(avg_reduction))
            )

        # Merge `stop_events` from different users into `stat_coords`
        try:
            self._stat_coords = np.vstack([se for se in stop_events if len(se) > 0])
        except ValueError:
            raise Exception(
                "No stop events found. Check that `r1`, `min_staying_time` and `min_size` parameters are chosen correctly."
            )

        # (2) Downsample (dramatically reduces computation time)
        if self._min_spacial_resolution > 0:
            self._stat_coords = (
                np.around(self._stat_coords / self._min_spacial_resolution)
                * self._min_spacial_resolution
            )

        if self._verbose:
            num_stat_orig = len(self._stat_coords)
            print(f"Downsampling {num_stat_orig} total stop events to...", end=" ")

        # Only keep unique coordinates for clustering
        self._stat_coords, inverse_indices, self._counts = np.unique(
            self._stat_coords, return_inverse=True, return_counts=True, axis=0
        )

        if self._verbose:
            print(f"{len(self._stat_coords)}", end=" ")
            print(
                "(%.1f%% duplicates)"
                % ((1 - len(self._stat_coords) / num_stat_orig) * 100)
            )

        # (3) Find neighbors within `r2` for each point
        if self._verbose:
            print("Finding neighbors...", end=" ")

        ball_tree_result = utils.query_neighbors(
            self._stat_coords, self._r2, self._distance_metric, self._weighted
        )

        if self._weighted:
            node_idx_neighbors, node_idx_distances = ball_tree_result
        else:
            node_idx_neighbors, node_idx_distances = ball_tree_result, None

        if self._verbose:
            print("done")

        # (4) Create network and run infomap
        if self._verbose:
            print("Creating network and clustering with Infomap...")
        self._stat_labels = utils.label_network(
            node_idx_neighbors,
            node_idx_distances,
            self._counts,
            self._weight_exponent,
            self._label_singleton,
            self._distance_metric,
            self._verbose,
            getattr(self, "_num_trials", 1),  # models pickled before num_trials: 1
        )

        # (5) Reverse the downsampling in step (2). NumPy 2.0.0 returns the inverse
        # of `np.unique(..., axis=0)` with shape (N, 1), so flatten it.
        self._labels = self._stat_labels[inverse_indices.reshape(-1)]

        # (6) Reverse the downsampling in step (1)
        self.labels = []
        for j, event_map_u in enumerate(event_maps):
            i0 = sum([len(stop_events[j_]) for j_ in range(j)])
            i1 = sum([len(stop_events[j_]) for j_ in range(j + 1)])
            labels_u = np.hstack([self._labels[i0:i1], -1])
            self.labels.append(labels_u[event_map_u])

        # Update model state and return labels
        self._is_fitted = True
        if self.multiuser:
            return self.labels if type(data) == list else tuple(self.labels)
        else:
            return self.labels[0]

    def compute_label_medians(self):
        """Compute the median location of inferred labels.

        The median of a label is the coordinate-wise median of its stop event medians.
        With 'haversine', a label whose longitudes span more than 180 degrees is taken
        to lie on both sides of longitude +-180, and its median longitude is taken
        across that line, in (-180, 180].

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
        labels_and_coords = np.hstack(
            [self._stat_labels.reshape(-1, 1), self._stat_coords]
        )

        # Remove outliers
        labels_and_coords = labels_and_coords[labels_and_coords[:, 0] != -1]

        # Get unique labels
        unique_labels = np.unique(labels_and_coords[:, 0]).astype(int)

        # For each unique label, take the median of its points. Latitude and
        # longitude also have a median across the antimeridian.
        median = (
            utils.median_lat_lon
            if self._distance_metric == "haversine"
            else lambda points: np.median(points, axis=0)
        )
        unique_label_medians = [
            median(labels_and_coords[labels_and_coords[:, 0] == label, 1:]).tolist()
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

        Returns
        -------
            labels : dict, {label: count}

        Example
        -------
            >>> label_counts = model.compute_label_counts()
        """
        # Assert model is fitted
        self._fitted_assertion()
        raise NotImplementedError

    def _hyperparam_assertions(self):
        assert self._r1 > 0, "`r1` must be > 0"
        assert self._r2 > 0, "`r2` must be > 0"
        assert (
            type(self._label_singleton) is bool
        ), "`label_singleton` either `True` or `False`"
        assert self._min_staying_time > 0, "`min_staying_time` must be > 0"
        assert self._max_time_between > 0, "`max_time_between` must be > 0"
        assert (
            self._max_time_between > self._min_staying_time
        ), "`max_time_between` must be > min_staying_time"
        assert self._min_size > 1, "`min_size` must be > 1"
        assert (
            0 <= self._min_spacial_resolution <= 1
        ), "`min_spacial_resolution` must be within [0, 1]"
        assert self._distance_metric in [
            "euclidean",
            "haversine",
        ], "`distance_metric` should be either 'euclidean' or 'haversine'"
        _check_num_trials(self._num_trials)

    def _data_assertions(self, data):
        assert len(data) > 0, "There are no traces in the input data."
        # Check each trace on its own: users may differ in their number of columns
        assert not any(np.isnan(np.asarray(c)).any() for c in data), (
            f"There are {sum(np.isnan(np.asarray(c)).sum() for c in data)} NaN values "
            "in the input data."
        )
        for u, coords_u in enumerate(data):
            error_insert = "" if not self.multiuser else f"User {u}: "
            assert hasattr(coords_u, "shape") and len(coords_u.shape) == 2, (
                "%sEach trace must be a 2D numpy array with one row per point, got %s. "
                "Pass a single trace as one 2D array, or several users as a list (or "
                "tuple) of 2D arrays."
                % (
                    error_insert,
                    (
                        f"an array of shape {coords_u.shape}"
                        if hasattr(coords_u, "shape")
                        else f"a {type(coords_u).__name__}"
                    ),
                )
            )
            assert coords_u.shape[1] in [2, 3], (
                "%sNumber of columns must be 2 or 3" % error_insert
            )
            if coords_u.shape[1] == 3:
                assert np.all(coords_u[:-1, 2] <= coords_u[1:, 2]), (
                    "%sTimestamps must be ordered" % error_insert
                )
            if self._distance_metric == "haversine" and len(coords_u) > 0:
                assert np.min(coords_u[:, 0]) > -90 and np.max(coords_u[:, 0]) < 90, (
                    "%sLatitude (column 0) must have values between -90 and 90"
                    % error_insert
                )
                assert np.min(coords_u[:, 1]) > -180 and np.max(coords_u[:, 1]) < 180, (
                    "%sLongitude (column 1) must have values between -180 and 180"
                    % error_insert
                )
        assert any(
            len(coords_u) > 0 for coords_u in data
        ), "There are no points in the input data."

    def _fitted_assertion(self):
        assert (
            self._is_fitted
        ), "Model must be fitted before label medians can be computed."


class SpatialInfomap:
    """Cluster a collection of points using Infomap.

    Units: with `distance_metric='haversine'` (the default), columns 0 and 1 are latitude and
    longitude in degrees, and `r2` is in meters. With 'euclidean', all distances are in the units
    of the coordinates.

    The method entails the following steps:
        1.  Spatial downsampling: Remove duplicate coordinates. Optionally round off
            coordinates before removing duplicates, to increase the downsampling.
        2.  For each coordinate, find its neighbors within `r2`.
        3.  Create Infomap network and optimize clusters in two-levels.
        4.  Reverse downsampling from step 1.

    Parameters
    ----------
        r2 : int/float
            Max distance between two points to link them with an edge. Meters for 'haversine',
            coordinate units for 'euclidean'.
        label_singleton: bool
            If True, give each point (or group of identical points) that has no other point within `r2`
            its own label. If False, label it -1.
        min_spacial_resolution : float
            The minimal difference allowed between points before they are considered the same points.
            Highly useful for spatially downsampling data and which dramatically reduces runtime. Higher
            values yields higher downsampling. In coordinate units: degrees for 'haversine', so 1e-5 is
            about 1 m. For geo data, it is not recommended to increase it beyond 1e-4. 1e-5, typically
            works well and has little impact on results. 0 (the default) turns rounding off.
        distance_metric : str
            Either 'haversine' (for geo data: latitude and longitude in degrees) or 'euclidean'.
        weighted : bool
            Weight edges in the network representation by distance. Each edge always has weight
            `max(count(a), count(b))`, where `count` is the number of input points with exactly these
            coordinates (after rounding by `min_spacial_resolution`). If True, that weight is divided
            by `distance(a, b) ** weight_exponent`, with the distance in meters for 'haversine' and in
            coordinate units for 'euclidean'.
        weight_exponent : float
            Exponent used when weighting edges in the network. Only used if `weighted=True`. Higher
            values increases the link strength between nearby locations yielding smaller stop
            locations. It is not recommended to use a negative `weight_exponent`, unless for a special
            purpose.
        verbose : bool
            Print output during the fitting procedure. Mostly for development and debugging. Option will
            be removed in future versions.
        num_trials : int
            Number of times Infomap clusters the network from a different random start. The partition
            with the shortest description (codelength) is kept. With 1 trial, labels on crowded points
            (clusters that overlap within `r2`) can differ between Infomap versions and between
            operating systems; the default 50 makes them agree for almost all inputs. Use 1 to get the
            labels that infostop gave before this option existed (on the same Infomap version and
            operating system). More trials take more time in the clustering step.

    Returns
    -------
        self : object

    Example
    -------
        >>> model = SpatialInfomap()
        >>> labels = model.fit_predict(coordinates)
    """

    def __init__(
        self,
        r2=10,
        label_singleton=True,
        min_spacial_resolution=0,
        distance_metric="haversine",
        weighted=False,
        weight_exponent=1,
        verbose=False,
        num_trials=50,
    ):
        # Set input parameters as attributes
        self._r2 = r2
        self._label_singleton = label_singleton
        self._min_spacial_resolution = min_spacial_resolution
        self._distance_metric = distance_metric
        self._weighted = weighted
        self._weight_exponent = weight_exponent
        self._verbose = verbose
        self._num_trials = num_trials
        self._data = None
        self._stat_coords = None
        self._stat_labels = None
        self._counts = None
        self._labels = None
        self._is_fitted = False

        # Run hyper parameter assertions
        self._hyperparam_assertions()

    def fit_predict(self, data):
        """Fit SpatialInfomap on a collection of points, and return labels.

        Parameters
        ----------
            data : numpy.array (shape (N, 2))
                One row per point, in any order. With 'haversine', column 0 is latitude and column 1
                is longitude, in degrees. With 'euclidean', the columns are x and y. A list of arrays
                or a time column is not supported.

        Returns
        -------
            coord_labels : 1d numpy.array (shape (N, ))
                One label per input row. With `label_singleton=False`, points that have no other
                point within `r2` are labeled -1.

        Example
        -------
            >>> model = SpatialInfomap()
            >>> labels = model.fit_predict(coordinates)
        """

        if self._verbose:
            progress = tqdm
        else:
            progress = utils.pass_func

        self._data = data

        # Assert the input data
        self._data_assertions(self._data)

        # (1) Downsample (dramatically reduces computation time)
        self._stat_coords = data
        if self._min_spacial_resolution > 0:
            self._stat_coords = (
                np.around(self._stat_coords / self._min_spacial_resolution)
                * self._min_spacial_resolution
            )

        if self._verbose:
            num_stat_orig = len(self._stat_coords)
            print(f"Downsampling {num_stat_orig} total stop events to...", end=" ")

        # Only keep unique coordinates for clustering
        self._stat_coords, inverse_indices, self._counts = np.unique(
            self._stat_coords, return_inverse=True, return_counts=True, axis=0
        )

        if self._verbose:
            print(f"{len(self._stat_coords)}", end=" ")
            print(
                "(%.1f%% duplicates)"
                % ((1 - len(self._stat_coords) / num_stat_orig) * 100)
            )

        # (2) Find neighbors within `r2` for each point
        if self._verbose:
            print("Finding neighbors...", end=" ")

        ball_tree_result = utils.query_neighbors(
            self._stat_coords, self._r2, self._distance_metric, self._weighted
        )

        if self._weighted:
            node_idx_neighbors, node_idx_distances = ball_tree_result
        else:
            node_idx_neighbors, node_idx_distances = ball_tree_result, None

        if self._verbose:
            print("done")

        # (3) Create network and run infomap
        if self._verbose:
            print("Creating network and clustering with Infomap...")
        self._stat_labels = utils.label_network(
            node_idx_neighbors,
            node_idx_distances,
            self._counts,
            self._weight_exponent,
            self._label_singleton,
            self._distance_metric,
            self._verbose,
            getattr(self, "_num_trials", 1),  # models pickled before num_trials: 1
        )

        # (4) Reverse the downsampling in step (1). NumPy 2.0.0 returns the inverse
        # of `np.unique(..., axis=0)` with shape (N, 1), so flatten it.
        self._labels = self._stat_labels[inverse_indices.reshape(-1)]

        # Update model state and return labels
        self._is_fitted = True
        return self._labels

    def _hyperparam_assertions(self):
        assert self._r2 > 0, "`r2` must be > 0"
        assert (
            type(self._label_singleton) is bool
        ), "`label_singleton` either `True` or `False`"
        assert (
            0 <= self._min_spacial_resolution <= 1
        ), "`min_spacial_resolution` must be within [0, 1]"
        assert self._distance_metric in [
            "euclidean",
            "haversine",
        ], "`distance_metric` should be either 'euclidean' or 'haversine'"
        _check_num_trials(self._num_trials)

    def _data_assertions(self, data):
        assert not np.any(
            np.isnan(data)
        ), f"There are {np.count_nonzero(np.isnan(data))} NaN values in the input data."
        assert data.shape[1] == 2, "Number of columns must be 2"
        if self._distance_metric == "haversine":
            assert (
                np.min(data[:, 0]) > -90 and np.max(data[:, 0]) < 90
            ), "Latitude (column 0) must have values between -90 and 90"
            assert (
                np.min(data[:, 1]) > -180 and np.max(data[:, 1]) < 180
            ), "Longitude (column 1) must have values between -180 and 180"
