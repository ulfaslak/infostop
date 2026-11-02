import warnings

import numpy as np

from infostop.models import Infostop

# Used, with a FutureWarning, when neither `max_time_between` nor `model` is given
_DEFAULT_MAX_TIME_BETWEEN = 86400


def compute_intervals(labels, times, max_time_between=None, *, model=None):
    """Compute stop and moves intervals from the list of labels.

    Give either `model` or `max_time_between`, so that the intervals use the same
    `max_time_between` as the model that produced `labels`.

    Parameters
    ----------
        labels: 1d np.array of integers
        times: 1d np.array of integers. `len(labels) == len(times)`.
        max_time_between: int/float
            Consecutive points with the same label are in one interval if the time between
            them is at most `max_time_between`. Use the `max_time_between` of the model that
            produced `labels`. Do not give it together with `model`.
        model: Infostop
            The model that produced `labels`. Its `max_time_between` is used. Do not give it
            together with `max_time_between`.

            If you give neither, `max_time_between` is 86400 (24 hours in seconds) and a
            FutureWarning is shown. A future version will require one of them.
    Returns
    -------
        intervals : array-like (shape=(N_intervals, 3))
            Columns are "label", "start_time", "end_time"

    Example
    -------
        >>> model = Infostop()
        >>> labels = model.fit_predict(data)
        >>> intervals = compute_intervals(labels, data[:, 2], model=model)
    """
    if model is not None:
        if max_time_between is not None:
            raise TypeError("Give `model` or `max_time_between`, not both")
        if not isinstance(model, Infostop):
            raise TypeError(
                "`model` must be an `Infostop` model, not %s. Only `Infostop` has "
                "`max_time_between`." % type(model).__name__
            )
        max_time_between = model._max_time_between
    elif max_time_between is None:
        warnings.warn(
            "compute_intervals() got neither `model` nor `max_time_between`, so it uses "
            "max_time_between=%d (24 hours in seconds). This is wrong if the model that "
            "produced the labels has another `max_time_between`, or if the timestamps are "
            "not in seconds. Give `model=` (the fitted Infostop model) or "
            "`max_time_between=`. A future version will require one of them."
            % _DEFAULT_MAX_TIME_BETWEEN,
            FutureWarning,
            stacklevel=2,
        )
        max_time_between = _DEFAULT_MAX_TIME_BETWEEN

    assert len(labels) == len(times), "`labels` and `times` must match in length"

    # Input and output sequence
    trajectory = np.hstack([labels.reshape(-1, 1), times.reshape(-1, 1)])
    final_trajectory = []
    if len(trajectory) == 0:
        return final_trajectory

    # Init values
    loc_prev, t_start = trajectory[0]
    t_end = t_start

    # Loop through trajectory
    for loc, time in trajectory[1:]:

        # if the location name has not changed update the end of the interval
        if (loc == loc_prev) and (time - t_end) <= max_time_between:
            t_end = time

        # if the location name has changed build the interval
        else:
            final_trajectory.append([loc_prev, t_start, t_end])

            # and reset values
            t_start = time
            t_end = time

        # update current values
        loc_prev = loc

    # Add last group
    final_trajectory.append([loc_prev, t_start, t_end])

    return final_trajectory
