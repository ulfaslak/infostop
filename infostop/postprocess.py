import numpy as np

def compute_intervals(labels, times, max_time_between=86400):
    """Compute stop and moves intervals from the list of labels.
    
    Parameters
    ----------
        labels: 1d np.array of integers
        times: 1d np.array of integers. `len(labels) == len(times)`.
    Returns
    -------
        intervals : array-like (shape=(N_intervals, 3))
            Columns are "label", "start_time", "end_time"
    
    """
    assert len(labels) == len(times), '`labels` and `times` must match in length'
    
    # Input and output sequence
    trajectory = np.hstack([labels.reshape(-1, 1), times.reshape(-1,1)])
    final_trajectory = []
    
    # Init values
    loc_prev, t_start = trajectory[0]  
    t_end = t_start

    # Loop through trajectory
    for loc, time in trajectory[1:]:
        
        # if the location name has not changed update the end of the interval
        if (loc == loc_prev) and (time - t_end) < max_time_between:
            t_end = time
              
        # if the location name has changed build the interval
        else:
            final_trajectory.append([loc_prev, t_start,  t_end])
            
            # and reset values
            t_start = time 
            t_end = time 
        
        # update current values
        loc_prev = loc
        
    # Add last group
    if loc_prev == -1:
        final_trajectory.append([loc_prev, t_start,  t_end])

    return final_trajectory