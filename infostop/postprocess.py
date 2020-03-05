import numpy as np

def compute_intervals(coords, coord_labels, max_time_between=86400, distance_metric="haversine"):
    """Compute stop and moves intervals from the list of labels.
    
    Parameters
    ----------
        coords : np.array (shape=(N, 2) or shape=(N,3))
        coord_labels: 1d np.array of integers

    Returns
    -------
        intervals : array-like (shape=(N_intervals, 5))
            Columns are "label", "start_time", "end_time", "latitude", "longitude"
    
    """
    
    if coords.shape[1] == 2:
        times = np.array(list(range(0, len(coords))))
        coords = np.hstack([coords, times.reshape(-1,1)])
        
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
        if (loc==loc_prec) and (time-t_end)<(max_time_between):
            t_end = time
            median_lat.append(lat)
            median_lon.append(lon)
            
            
        #if the location name has changed build the interval and reset values
        else:
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