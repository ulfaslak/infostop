#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

// ----------------
// Regular C++ code
// ----------------

static double haversine(double lat1, double lon1, double lat2, double lon2)
{
	// source: https://bit.ly/2S0OdUs
	// distance between latitudes
	// and longitudes
	double dLat = (lat2 - lat1) * M_PI / 180.0;
	double dLon = (lon2 - lon1) * M_PI / 180.0;

	// convert to radians
	lat1 = (lat1) * M_PI / 180.0;
	lat2 = (lat2) * M_PI / 180.0;

	// apply formulae
	double a = pow(sin(dLat / 2), 2) + pow(sin(dLon / 2), 2) * cos(lat1) * cos(lat2);
	double rad = 6371000;
	double c = 2 * asin(sqrt(a));
	return rad * c;
}

static double euclidean(double x1, double y1, double x2, double y2)
{
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}


double median(std::vector<double> &values)
{
	// the vector comes in pre-sorted
	// if (!std::is_sorted(values.begin(), values.end()))
	// {
	// 	std::cout << "!";
	// }

	// Find the two middle positions (they will be the same if size is odd)
	int i0 = (values.size()-1) / 2;
	int i1 = values.size() / 2;
	return 0.5 * (values[i0] + values[i1]);
}

double normalize_longitude_360(double longitude)
{
	longitude = std::fmod(longitude, 360.0);
	if (longitude < 0.0)
	{
		longitude += 360.0;
	}
	return longitude;
}

double median_longitude(std::vector<double> &values)
{
	// Haversine longitudes are kept sorted in [0, 360).
	// Unwrap at the largest gap so a group near the antimeridian stays contiguous.
	double largest_gap = values.front() + 360.0 - values.back();
	size_t gap_end = values.size() - 1;
	for (size_t i = 0; i + 1 < values.size(); i++)
	{
		double gap = values[i + 1] - values[i];
		if (gap > largest_gap)
		{
			largest_gap = gap;
			gap_end = i;
		}
	}

	const size_t start = (gap_end + 1) % values.size();
	std::vector<double> unwrapped;
	unwrapped.reserve(values.size());
	for (size_t offset = 0; offset < values.size(); offset++)
	{
		size_t index = (start + offset) % values.size();
		double longitude = values[index];
		if (index < start)
		{
			longitude += 360.0;
		}
		unwrapped.push_back(longitude);
	}

	double result = std::fmod(median(unwrapped), 360.0);
	if (result > 180.0)
	{
		result -= 360.0;
	}
	else if (result < -180.0)
	{
		result += 360.0;
	}
	return result;
}

void insert_ordered(std::vector<double>& arr, double elem)
{
	// add space
    arr.push_back(elem);

    // find the position for the element
    auto pos = std::upper_bound(arr.begin(), arr.end()-1, elem);

    // and move the array around it:
    std::move_backward(pos, arr.end()-1, arr.end());

    // and set the new element:
    *pos = elem;
};


//std::vector<py::array_t<double>> get_stationary_events(
std::tuple<std::vector<std::vector<double>>, std::vector<int>> get_stationary_events(
	py::array_t<double> input,
	double r_C,
	int min_size,
	double min_staying_time,
	double max_staying_time,
	std::string distance_metric
	)
{
	auto coords = input.unchecked<2>();

	// Get shape of data
	py::buffer_info input_buf = input.request();
	int N = input_buf.shape[0];
	int M = input_buf.shape[1];

	// Output variables
	std::vector<std::vector<double>> stat_coords;  // stat_coords (py: list of lists)
	std::vector<int> event_map(N);                 // event_map   (py: list)

	// Intermediate variables
	std::vector<double> stop_points_lat;
	std::vector<double> stop_points_lon;
	double ddist;
	double dtime;
	int i0 = 0;                           // index at which stops begin
	int j = 0;                            // index of stat_coords
	int outlier = -1;

	// Prepare distance function
	double (* distance_function)(double, double, double, double);
	double (* longitude_median_function)(std::vector<double>&) = &median;
	const bool is_haversine = distance_metric == "haversine";
	if (is_haversine)
	{
		distance_function = &haversine;
		longitude_median_function = &median_longitude;
	} else if (distance_metric == "euclidean")
	{
		distance_function = &euclidean;
	}

	if (M == 2)
	{

		// Cluster with no time information //
		// -------------------------------- //

		// Set current group
		stop_points_lat = { coords(0, 0) };
		double first_longitude = coords(0, 1);
		if (is_haversine)
		{
			first_longitude = normalize_longitude_360(first_longitude);
		}
		stop_points_lon = { first_longitude };

		// Loop over points
		for (size_t i = 1; i < N; i++)
		{
			double longitude = coords(i, 1);
			if (is_haversine)
			{
				longitude = normalize_longitude_360(longitude);
			}

			// compute distance to median of previous group
			ddist = distance_function(
				coords(i, 0), coords(i, 1),
				median(stop_points_lat), longitude_median_function(stop_points_lon)
			);

			if (ddist <= r_C)
			{
				// append to current group
				insert_ordered(stop_points_lat, coords(i, 0));
				insert_ordered(stop_points_lon, longitude);
			} else
			{
				// test if there are enough points in the stop
				if (i - i0 >= min_size)
				{
					// add previous group median to `stat_coords`,
					stat_coords.push_back(
						{ median(stop_points_lat), longitude_median_function(stop_points_lon) }
					);

					// add indices to event_map
					for (size_t idx = i0; idx < i; idx++)
					{
						event_map[idx] = j;
					}

					// increment group index
					j += 1;

				} else
				{
					// add indices to event_map
					for (size_t idx = i0; idx < i; idx++)
					{
						event_map[idx] = outlier;
					}
				}

				// clear current groups
				stop_points_lat.clear();
				stop_points_lon.clear();

				// and write current coordinates as new group
				stop_points_lat = { coords(i, 0) };
				stop_points_lon = { longitude };

				// reset i0 index
				i0 = i;
			}
		}
		// append the last group (compact version of what's inside the above for->else)
		if (N - i0 >= min_size)
		{
			stat_coords.push_back({ median(stop_points_lat), longitude_median_function(stop_points_lon) });
			for (size_t idx = i0; idx < N; idx++)
				event_map[idx] = j;
		} else
		{
			for (size_t idx = i0; idx < N; idx++)
				event_map[idx] = outlier;
		}
	}
	else if (M == 3)
	{

		// Cluster WITH time information //
		// ----------------------------- //

		// Set current group
		stop_points_lat = { coords(0, 0) };
		double first_longitude = coords(0, 1);
		if (is_haversine)
		{
			first_longitude = normalize_longitude_360(first_longitude);
		}
		stop_points_lon = { first_longitude };

		// Loop over points
		for (size_t i = 1; i < N; i++)
		{
			double longitude = coords(i, 1);
			if (is_haversine)
			{
				longitude = normalize_longitude_360(longitude);
			}

			// compute distance to median of previous group
			ddist = distance_function(
				coords(i, 0), coords(i, 1),
				median(stop_points_lat), longitude_median_function(stop_points_lon)
			);
			dtime = coords(i, 2) - coords(i-1, 2);

			if (ddist <= r_C && dtime <= max_staying_time)
			{
				// append to current group
				insert_ordered(stop_points_lat, coords(i, 0));
				insert_ordered(stop_points_lon, longitude);
			} else
			{
				// test if there are enough points in the stop and the stop lasts long enough
				if (i - i0 >= min_size && coords(i-1, 2) - coords(i0, 2) >= min_staying_time)
				{
					// add previous group median to `stat_coords`,
					stat_coords.push_back(
						{ median(stop_points_lat), longitude_median_function(stop_points_lon) }
					);

					// add indices to event_map
					for (size_t idx = i0; idx < i; idx++)
					{
						event_map[idx] = j;
					}

					// increment group index
					j += 1;

				} else
				{
					// add indices to event_map
					for (size_t idx = i0; idx < i; idx++)
					{
						event_map[idx] = outlier;
					}
				}

				// clear current groups
				stop_points_lat.clear();
				stop_points_lon.clear();

				// and write current coordinates as new group
				stop_points_lat = { coords(i, 0) };
				stop_points_lon = { longitude };

				// reset i0 index
				i0 = i;
			}
		}

		// append the last group
		if (N - i0 >= min_size && coords(N-1, 2) - coords(i0, 2) >= min_staying_time)
		{
			stat_coords.push_back({ median(stop_points_lat), longitude_median_function(stop_points_lon) });
			for (size_t idx = i0; idx < N; idx++)
				event_map[idx] = j;
		} else
		{
			for (size_t idx = i0; idx < N; idx++)
				event_map[idx] = outlier;
		}
	}

	return std::make_tuple(stat_coords, event_map);
}


// ----------------
// Python interface
// ----------------

PYBIND11_MODULE(cpputils, m)
{
    m.doc() = R"pbdoc(
	    Infostop C++ plugin
	    -------------------
	    .. currentmodule:: infostop
	    .. autosummary::
	       :toctree: _generate
	       get_stationary_events
	)pbdoc";

	m.def("get_stationary_events", &get_stationary_events, R"pbdoc(
        Group temporally adjacent points if they are closer than r_C,
        then save their median (`stat_coords`) and store the median
        index in vector that maps it to `coords` indices.

	    Parameters
	    ----------
	        coords : array-like (shape=(N, 2) or shape=(N,3))
	        r_C : number (critical radius)
	        min_size : int,
	        min_staying_time : int
	        max_staying_time : int
	        distance_metric : str

	    Returns
	    -------
	        groups : list-of-list
	            Each list is a group of points
    )pbdoc");
}
