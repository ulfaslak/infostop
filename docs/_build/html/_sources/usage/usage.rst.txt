.. _usage:

Label location trace
--------------------

For a time-ordered series of coordinates, such as a GPS trace, annotate each point with a location label. Such a coordinate series may look like:

.. code:: python

	array([[ 55.75259295,  12.34353885 ],
	       [ 55.7525908 ,  12.34353145 ],
	       [ 55.7525876 ,  12.3435386  ],
	       ...,
	       [ 63.40379175,  10.40477095 ],
	       [ 63.4037841 ,  10.40480265 ],
	       [ 63.403787  ,  10.4047871  ]])

The trace can then be labeled using :mod:`infostop.detect.label_trace`:

.. code:: python

	labels = infostop.label_trace(data)
	labels
	# array([0, 0, -1, ..., 164, 164, 164])

And stacked with the input data:

.. code:: python
	
	np.hstack([data, labels.reshape(-1, 1)])
    array([[ 55.75259295,  12.34353885,   0.        ],
           [ 55.7525908 ,  12.34353145,   0.        ],
           [ 55.7525876 ,  12.3435386 ,  -1.        ],
           ...,
           [ 63.40379175,  10.40477095, 164.        ],
           [ 63.4037841 ,  10.40480265, 164.        ],
           [ 63.403787  ,  10.4047871 , 164.        ]])

Labels 0 and 164 in this case correspond to locations where the user was static, and possibly returned. The label -1 is given to location measurements which are recorded while the user is not stationary.

It is relevant for the user to consider first setting the ``r1`` parameter in :mod:`infostop.detect.label_trace`, which sets the threshold for the maximum distance between two consecutive location measurements allowed within the same stop location. As a rule of thumb, ``r1`` should be slightly greater than the standard deviation of the uncertainty distribution on the location measurement.

To get bigger stop locations consider increasing ``r2``, which sets the distance threshold for link creation.

If the input data is not GPS points, the ``distance_function`` parameter should be set to a valid metric, and ``r1`` and ``r2`` should most likely be specified to something meaningful. Per default, Infostop assumes that points are (lat, lon) and uses the haversine distance function, where the thresholds are given in meters.

Label timestamped trace
~~~~~~~~~~~~~~~~~~~~~~~

In a third (rightmost) column, timestamps can be listed so the input series looks like:

.. code:: python

	array([[ 55.75259295,  12.34353885,  1356998400 ],
	       [ 55.7525908 ,  12.34353145,  1356998700 ],
	       [ 55.7525876 ,  12.3435386 ,  1356999000 ],
	       ...,
	       [ 63.40379175,  10.40477095,  1357085400 ],
	       [ 63.4037841 ,  10.40480265,  1357085700 ],
	       [ 63403787  ,  10.4047871  ,  1357086000 ]])

This is useful, because it allows the user to specify ``min_staying_time`` which sets a lower bound on how short stops can be, and ``max_time_between`` which sets an upper bound on the longest allowed time between two samples before a new stop event is created.

