# Infostop
*Python package for detecting stop locations in mobility data*

This package implements the algorithm described in (paper not written yet), for detecting stop locations in time-ordered location data.

## Usage
Given a location trace such as:

```Python
>>> data 
array([[ 55.75259295,  12.34353885 ],
       [ 55.7525908 ,  12.34353145 ],
       [ 55.7525876 ,  12.3435386  ],
       ...,
       [ 63.40379175,  10.40477095 ],
       [ 63.4037841 ,  10.40480265 ],
       [ 63.403787  ,  10.4047871  ]])
```

A stop location solution can be obtained using:

```Python
>>> import infostop
>>> labels = infostop.best_partition(data)
```

Here, `labels` matches `data` in size, and can easily be mapped back onto `data`:

```Python
>>> np.hstack([data, labels.reshape(-1, 1)])
array([[ 55.75259295,  12.34353885,   0.        ],
       [ 55.7525908 ,  12.34353145,   0.        ],
       [ 55.7525876 ,  12.3435386 ,   0.        ],
       ...,
       [ 63.40379175,  10.40477095, 164.        ],
       [ 63.4037841 ,  10.40480265, 164.        ],
       [ 63.403787  ,  10.4047871 , 164.        ]])
```

Plotting this onto a map:

![img](https://ulfaslak.com/files/infostop_example_map.png)

## Advantages
* **Simplicity**: At its core, the method works by two steps. (1) Reducing the location trace to the medians of each stationary event and (2) embedding the resulting locations into a network that connects locations that are within a user-defined distance and clustering that network.
* **Flow based**: Spatial clusters correspond to collections of location points that contain large amounts of flow when represented as a network. This enables the recovery of locations where traces slightly overlap.
* **Speed**: First the point space is reduced to the median of stationary points, then pairwise distances between these medians are computed using a vectorized implementation of the haversine function, and finally the resulting network at some distance threshold is clustered using the C++ based Infomap implementation. For example, clustering 70.000 location points takes aroung 16 seconds.

## Installation
`pip install infostop`
