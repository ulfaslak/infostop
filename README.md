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

Or with time information

```Python
>>> data 
array([[ 55.75259295,  12.34353885, 1581401760 ],
       [ 55.7525908 ,  12.34353145, 1581402760 ],
       [ 55.7525876 ,  12.3435386 , 1581403760 ],
       ...,
       [ 63.40379175,  10.40477095, 1583401760 ],
       [ 63.4037841 ,  10.40480265, 1583402760 ],
       [ 63.403787  ,  10.4047871 , 1583403760 ]])
```

A stop location solution can be obtained using:

```Python
>>> from infostop import Infostop
>>> model = Infostop()
>>> labels = model.fit_predict(data)
```

Alternatively, `data` can also be a list of `numpy.array`s, in which case it is assumed that list elements are seperate traces in the same space. In this *multi segment* (or *multi user*) case, Infostop finds stop locations that are shared by different segments.

Solutions can be plotted using:

```Python
>>> from infostop import plot_map
>>> folmap = plot_map(model)
>>> folmap.m
```

Plotting this onto a map:

![img](https://ulfaslak.com/files/infostop_example_geomap.png)

For more examples and full documentation check out the [documentation](https://infostop.readthedocs.io/en/latest/about.html) page.

## Advantages
* **Simplicity**: At its core, the method works by two steps. (1) Reducing the location trace to the medians of each stationary event and (2) embedding the resulting locations into a network that connects locations that are within a user-defined distance and clustering that network.
* **Multi-trace support:** Currently, no other libraries support clustering multiple traces at once to find global stop locations. Infostop does. The image above visualizes stop locations at a campus for a population of almost 1000 university students.
* **Flow based**: Spatial clusters correspond to collections of location points that contain large amounts of flow when represented as a network. This enables the recovery of locations where traces slightly overlap.
* **Speed**: First the point space is reduced to the median of stationary points (executed in a fast C++ module), then spatially neighboring points connected using a Ball search tree algorithm, and finally the network is clustered using the C++ based Infomap program. For example, clustering 100.000 location points takes about a second.

## Installation
`pip install infostop`
