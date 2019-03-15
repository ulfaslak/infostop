# Infostop
*Python package for detecting stop locations in mobility data*

This package implements the algorithm described in (paper not written yet), for detecting stop locations in time-ordered location data.

## Usage
Given a location trace such as:

![img](https://ulfaslak.com/files/infostop_example_code1.png)

A stop location solution can be obtained like:

![img](https://ulfaslak.com/files/infostop_example_code2.png)

Plotting it onto a map:

![img](https://ulfaslak.com/files/infostop_example_map.png)

## Advantages
* **Simplicity**: At its core, the method works by two steps. (1) Reducing the location trace to the medians of each stationary event and (2) embedding the resulting locations into a network that connects locations that are within a user-defined distance and clustering that network.
* **Flow based**: Spatial clusters correspond to collections of location points that contain large amounts of flow when represented as a network. This enables the recovery of locations where traces slightly overlap.
* **Speed**: First the point space is reduced to the median of stationary points, then pairwise distances between these medians are computed using a vectorized implementation of the haversine function, and finally the resulting network at some distance threshold is clustered using the C++ based Infomap implementation. For example, clustering 70.000 location points takes aroung 16 seconds.

## Installation
`pip install infostop`