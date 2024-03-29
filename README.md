# Infostop
*Python package for detecting stop locations in mobility data*

[![Build Status][build-image]][build-url]

This package implements the algorithm described in https://arxiv.org/pdf/2003.14370.pdf, for detecting stop locations in time-ordered location data.

Infostop is useful to anyone who wishes to detect stationary events in location coordinate streams. It is, thus, a framework to simplify dense and rich location time-series into sequences of events.

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

## Advantages
* **Simplicity**: At its core, the method works by two steps. (1) Reducing the location trace to the medians of each stationary event and (2) embedding the resulting locations into a network that connects locations that are within a user-defined distance and clustering that network.
* **Multi-trace support:** Currently, no other libraries support clustering multiple traces at once to find global stop locations. Infostop does. The image above visualizes stop locations at a campus for a population of almost 1000 university students.
* **Flow based**: Spatial clusters correspond to collections of location points that contain large amounts of flow when represented as a network. This enables the recovery of locations where traces slightly overlap.
* **Speed**: First the point space is reduced to the median of stationary points (executed in a fast C++ module), then spatially neighboring points connected using a Ball search tree algorithm, and finally the network is clustered using the C++ based Infomap program. For example, clustering 100.000 location points takes about a second.

## Installation
`pip install infostop`

## Development notes
We welcome contributions. Before you get started, you may want to read the notes below.

You should **create a virtual environment**. In your local `infostop` folder, do:
```Bash
$ make env
```

**Install `infostop`** into your virtual environment.
Do this by running:
```Bash
(env) $ make install
```
This command will also delete any pre-existing installation of Infostop, so you will probably want to run it after each code update.

**Run tests**:
```Bash
(env) $ make test
```

**Check test coverage**:
```Bash
(env) $ make coverage
(env) $ cd htmlcov
(env) $ python -m http.server 8001
```
Then go to localhost:8001 in your browser to look at the coverage report.

**Format code with `black`**. We don't want to argue about code formatting. Please run `black` to apply standard formatting to your code before your make a pull request.

The `Makefile` implements a number of commands that are useful during development.
Go ahead and execute `make help` to see descriptions of available commands, or inspect the file so you understand what's going on. 

**Convenient: create an ipykernel for the virtual environment**
If you use Jupyter notebooks, you can install the virtual environment into Jupyter as a kernel. Run:
```Bash
(env) $ pip install ipykernel
(env) $ python -m ipykernel install --user --name=infostop_env
```
This lets you select the virtual environment as a kernel in a Jupyter notebook.

**Versioning and deployment to PyPI**
If your update should trigger a version increment and package rerelease, please execute the `increment_version.py` script ONCE and [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) your final commit. After running the `commit` command, to tag the commit you would run something like:
```Bash
(env) $ git tag -a v1.0.11 -m "Infostop version 1.0.11"
```
Finally, push first the tags and then your commits.
```Bash
(env) $ git push --tags && git push
```
When mergining a PR with a tagged commit, the PyPI deployment action is triggered, and the new version of Infostop becomes publicly available shortly thereafter.


[build-image]: https://github.com/ulfaslak/infostop/actions/workflows/deploy.yml/badge.svg
[build-url]: https://github.com/ulfaslak/infostop/actions/workflows/deploy.yml