.. _usage:

Usage
=====

.. contents:: On this page
   :local:

Input data
----------

:class:`infostop.Infostop` labels a time-ordered trace. The trace is a NumPy
array with one row per point, in time order, and 2 or 3 columns:

* ``[latitude, longitude]``
* ``[latitude, longitude, time]``

With the default ``distance_metric="haversine"``, latitude and longitude are in
degrees. For coordinates that are not latitude and longitude, see
:ref:`euclidean`. Timestamps must not decrease from one row to the next. Their
unit is up to you; see :ref:`timestamps`.

Label a trace
-------------

The examples on this page use a small synthetic trace. A person stays at home,
walks to work, stays at work and walks home again. Their phone records one GPS
fix per minute, with timestamps in seconds.

.. code-block:: pycon

   >>> import numpy as np
   >>> from infostop import Infostop
   >>> rng = np.random.default_rng(0)
   >>> home = (55.6761, 12.5683)
   >>> work = (55.6867, 12.5700)
   >>> def stay(place, t0, n=10):
   ...     # n fixes, one per minute from time t0, about 1 m around `place`
   ...     lat = place[0] + rng.normal(0, 1e-5, n)
   ...     lon = place[1] + rng.normal(0, 1e-5, n)
   ...     return np.column_stack([lat, lon, t0 + 60 * np.arange(n)])
   >>> def trip(a, b, t0, n=3):
   ...     # n fixes, one per minute from time t0, evenly spaced from `a` to `b`
   ...     f = np.arange(1, n + 1) / (n + 1)
   ...     lat = a[0] + f * (b[0] - a[0])
   ...     lon = a[1] + f * (b[1] - a[1])
   ...     return np.column_stack([lat, lon, t0 + 60 * np.arange(n)])
   >>> data = np.vstack([
   ...     stay(home, 0),
   ...     trip(home, work, 600),
   ...     stay(work, 780),
   ...     trip(work, home, 1380),
   ...     stay(home, 1560),
   ... ])
   >>> data.shape
   (36, 3)

Create a model and fit it. :meth:`~infostop.Infostop.fit_predict` returns one
label per input row:

.. code-block:: pycon

   >>> model = Infostop()
   >>> labels = model.fit_predict(data)
   >>> labels
   array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  2,  2,  2,  2,
           2,  2,  2,  2,  2,  2, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1])

Both visits to home get the same label, and work gets another. Labels are
non-negative integers that only identify stop locations: their values and
their order carry no meaning. For traces where the stop locations are
ambiguous, the result can also differ between versions of Infomap.

The label ``-1`` marks points that are not part of any stop. A point gets
``-1`` if it is in no stationary event: the person was moving, or they stayed
for too few points (``min_size``) or too short a time (``min_staying_time``).
With ``label_singleton=False``, a stationary event that has no other event
within ``r2`` also gets ``-1``.

Choose ``r1`` and ``r2``
------------------------

Two distances control the result. Both are parameters of
:class:`infostop.Infostop`. With ``distance_metric="haversine"`` (the default)
they are in meters. With ``distance_metric="euclidean"`` they are in the units
of your coordinates.

``r1`` (default 10)
   How far a point can be from the median of the current group of points and
   still join the group. Set ``r1`` first. As a rule of thumb, make it
   slightly larger than the standard deviation of the location measurement
   error.

``r2`` (default 10)
   Stationary events whose medians are at most ``r2`` apart are linked in the
   network that Infomap clusters. A larger ``r2`` gives larger stop locations.

For example, for GPS data with a location error of about 20 m (standard
deviation):

.. code-block:: pycon

   >>> noisy_model = Infostop(r1=25, r2=25)

.. _timestamps:

Timestamps
----------

A third column with timestamps lets you set limits on time:

``min_staying_time`` (default 300)
   A stationary event must last at least this long, from its first to its last
   point. Shorter events get the label ``-1``.

``max_time_between`` (default 86400)
   The longest allowed time between two consecutive points in the same
   stationary event. A longer gap ends the event, even if the person did not
   move.

Both are in the units of your timestamp column. The defaults assume seconds
(5 minutes and 24 hours). If your timestamps are in milliseconds, scale the
values to match, for example
``Infostop(min_staying_time=300_000, max_time_between=86_400_000)``.

Without a time column, Infostop ignores both parameters, and only ``r1`` and
``min_size`` decide what a stationary event is. In this example the person
stops at a shop for only 2 minutes. With timestamps, the shop visit is too
short to be a stop. Without timestamps, its 3 points are enough:

.. code-block:: pycon

   >>> shop = (55.6800, 12.5800)
   >>> errand = np.vstack([
   ...     stay(home, 0),
   ...     trip(home, shop, 600),
   ...     stay(shop, 780, n=3),  # 3 fixes: 2 minutes at the shop
   ...     trip(shop, home, 960),
   ...     stay(home, 1140),
   ... ])
   >>> Infostop().fit_predict(errand)
   array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,
          -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
   >>> Infostop().fit_predict(errand[:, :2])
   array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  2,  2,  2, -1,
          -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1])

Several traces at once
----------------------

Pass a Python list of arrays, one per person (or one per trace segment), and
:meth:`~infostop.Infostop.fit_predict` returns a list of label arrays in the
same order. All traces share the same stop locations, so a place that two
people visit gets the same label in both traces. Each array has the format
described in `Input data`_, and all arrays must have the same number of
columns: either all with timestamps or all without.

.. code-block:: pycon

   >>> alice = data
   >>> bob = np.vstack([
   ...     stay(work, 0),
   ...     trip(work, shop, 600),
   ...     stay(shop, 780),
   ...     trip(shop, work, 1380),
   ...     stay(work, 1560),
   ... ])
   >>> labels_alice, labels_bob = Infostop().fit_predict([alice, bob])
   >>> labels_alice
   array([ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2, -1, -1, -1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1, -1, -1, -1,  2,  2,  2,  2,  2,  2,  2,  2,
           2,  2])
   >>> labels_bob
   array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  3,  3,  3,  3,
           3,  3,  3,  3,  3,  3, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1])

Work has label ``1`` in both traces. Alice's home now has label ``2``, not
``1`` as in the single-trace example: labels identify places within one fit
only.

Stop location medians
---------------------

After fitting, :meth:`~infostop.Infostop.compute_label_medians` returns a
dictionary that maps each label to the median ``[latitude, longitude]`` of
the medians of its stationary events:

.. code-block:: pycon

   >>> for label, (lat, lon) in model.compute_label_medians().items():
   ...     print(label, round(lat, 5), round(lon, 5))
   1 55.67611 12.5683
   2 55.6867 12.57

Stop and move intervals
-----------------------

:func:`infostop.compute_intervals` turns the labels of one trace into a list
of intervals. Each interval is a row ``[label, start_time, end_time]`` and
covers a run of consecutive points with the same label. Moves are the
intervals with label ``-1``. For the trace above:

.. code-block:: pycon

   >>> from infostop import compute_intervals
   >>> intervals = compute_intervals(labels, data[:, 2])
   >>> for label, start, end in intervals:
   ...     print(label, start, end)
   1.0 0.0 540.0
   -1.0 600.0 720.0
   2.0 780.0 1320.0
   -1.0 1380.0 1500.0
   1.0 1560.0 2100.0

Here the labels are floats, because the timestamps are.

For several traces, call it once per trace, with that trace's labels and
timestamps. Two consecutive points with the same label stay in one interval
only if the time between them is at most ``max_time_between`` (default
86400).

.. note::

   ``compute_intervals`` does not get ``max_time_between`` from the model.
   If you fitted the model with another value, pass the same value here, for
   example ``compute_intervals(labels, times, max_time_between=86_400_000)``
   for timestamps in milliseconds. With the default, one stay can be split
   into many intervals. See
   `issue #83 <https://github.com/ulfaslak/infostop/issues/83>`_.

Plot stop locations on a map
----------------------------

:func:`infostop.plot_map` draws the stop locations of a fitted model on an
interactive `folium <https://python-visualization.github.io/folium/>`_ map.
It works only with ``distance_metric="haversine"``. It returns an object
whose ``m`` attribute is the :class:`folium.Map`:

.. code-block:: pycon

   >>> from infostop import plot_map
   >>> folmap = plot_map(model)
   >>> type(folmap.m)
   <class 'folium.folium.Map'>

In a Jupyter notebook, put ``folmap.m`` as the last line of a cell to show the
map. Elsewhere, save it as an HTML file and open that in a browser:

.. code-block:: python

   folmap.m.save("stop_locations.html")

The ``display_data`` parameter selects which points to draw. The default,
``"unique_stationary"``, draws the medians of the stationary events that
belong to a stop (for :class:`~infostop.SpatialInfomap`, the input points
that are not labeled ``-1``). ``"all"`` draws every input point,
``"all_stationary"`` only the input points that belong to a stop, and
``"all_nonstationary"`` only the input points labeled ``-1``. Each location
is drawn once.

.. _euclidean:

Coordinates that are not latitude and longitude
-----------------------------------------------

For points in a plane, such as positions in meters in a projected coordinate
system or indoor positions, set ``distance_metric="euclidean"``. Then ``r1``,
``r2`` and ``min_spacial_resolution`` are in the units of your coordinates.
:func:`~infostop.plot_map` does not support these models.

.. code-block:: pycon

   >>> xy = np.vstack([
   ...     rng.normal(0, 1, (5, 2)),      # 5 points around (0, 0)
   ...     [[20, 0], [40, 0]],            # 2 points on the move
   ...     rng.normal([60, 0], 1, (5, 2)),  # 5 points around (60, 0)
   ... ])
   >>> Infostop(r1=3, r2=3, distance_metric="euclidean").fit_predict(xy)
   array([ 0,  0,  0,  0,  0, -1, -1,  1,  1,  1,  1,  1])

Cluster points without time order
---------------------------------

:class:`infostop.SpatialInfomap` clusters a set of points that have no time
order, such as check-ins or the stop locations of many people. It skips the
time step of Infostop: it links all points that are at most ``r2`` apart and
clusters that network with Infomap. The input is an array with 2 columns.

.. code-block:: pycon

   >>> from infostop import SpatialInfomap
   >>> points = np.vstack([
   ...     rng.normal(0, 0.5, (6, 2)),         # 6 points around (0, 0)
   ...     rng.normal([50, 50], 0.5, (3, 2)),  # 3 points around (50, 50)
   ...     [[100, 0]],                         # 1 point on its own
   ... ])
   >>> SpatialInfomap(r2=5, distance_metric="euclidean").fit_predict(points)
   array([1, 1, 1, 1, 1, 1, 2, 2, 2, 3])

The point on its own has no neighbor within ``r2``. It gets its own label
because ``label_singleton=True`` is the default. With
``label_singleton=False`` it gets ``-1``:

.. code-block:: pycon

   >>> SpatialInfomap(
   ...     r2=5, distance_metric="euclidean", label_singleton=False
   ... ).fit_predict(points)
   array([ 1,  1,  1,  1,  1,  1,  2,  2,  2, -1])

Large datasets
--------------

Clustering time grows with the number of distinct stationary event medians.
``min_spacial_resolution`` rounds the medians to multiples of its value before
Infostop removes duplicates, so nearby medians become one node in the network.
It is in coordinate units: degrees with ``"haversine"``. ``1e-5`` (about 1 m)
typically works well and has little effect on the result. Values above
``1e-4`` (about 11 m) are not recommended for geographic data. The default,
``0``, turns rounding off.

Migrate from ``label_trace``
----------------------------

Before version 0.1.0, Infostop had functions in an ``infostop.detect``
module. That module no longer exists, and the old functions raise an error
that shows what to use instead.

=========================================  =========================================
Old                                        New
=========================================  =========================================
``infostop.label_trace(data, ...)``        ``Infostop(...).fit_predict(data)``
``infostop.label_static_points(data)``     ``SpatialInfomap(...).fit_predict(data)``
``tolerance``                              ``min_spacial_resolution``
=========================================  =========================================

Most parameters keep their names and meaning. One default changed:
``label_trace`` used ``label_singleton=False``, and :class:`~infostop.Infostop`
uses ``label_singleton=True``. Pass ``label_singleton=False`` to keep the old
behaviour.
