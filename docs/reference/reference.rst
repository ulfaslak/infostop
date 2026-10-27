API reference
=============

The models and functions on this page can be imported from the top-level
``infostop`` package, for example ``from infostop import Infostop``.
:func:`~infostop.plot_map` returns a :class:`~infostop.visualize.FoliumMap`.

Models
------

.. autoclass:: infostop.Infostop
   :members: fit_predict, compute_label_medians

.. autoclass:: infostop.SpatialInfomap
   :members: fit_predict

Plotting
--------

.. autofunction:: infostop.plot_map

.. autoclass:: infostop.visualize.FoliumMap
   :members:

Post-processing
---------------

.. autofunction:: infostop.compute_intervals
