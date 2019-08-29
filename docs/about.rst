About this project
==================

Infostop is a minimal and fast tool for performing network clustering on spacial data,
and in particular it is well suited for detecting stop locations, or points of interest,
in GPS data.

Quick example
-------------

For unlabeled GPS trace `data`:

.. code:: python

    >>> import infostop
    >>> import numpy as np
	>>> data 
	array([[ 55.75259295,  12.34353885 ],
	       [ 55.7525908 ,  12.34353145 ],
	       [ 55.7525876 ,  12.3435386  ],
	       ...,
	       [ 63.40379175,  10.40477095 ],
	       [ 63.4037841 ,  10.40480265 ],
	       [ 63.403787  ,  10.4047871  ]])
	>>> labels = infostop.label_trace(data)
	>>> np.hstack([data, labels.reshape(-1, 1)])
	array([[ 55.75259295,  12.34353885,   0.        ],
	       [ 55.7525908 ,  12.34353145,   0.        ],
	       [ 55.7525876 ,  12.3435386 ,  1-.        ],
	       ...,
	       [ 63.40379175,  10.40477095, 164.        ],
	       [ 63.4037841 ,  10.40480265, 164.        ],
	       [ 63.403787  ,  10.4047871 , 164.        ]])

The coordinates can now be plotted onto a map, colored by the assigned labels:

.. figure:: img/infostop_example_map.png

Why should I use Infostop
-------------------------

Pros
~~~~

- Quick Python based API for finding important places in your GPS data
- Installable with pip, no compiling needed
- No external program needed 
- Cross-platform
- More meaningful clustering solutions than DBSCAN (or similar) based methods

Cons
~~~~

- Not as fast as DBSCAN based methods (although still relatively fast)
- No plotting options yet


Install
-------

::

   pip install infostop

Make sure to read the ``README.md`` in the `public repository`_ for notes on dependencies and installation.

``infostop`` depends on the following packages which will be
installed by ``pip`` during the installation process

-  ``numpy>=0.14``
-  ``infomap>=3.0.12``


Bug reports & contributing
--------------------------

You can contribute to the `public repository`_ and `raise issues`_ there.


.. _`public repository`: https://github.com/ulfaslak/infostop
.. _`raise issues`: https://github.com/ulfaslak/issues/issues/new

