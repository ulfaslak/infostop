#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package implements stop location detection using Infomap.
"""

from .detect import label_trace
from .detect import best_partition
from .detect import label_static_points
from .detect import label_distance_matrix

name = "infostop"

__version__ = "0.0.13"
__author__ = "Ulf Aslak"
__copyright__ = "Copyright 2019, Ulf Aslak"
__credits__ = ["Ulf Aslak"]
__license__ = "MIT"
__maintainer__ = "Ulf Aslak"
__email__ = "ulfjensen@gmail.com"
__status__ = "Development"
