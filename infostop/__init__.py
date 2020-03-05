#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package implements stop location detection using Infomap.
"""

from .deprecated import label_trace
from .deprecated import label_static_points
from .deprecated import label_network
from .models import Infostop
from .models import SpatialInfomap
from .visualize import plot_map
from .postprocess import compute_intervals
import cpputils

from .metadata import __version__, __name__
