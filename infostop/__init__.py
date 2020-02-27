#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package implements stop location detection using Infomap.
"""

from .detect import label_trace
from .detect import label_static_points
from .detect import label_network
import cpputils

from .metadata import __version__, __name__
