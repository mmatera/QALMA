#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Tue Sep 12 19:07:05 2023.

@author: mauricio
"""

import logging as _logging

from . import (
    evolution,
    geometry,
    io,
    maxent,
    meanfield,
    model,
    operators,
    projections,
    scalarprod,
    utils,
)
from .alpsmodels import list_models_in_alps_xml, model_from_alps_xml
from .geometry import graph_from_alps_xml, list_geometries_in_alps_xml
from .model import FermionicSignWarning, build_system
from .version import __version__

# Library logging: QALMA never configures handlers or levels. Messages go to
# the "qalma" logger hierarchy and are silent below WARNING unless the
# application configures logging (e.g. ``logging.basicConfig(level="INFO")``
# or ``logging.getLogger("qalma").setLevel("DEBUG")``).
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "FermionicSignWarning",
    "__version__",
    # Top-level factories
    "build_system",
    # Geometry / model
    "geometry",
    "graph_from_alps_xml",
    "list_geometries_in_alps_xml",
    "list_models_in_alps_xml",
    "model",
    "model_from_alps_xml",
    # Core subpackages
    "evolution",
    "io",
    "maxent",
    "meanfield",
    "operators",
    "projections",
    "scalarprod",
    # Utilities
    "utils",
]
