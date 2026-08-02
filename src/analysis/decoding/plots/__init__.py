"""Decoding figures. Selects the Matplotlib Agg backend for headless/cluster use."""
import matplotlib
from src.analysis.utils.mpl_backend import ensure_headless_backend
# Headless on the cluster; leaves a notebook's inline backend alone.
ensure_headless_backend()
from .style import NATURE_STYLE
