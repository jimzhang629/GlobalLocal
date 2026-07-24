"""Decoding figures. Sets the Matplotlib Agg backend for headless/cluster use."""
import matplotlib
matplotlib.use('Agg')
from .style import NATURE_STYLE
