from fastspt import *
from fastspt.version import __version__

from fastspt import fastSPT_tools, readers, writers
try:
	import fastSPT_plot
except Exception as e:
	print("Could not import the plot submodule, error:")
	print(e)
