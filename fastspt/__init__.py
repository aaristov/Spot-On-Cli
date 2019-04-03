from fastspt import *
from fastspt.version import __version__

from fastspt import tools, readers, writers
try:
	from fastspt import plot
except Exception as e:
	print("Could not import the plot submodule, error:")
	print(e)
