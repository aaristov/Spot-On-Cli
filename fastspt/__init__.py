from fastspt import *
from version import __version__

import fastSPT_tools, readers, writers
try:
	import fastSPT_plot
except Exception, e:
	print "Could not import the plot submodule, error:"
	print e
