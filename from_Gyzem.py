#!/usr/bin/env python
# coding: utf-8

# A short introduction to fast SPT modeling
# -----------------------------------------
# 
# This notebook presents a short introduction to the modeling of 
# fast single particle tracking datasets. 
# The methodology, together with examples of biological 
# relevance, can be found at:
# 
# > Hansen, Anders S., Iryna Pustova, Claudia Cattoglio, 
# Robert Tjian, and Xavier Darzacq. “CTCF and Cohesin Regulate 
# Chromatin Loop Stability with Distinct Dynamics.” 
# bioRxiv, 2016, 093476.
# 
# This package includes several sample datasets, that will be 
# used by this notebook. Make sure that they can be found by Python.

# ## 1. Loading of a dataset
# 
# To help us with basic and repeated tasks such as dataset loading, 
# we created a small library, `fastSPT_tools` that contain 
# several helper functions that we will use all across this tutorial. 
# We thus need to import it by typing `import fastSPT_tools`.
# 
# One of the functions list the available datasets: 
# `list_sample_datasets(path)`. Let's first see what datasets 
# we can get. Then, we will use the `load_dataset(path, dataset_id, 
# cells)` function to load the relevant dataset. This latter function 
# can either load one single cell or a series of cells 
# (identified by their id).

# In[1]:
# jlkbli

import sys
from tqdm.autonotebook import tqdm
sys.version

#sys.path.append(r'C:\Users\andre\Documents\Spot-On-cli')


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import lmfit

import fastspt as fs
from fastspt import fit, tools, plot, readers, matimport, tracklen
# get_ipython().run_line_magic('load_ext', 'autoreload')

# get_ipython().run_line_magic('autoreload', '2')

print("Using fastspt version {}".format(fs.__version__))


# In[3]:


import os
os.getcwd()

xml_path = r"movie_continuos_exp_60ms_561_50W_405_12_Tracks_filter_min_intensity.xml"
path_wt = r"/Users/gizemozbaykal/Dropbox/G5MCM/Gizem/forAndrey/WTforAndrey.mat"
#path_oe = r"Z:\Andrey\fromGizem\2019-04-18-PBP2-tracks\PBP2OverExpforAndrey.mat"

# In[67]:


from glob import glob


# In[70]:

data_paths = glob('/Users/gizemozbaykal/Dropbox/G5MCM/Gizem/forAndrey/*.mat')

data_paths


# In[91]:

data_path = data_paths[0]


# In[92]:


all_exp = matimport.read_gizem_mat(data_path)
#cell4 = matimport.concat_all(all_exp, exposure_ms=60., pixel_size_um=0.075)
reps = matimport.concat_reps(all_exp, exposure_ms=60., pixel_size_um=0.075)


# # 2. Track lengths analysis

# In[93]:


for rep in reps:
    tracklen.get_track_lengths_dist(rep)


# ## 3. Fitting of the jump lengths distribution to the model (2 states)
# 

# In[94]:


## Generate a dictionary of parameters
fit_params = dict(states=2,
                 iterations=1,
                 CDF=False,
                 CDF1 = True,
                 Frac_Bound = [0, 1],
                 D_Free = [0.01, 1.],
                 D_Med = [0.005, 0.1],
                 D_Bound = [0.0, 0.005],
                 sigma = 0.02,
                 sigma_bound = [0.005, 0.1],
                 fit_sigma=True,
                 dT=0.06,
                 dZ=0.7,
                 a=0.15716,
                 b=0.20811,
                 useZcorr=False ) 

def my_fit(rep):
    
    cell_spt = readers.to_fastSPT(rep, from_json=False)
    fit_result = tools.auto_fit(cell_spt,
                                fit_params=fit_params,
                                plot_hist=False,
                                plot_result=True)
    return fit_result
print('geldi')

reps_fits = list(map(my_fit, reps))


# In[95]:
print(reps_fits)

#get stats
fit_stats = pd.DataFrame(columns=list(reps_fits[0].best_values.keys()) + ['chi2'])
for i, fit_result in enumerate(reps_fits):
    fit_stats.loc[f'rep {i+1}'] = list(fit_result.best_values.values()) + [fit_result.chisqr]
    
fit_stats.loc['mean'] = fit_stats.mean(axis=0)
fit_stats.loc['std'] = fit_stats.std(axis=0)

fit_stats.to_json(data_path + '.stats.json')

print(fit_stats)
#fit_stats.to_excel(path_oe + '.stats.xls')
