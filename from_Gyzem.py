#!/usr/bin/env python
# coding: utf-8
# > Hansen, Anders S., Iryna Pustova, Claudia Cattoglio, 
# Robert Tjian, and Xavier Darzacq. “CTCF and Cohesin Regulate 
# Chromatin Loop Stability with Distinct Dynamics.” 
# bioRxiv, 2016, 093476.

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

# In[67]:


from glob import glob

# In[70]:
# A2250_inliq40forAndrey A22inliq30forAndrey
data_paths = glob('/Users/gizemozbaykal/Dropbox/G5MCM/Gizem/forAndrey/*WT*.mat')
data_paths

# In[71]:
# 1 2 3 5
data_path = data_paths[0]

print(data_path)
# In[92]:


all_exp = matimport.read_gizem_mat(data_path)
print(all_exp)
#cell4 = matimport.concat_all(all_exp, exposure_ms=60., pixel_size_um=0.075)
reps = matimport.concat_reps(all_exp, exposure_ms=60., pixel_size_um=0.075) #075


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
                 useZcorr=False,
                 plot_hist=False,
                 plot_result=True ) 

def my_fit(rep):
    
    cell_spt = readers.to_fastSPT(rep, from_json=False)
    fit_result = tools.auto_fit(cell_spt,
                                fit_params=fit_params,
                                )
    return fit_result

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
