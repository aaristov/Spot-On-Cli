# TODO: Paramters of fit are ignores when calling module from comman line. In iPython works fine. 

## fastSPT_tools
## Some tools for the fastSPT package
## By MW, GPLv3+
## March 2017

## ==== Imports
import pickle, sys, scipy.io, os
import numpy as np
from fastspt import plot, fit, matimport
from fastspt.fit import fit_kinetics
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt

tp.ignore_logging()

def open_ts_table(path, verbose=0):
    
    ts_table = pd.read_csv(path)
    if verbose: ts_table.head()
    return ts_table

def get_loc_numbers(ts_table:pd.DataFrame, plot=True):
    '''
    Analyses number of particles per frame, returns the list with the length of number of frames
    '''
    _, idx = np.unique(ts_table.frame, return_index=True)
    num_locs_per_frame = idx[1:] - idx[:-1]
    if plot:
        plt.plot(num_locs_per_frame)
    return num_locs_per_frame

def open_and_link_ts_table(path, min_frame=None, max_frame=None, exposure_ms=60, link_distance_um=0.3, link_memory=1, verbose=0, loc_num_plot=True):
    ts_table = open_ts_table(path)
    grouped_tracks = link_ts_table(ts_table, min_frame, max_frame, exposure_ms, link_distance_um, link_memory, verbose, loc_num_plot)
    return grouped_tracks

def link_ts_table(ts_table:pd.DataFrame, min_frame=None, max_frame=None, exposure_ms=60, link_distance_um=0.3, link_memory=1, verbose=0, loc_num_plot=True):
    '''
    links particles with trackpy and groups particles into tracks using their index.
    '''
    df = pd.DataFrame(columns = ['x', 'y', 'frame'], data=ts_table[['x [nm]', 'y [nm]', 'frame']].values)    
    
    
    df.x = df.x / 1000 # nm -> um
    df.y = df.y / 1000 # nm -> um
    if min_frame:
        df = df[df.frame >= min_frame]
    if max_frame:
        df = df[df.frame <= min_frame]
    if verbose: df.head()
    tracks = tp.link_df(df, search_range=link_distance_um, memory=link_memory)
    if verbose: print(tracks.head())
    # print('\n')
    grouped_tracks =  matimport.group_tracks(tracks, exposure_ms=exposure_ms)
    
    return grouped_tracks

def get_low_density_frame(num_locs_per_frame:list, max_locs=200):
    assert len(num_locs_per_frame) > 10

    if max(num_locs_per_frame) > max_locs:
        peak = np.argmax(num_locs_per_frame)
        # print(peak)
        indices_with_fewer_locs = np.where(num_locs_per_frame[peak:] > max_locs)[0]
        # print(indices_with_fewer_locs)
        try:
            thr = indices_with_fewer_locs[-1]
        except IndexError:
            thr = 0
        return thr + peak
    else:
        return 0
    
    
## ==== Sample dataset-related functions
def list_sample_datasets(path):
    """Simple relay function that allows to list datasets from a datasets.py file"""
    sys.path.append(path)
    import datasets
    #reload(datasets) # Important I think
    return datasets.list(path, string=True)

def load_dataset(path, datasetID, cellID):
    """Simple helper function to load one or several cells from a dataset"""
    ## Get the information about the datasets
    sys.path.append(path)
    import datasets
    #reload(datasets) # Important I think
    li = datasets.list(path, string=False)

    if type(cellID) == int:
        cellID = [cellID]
    
    try: ## Check if our dataset(s) is/are available
        for cid in cellID:
            if not li[1][datasetID][cid].lower() == "found":
                raise IOError("This dataset does not seem to be available. Either it couldn't be found or it doesn't exist in the database.")
    except:
        raise IOError("This dataset does not seem to be available. Either it couldn't be found or it doesn't exist in the database or there is a problem with the database.")

    da_info = li[0][datasetID]

    ## Load the datasets
    AllData = []
    for ci in cellID:
        mat = scipy.io.loadmat(os.path.join(path,
                                            da_info['path'],
                                            da_info['workspaces'][ci]))
        AllData.append(np.asarray(mat['trackedPar'][0]))
    return np.hstack(AllData) ## Concatenate them before returning

def load_matlab_dataset_from_path(path):
    """Returns a dataset object from a Matlab file"""
    mat = scipy.io.loadmat(path)
    return np.asarray(mat['trackedPar'][0])


def auto_fit(cell_spt, fit_params ):
    '''
    Generates histograms and fits kinetic model according to intialization dictionary fit_params
    
    returns:
    
    lmfit.model.ModelResult
    
    '''
    

    jump_histrogram = get_jump_length_histrogram(cell_spt, **fit_params)
    
    if fit_params['plot_hist']: plot_hist_jumps(jump_histrogram)
    
    
    print(f'Fitting {fit_params["states"]} states')
    
    fit_result = fit_kinetics(jump_histrogram,
                             **fit_params)
    
    if fit_params['plot_result']: plot.plot_kinetics_fit(jump_hist=jump_histrogram,  
                                            fit_result=fit_result, **fit_params)
    
    return fit_result
    
def get_jump_length_histrogram(cell_spt, CDF=False, CDF1 = True, **kwargs):
    '''
    Computes JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF from tracks list.
    '''
    h1 = fit.compute_jump_length_distribution(cell_spt, CDF=CDF1, useEntireTraj=False)

    print("Computation of jump lengths performed in {:.2f}s".format(h1[-1]['time']))
    return h1


def plot_hist_jumps(jump_histrogram):
    HistVecJumps = jump_histrogram[2]
    JumpProb = jump_histrogram[3]
    plot.plot_histogram(HistVecJumps, JumpProb) ## Read the documentation of this function to learn how to populate all the 'na' fields
    return True
