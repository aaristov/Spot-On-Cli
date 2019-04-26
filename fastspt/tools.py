## fastSPT_tools
## Some tools for the fastSPT package
## By MW, GPLv3+
## March 2017

## ==== Imports
import pickle, sys, scipy.io, os
import numpy as np
from fastspt import plot, fit
from fastspt.fit import fit_kinetics


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


def auto_fit(cell_spt, **fit_params ):
    '''
    Generates histograms and fits kinetic model according to intialization dictionary fit_params
    
    returns:
    
    lmfit.model.ModelResult
    
    '''
    
    
    jump_histrogram = get_jump_length_histrogram(cell_spt, **fit_params)
    
    if fit_params['plot_hist']: plot_hist_jumps(jump_histrogram)

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
