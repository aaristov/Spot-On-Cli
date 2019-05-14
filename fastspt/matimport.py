# TODO: debug MatlabTable methods

import os
import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm
from fastspt import readers, tracklen, tools
import pandas as pd
try:
    from IPython.core.display import display
except ImportError:
    display = print

import logging
logger = logging.getLogger(__name__)
from fastspt.__main__ import pprint

def read_gizem_mat(path):
    from mat4py import loadmat
    data = loadmat(path)
    rep_fov_xyft = data['tracks']['t']
    return rep_fov_xyft

def concat_all(rep_fov_xyft, min_len=3, exposure_ms=None, pixel_size_um=None):
    '''
    Concatenates all the data into one table
    returns [xyft_table]
    '''

    n_rep = len(rep_fov_xyft)
    each_len_rep = [len(t) for t in rep_fov_xyft]
    print(f'discovered {n_rep} replicates containing {each_len_rep} acquisitions')
    print(f'Assembling tracks with minimal length {min_len}, using exposure {exposure_ms} ms and px size {pixel_size_um} µm')
    tracks = []
    for i in tqdm(range(n_rep)):
        for fov in tqdm(rep_fov_xyft[i], desc=f'rep {i+1}'):
            grouped = group_tracks(fov, min_len, exposure_ms, pixel_size_um)
            tracks = sum([tracks, grouped], [])
            
    print(f'Total {len(tracks)} tracks')
    return [tracks]

def concat_reps(rep_fov_xyft, min_len=3, exposure_ms=None, pixel_size_um=None):
    '''
    Concatenates data by replicates
    returns [xyft_table_rep1, ..., ..._repN]
    '''
    
    n_rep = len(rep_fov_xyft)
    each_len_rep = [len(t) for t in rep_fov_xyft]
    print(f'discovered {n_rep} replicates containing {each_len_rep} acquisitions')
    print(f'Assembling tracks with minimal length {min_len}, using exposure {exposure_ms} ms and px size {pixel_size_um} µm')
    reps = []
    for i in tqdm(range(n_rep)):
        tracks = []
        for fov in tqdm(rep_fov_xyft[i], desc=f'rep {i+1}'):
            grouped = group_tracks(fov, min_len, exposure_ms, pixel_size_um)
            tracks = sum([tracks, grouped], [])
        reps.append(tracks) 
        print(f'Replicate {i+1}: Total {len(tracks)} tracks')

    return reps

def group_tracks(xyft, min_len=3, exposure_ms=None, pixel_size_um=None):
    xyft = np.array(xyft)
    assert xyft.ndim == 2 and xyft.shape[1] == 4
    _, ids = np.unique(xyft[:,3], return_index=True)
    tracks = []
    xyft[:,3] = xyft[:,2]
    if exposure_ms:
        xyft[:,2] = xyft[:,2] * exposure_ms * 1.e-3
    
    if pixel_size_um:
        xyft[:,:2] = xyft[:,:2] * pixel_size_um
        
    for i, ii in tqdm(zip(ids[:-1], ids[1:]), disable=True):
        track = xyft[i:ii]
        try:
            if len(track) >= min_len:
                tracks.append(track)
        except Exception as e:
            print(min_len)
            raise e

    print(f'{len(tracks)}  tracks ')
    return tracks

# def analyse_mat_file(data_path,
#                      fit_params
#                      ):
    
#     print(f'Analysing {data_path}')

#     pprint(fit_params)
#     all_exp = read_gizem_mat(data_path)
    
#     if all_exp:
#         reps = concat_reps(all_exp, min_len=min_len, exposure_ms=exposure_ms, pixel_size_um=pixel_size_um)
#     for rep in reps:
#         tracklen.get_track_lengths_dist(rep, plot=plot_track_len)
        
#     def my_fit(rep):
    
#         cell_spt = readers.to_fastSPT(rep, from_json=False)
#         fit_result = tools.auto_fit(cell_spt,
#                                     **fit_params)
#         return fit_result

#     reps_fits = list(map(my_fit, reps))
#     stats = get_stats(reps_fits, save_path=data_path)
#     return stats



# def analyse_mat_files(*path_list, exposure_ms=60., pixel_size_um=0.075, **kwargs):
#     print(path_list)
#     stats = {}
#     for data_path in path_list[0]:
#         stats[data_path] = analyse_mat_file(data_path)
#     return stats

def get_stats(reps_fits, save_path=None, print_stats=True, save_fmt='json', suffix='.stats'):
    #get stats
    fit_stats = pd.DataFrame(columns=list(reps_fits[0].best_values.keys()) + ['chi2'])
    
    for i, fit_result in enumerate(reps_fits):
        fit_stats.loc[f'rep {i+1}'] = list(fit_result.best_values.values()) + [fit_result.chisqr]

    fit_stats.loc['mean'] = fit_stats.mean(axis=0)
    fit_stats.loc['std'] = fit_stats.std(axis=0)
    
    if save_path:
        path = f'{save_path}{suffix}.{save_fmt}'
        getattr(fit_stats, 'to_' + save_fmt)(path)
        logger.info(f'Saving stats table to {path}')
    if print_stats:
        display(fit_stats)
    return fit_stats
    
def _read_gizem_mat(path):
    '''
    Opens matlab file and retirns MatlabTable object
    '''

    from mat4py import loadmat
    data = loadmat(path)
    rep_fov_xyft = data['tracks']['t']
    return MatlabTable(rep_fov_xyft)

class MatlabTable:
    '''
    Summarizes the routines needed to process tracking tables.
    '''
    
    def __init__(self, raw_tracks:list):
        self.rep_fov_xyft = raw_tracks
        self.n_rep = len(self.rep_fov_xyft)
        each_len_rep = [len(t) for t in self.rep_fov_xyft]
        logger.info(f'discovered {self.n_rep} replicates containing {each_len_rep} acquisitions')
        
        
    def concat_all(self, min_len=3, exposure_ms=None, pixel_size_um=None):
        logger.info(f'Assembling tracks with minimal length {min_len}, using exposure {exposure_ms} ms and px size {pixel_size_um} µm')
        tracks = []
        for i in tqdm(range(self.n_rep)):
            fovs = self.rep_fov_xyft[i]
            rep = self.group_fovs(fovs, min_len, exposure_ms, pixel_size_um, desc=f'rep {i+1}')
            tracks = self.fuse_lists([tracks, rep])
        logger.info(f'Total {len(tracks)} tracks')
        return tracks

    def concat_reps(self, min_len=3, exposure_ms=None, pixel_size_um=None):
        logger.info(f'Assembling tracks with minimal length {min_len}, using exposure {exposure_ms} ms and px size {pixel_size_um} µm')
        
        reps = []
        for i in tqdm(range(self.n_rep)):
            fovs = self.rep_fov_xyft[i]
            rep = self.group_fovs(fovs, min_len, exposure_ms, pixel_size_um, desc=f'rep {i+1}')
            reps.append(rep) 
            logger.info(f'Replicate {i+1}: Total {len(rep)} tracks')

        return reps
    
    def group_fovs(self, fovs, min_len=3, exposure_ms=None, pixel_size_um=None, desc=''):
        tracks = []
        for fov in tqdm(fovs, desc=desc):
            grouped = group_tracks(fov, min_len=min_len, exposure_ms=exposure_ms, pixel_size_um=pixel_size_um)
            tracks = self.fuse_lists([tracks, grouped])
        return tracks
    
    def fuse_lists(self, lists):
        return sum(lists, [])



def table_import(path):
    '''
    File importer. Selects file hadler from the dictionary, returns table.
    '''
    supported_extensions = {'.mat': {'description': 'matlab table from Gizem. rep.fov.[loc_xyft]',
                                        'function': read_gizem_mat},
                            '.xml': {'description': 'xml file from Trackmate. ',
                                    'function': readers.read_trackmate_xml},
                           }
    _, file_extension = os.path.splitext(path)
    try:
        file_handler = supported_extensions[file_extension]['function']
        description = supported_extensions[file_extension]['description']
        logger.info(f"Opening {file_extension} which is expected to be {description}")
    except KeyError:
        logger.error(f'Unsupported extension {file_extension}. Only support {list(supported_extensions.keys())}')
        return False
    table = file_handler(path)
    
    return table