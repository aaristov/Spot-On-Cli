import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm

def read_gizem_mat(path):
    from mat4py import loadmat
    data = loadmat(path)
    rep_fov_xyft = data['tracks']['t']
    return rep_fov_xyft

def concat_all(rep_fov_xyft, min_len=3, exposure_ms=None, pixel_size_um=None):
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
    return tracks

def concat_reps(rep_fov_xyft, min_len=3, exposure_ms=None, pixel_size_um=None):
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
        if len(track) > 3:
            tracks.append(track)
    print(f'{len(tracks)}  tracks ')
    return tracks