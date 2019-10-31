import numpy as np
import fastspt.simulate as sim
from fastspt import swift
from scipy.ndimage import gaussian_filter1d as gf1
from functools import reduce
from tqdm.auto import tqdm
import os
import json
import pandas as pd


d_ = lambda time, sigma, D: D * time + sigma ** 2
p_jd = lambda time, sigma, D: lambda r: r / (2 * d_( time, sigma, D)) * np.exp(- r ** 2 / ( 4 * d_( time, sigma, D)))
deriv = lambda time, sigma, D: lambda r: 1 / (2 *  d_( time, sigma, D)) * (1 -  r ** 2 / (2 * d_( time, sigma, D))) * np.exp(- r **2 / (4 * d_( time, sigma, D)))   
r_max = lambda time, sigma, D: np.sqrt(2 * d_(time, sigma, D))
p_jd_max =lambda time, sigma, D:  p_jd(time, sigma, D)(r_max(time, sigma, D))
p_jd_norm =  lambda time, sigma, D: lambda r: p_jd(time, sigma, D)(r) / p_jd_max( time, sigma, D)

def cdf_bound(sigma, r):
    jd_max = r_max(1, sigma, 0)
    if r <= jd_max:
        return 1
    else:
        return p_jd_norm(1, sigma, 0)(r)

def cdf_unbound(sigma, r):
    return 1 - cdf_bound(sigma, r)

def get_jd(xy:np.array, lag=1, extrapolate=False, filter_frame_intevals=None):

# TODO: send frame jumps to higher lags 

    """    
    Computes jumping distances out of xy coordinates array.
    Parameters:
    -----------
    xy = [[x1, y1]
        [x2, y2]
            ---  ] 
    
    lag: int
        jump interval in frames, default 1

    extrapolate: bool
        If True, returns the same size by filling edges with the same values. Default False.

    filter_frame_intevals: list
        if not None, will use only frame intervals equal to lag, default None

    """ 

    if len(xy) > lag:
        
        dxy = xy[lag:] - xy[:-lag] 

        if filter_frame_intevals is not None:
            frames = np.ravel(filter_frame_intevals)
            d_frames = frames[lag:] - frames[:-lag]
            dxy = dxy[d_frames == lag]

        jd = np.sqrt((dxy ** 2).sum(axis=1))

        if extrapolate:
            while len(jd) < len(xy):
                jd = np.concatenate(([jd[0]], jd[:]))
#                 even = lag % 2 == 0
                if len(jd) < len(xy):
                    jd = np.concatenate((jd[:], [jd[-1]]))

        return jd
    else:
        return []

def classify_bound_segments(track:sim.Track, sigma:float, max_lag:int=4, col_name='prediction', extrapolate_edges=True, verbose=False, return_p_unbinds=False):
    jds = [get_jd(track.xy, lag=l, extrapolate=1) for l in range(1,max_lag+1)]

    jds = list(filter(len, jds))

    p_unbinds = list(map(lambda i: list(map(lambda x: cdf_unbound(sigma, x), jds[i])), range(len(jds))))
#     print(p_unbinds)
    if verbose:
        print('jds: ', jds)
        print(p_unbinds)
    try:
        hl = max_lag // 2
        bound_vector = gf1(np.median(p_unbinds, axis=0), hl) > 0.5

        if verbose:
            print('half lag: ', hl)
            print('bound_vector: ', bound_vector)

        if extrapolate_edges:
            bound_vector[:hl] = bound_vector[hl]
            bound_vector[-hl:] = bound_vector[-hl - 1]
            
            if verbose:
                print('after extrapolation: bound_vector: ', bound_vector)
        
        
    except TypeError as e:
        print(track)
        print(jds)
        bound_vector = [None] * len(track) 
        raise e
    
#     new_track = np.insert(track, track.shape[1], bound_vector, axis=1)
    new_track = track.add_column(col_name, bound_vector, 'free: 1, bound: 0')

    if return_p_unbinds:
        return new_track, p_unbinds
    else:
        return new_track


sum_list = lambda l: reduce(lambda a, b: a + b, l)



def get_switching_rates(xytfu:list, fps:float, lag:int=1, column='free'):
    '''
    Parameters:
    -----------
    xytfu: list of simulate.Track objects
    fps: float
        Framerate
    lag: int
        if lag is more than 1, halpf of this length will be cut off the ends of the track to avoid artefacts
    column: str
        where to look for the label
    Return:
    -------
    stats: dict
        {'F_bound': n_bound_spots / n_total_spots, 'u_rate_frame': u_rate_frame, 'b_rate_frame': b_rate_frame }
    '''
    assert isinstance(lag, int)

    if lag>1:
        n_tracks = len(xytfu)
        s = lag // 2
        e = lag - s
        
        xytfu = filter(lambda t: len(t) > lag + 1, xytfu)
        xytfu = list(map(lambda t: t.crop_frames(s, -e), xytfu))
        print(f'Due to lag={lag}, {len(xytfu)} tracks left out of {n_tracks} with len > {lag + 1}')

    n_bound_spots = sum_list(map(lambda a: sum(a.col(column)[:] == 0), xytfu))
    n_bound_spots_for_rates = sum_list(map(lambda a: sum(a.col(column)[:-1] == 0), xytfu))
    n_unbound_spots_for_rates = sum_list(map(lambda a: sum(a.col(column)[:-1] == 1), xytfu))
    # print(n_bound_spots, n_bound_spots_for_rates)
    n_total_spots = sum_list(map(lambda a: len(a), xytfu))

    
#     n_total_segments = n_total_spots - len(xytfu)
#     n_bound_segments = n_bound_spots - len(bound)

    print( f'bound fraction based on number of spots: {n_bound_spots} / {n_total_spots} = {n_bound_spots / n_total_spots:.1%}')
#     print( f' bound fraction based on number of segments: {n_bound_segments} / {n_total_segments} = {n_bound_segments / n_total_segments:.1%}')
    
    get_n_switch_unbind = lambda xytfu: sum_list(map(lambda a: sum(a.col(column)[1:] - a.col(column)[:-1] == 1), xytfu))
    get_n_switch_bind = lambda xytfu: sum_list(map(lambda a: sum(a.col(column)[1:] - a.col(column)[:-1] == -1), xytfu))

    n_switch_unbind = get_n_switch_unbind(xytfu)
    n_switch_bind = get_n_switch_bind(xytfu)

    print(f'{n_switch_bind} binding events, {n_switch_unbind} unbinding events')
    u_rate_frame = n_switch_unbind / n_bound_spots_for_rates
    b_rate_frame = n_switch_bind / n_unbound_spots_for_rates
    print(f'Unbinding switching rates: {u_rate_frame:.1%} per frame, {u_rate_frame * fps:.1%} per second {fps} fps')
    print(f'Binding switching rates: {b_rate_frame:.1%} per frame, {b_rate_frame * fps:.1%} per second {fps} fps')
    print(f'Bound fraction based on switching rates: {b_rate_frame / (b_rate_frame + u_rate_frame):.1%}')
    
    return {'F_bound': n_bound_spots / n_total_spots, 'u_rate_frame': u_rate_frame, 'b_rate_frame': b_rate_frame, 'F_bound_from_rates': b_rate_frame / (u_rate_frame + b_rate_frame) }

def classify_csv_tracks(
    tracks:sim.Track, 
    max_lag:int=4, 
    col_name='uncertainty_xy [nm]', 
    use_map=map
    ):
    '''
    Runs bayes.classify_bound_segments on a list of simulate.Track tracks.
    '''
    return list(use_map(lambda t: \
         classify_bound_segments(
            t, 
            sigma=t.col(col_name).mean(), 
            max_lag=max_lag,
            extrapolate_edges=False,
            verbose=False
        ), 
        tqdm(tracks)))
  

def get_rates_csv(csv_path='', fps=15, lag=4, force=False, **kwargs):
    '''
    Reads csv with tracks from swift using pandas.
    Groups tracks using `track.id`.
    Performs Bayes classification on tracks (bound/unbound) and computed switching rates.
    Return disctionary and saves rates.json to the disk.
    
    Parameters:
    -----------
    csv_path: str
        Csv file with tracks from Swift. 
        Must contain x [nm], y [nm], frame, track.id, seg.id, uncertainty_xy [nm]
    
    fps: int
        frames per second.
    
    lag: int
        how many jumps to consider in the classification
        
    force: bool
        If force = False, the function tries to find .rates.json file and recover it. 
        If force = True, you want to reanalyse the data.
        
    **kwargs: dict
        Additinal information to include to output.
        Doesn\'t change the analysis.
        
    Return:
    -------
    {
        'F_bound': n_bound_spots / n_total_spots, 
        'u_rate_frame': u_rate_frame, 
        'b_rate_frame': b_rate_frame, 
        'F_bound_from_rates': b_rate_frame / (u_rate_frame + b_rate_frame),
        **kwargs
    }
    '''
    
    if not os.path.exists(csv_path):
        raise ValueError(f'wrong csv_path {csv_path}')
    
    print('Analysing ', csv_path)
    
    json_path = csv_path.replace('.csv', '.rates.json')
    
    if os.path.exists(json_path) and not force:
        with open(json_path) as fp:
            out_dict = json.load(fp)
        print(f'Found `{json_path}`, recovering data. Use force=True to reanalyse.')
        return out_dict
            
    df = pd.read_csv(csv_path)
    
    tracks_with_sigma = swift.group_tracks_swift(
            df, 
            additional_columns=['seg.id', 'uncertainty_xy [nm]'],
            additional_scale=[1, 0.001],
            additional_units=[int, 'um'],
            group_by='track.id',
            min_len=5,
            max_len=50
            )
    c_csv_tracks = classify_csv_tracks(tracks_with_sigma[:], lag)
    stats = get_switching_rates(c_csv_tracks, fps, column='prediction')
    stats['path'] = csv_path
    stats['lag'] = lag
    
    out_dict = {**stats, **kwargs}
    
    with open(json_path, 'w') as fp:
        json.dump(out_dict, fp)
        print(f'saved data to {json_path}')
            
    return out_dict