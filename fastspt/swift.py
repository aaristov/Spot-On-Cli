import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re
from fastspt.plot import plt
from fastspt import tracklen
from scipy.optimize import minimize
from functools import reduce

class StroboscopicDataset:
    
    def __init__(self, fps, paths=None, decay_rate=None):
        self.fps = fps
        self.paths = paths
        if decay_rate:
            self.decay_rate = decay_rate
        else:
            self.compute_decay_rate()
    
    def set_decay_rate(self, d):
        self.decay_rate = d
    
    def compute_decay_rate(self):
        self.decay_rate = get_decay_rate_of_bound_molecules(self.paths)
        
    def __repr__(self):
        return f'''\nStroboscopicDataset Object: \n\tFPS: {self.fps}\
             \r\tPATHS: {len(self.paths)} items,\
             \r\tDecay rate : {self.decay_rate:.2f}'''

def get_bleaching_unbinding_rate_from_datasets(*datasets:StroboscopicDataset, plot=True):
    '''
    Computes bleachung rate and unbinding rate for 2 or more datasets.
    
    Return
    ------
    opt_bleach_rate: float, per frame, opt_unbind_rate: float, per second
    '''
    
    funcs = [get_bleach_rate_vs_unbind_rate_fun(d.fps, d.decay_rate) for d in datasets]
    res = find_overlap(*funcs)
    opt_unbind_rate = res.x
    mean_bleach_rate = np.mean(list(map(lambda f: f(opt_unbind_rate), funcs)))
    print (f'bleach rate: {mean_bleach_rate:0.2f} per exposure , unbind rate: {opt_unbind_rate[0]:0.2f} per second')
    if plot:
        plt.figure(facecolor='w')
        r = [opt_unbind_rate - 0.1, opt_unbind_rate + 0.1]
        [plt.plot(r,f(r), label=str(d.fps) + ' fps') for f, d in zip(funcs, datasets)]

        plt.plot(opt_unbind_rate, mean_bleach_rate, 'ro', label='optimal solution')
        
        # pylint: disable=anomalous-backslash-in-string
        plt.xlabel('$\lambda_{unbind} (sec^{-1})$')
        plt.ylabel('$\lambda_{bleach} (frame^{-1})$')
        plt.grid()
        plt.legend()
    return mean_bleach_rate, opt_unbind_rate[0]


def get_decay_rate_of_bound_molecules(paths, bins=30, max_range=100):
    
    fun = get_lengths_of_bound_tracks_from_path 

    bound_lengths_over_replicate = list(map(fun, tqdm(paths)))       
#     plt.show()
    fused_bound_lengths_over_replicate = reduce(lambda a, b: a + b, bound_lengths_over_replicate)
    values, bins, _ = plt.hist(fused_bound_lengths_over_replicate, bins=bins, range=(min(fused_bound_lengths_over_replicate), max_range))
    plt.xlabel('bound tracks length, frame')
    vector, _ = bins2range(bins)
    fit_result, _, decay_rate = fit_exp(values, vector, p0=(1000, 0.1))
    plt.plot(vector, fit_result)
    plt.title(f'decay rate: {decay_rate:.2f} per frame')
    plt.show()
    return decay_rate

def get_bleach_rate_vs_unbind_rate_fun(fps, decay_rate):
    '''
    FPS * λ_apparent =  FPS *λ_bleach + λ_unbind (per second)
    λ_bleach( λ_unbind) =   λ_apparent  -  λ_unbind  / FPS 
    '''
    return lambda unbind_rate: - np.array(unbind_rate) / fps + decay_rate

def find_overlap(*linear_funcs):
    assert len(linear_funcs) > 1
    sds = []
    f0 = linear_funcs[0]
    for f in linear_funcs[1:]:
        sds.append(lambda x: (f(x) - f0(x)) ** 2)
    residual = lambda x: sum([sd(x) for sd in sds])
    
    popt = minimize(residual, (0.1,), callback=None)
    return popt

def open_and_group_tracks_swift(
    path, 
    by='seg.id', 
    exposure_ms=60,
    min_len=3, 
    max_len=np.inf
    ):
    """
    Reads csv from path
    Groups tracks - see swift.group_tracks_swift
    Returns list with tracks in xytf format
    """
    print('Processing ', path)
    df = pd.read_csv(path)
    tracks = group_tracks_swift(df, by, exposure_ms, min_len, max_len)
    return tracks

def group_tracks_swift(df:pd.DataFrame, by='seg.id', exposure_ms = 60, min_len=3, max_len=20):
    tracks = []
    try:
        xyif = df[['x [nm]', 'y [nm]', by, 'frame']].sort_values(by)
    except KeyError:
        xyif = df[['x', 'y', by, 'frame']].sort_values(by)
    
    print(len(xyif), 'localizations')
    seg_ids = xyif[by]
    _, ids = np.unique(seg_ids, return_index=True)
    print(len(ids), "unique ", by)
    time = xyif.frame.values
    if exposure_ms:
        time = time * exposure_ms * 1.e-3
    
    xytf = xyif.values
    xytf[:, 2] = time
    xytf[:, :2] = xytf[:, :2] / 1000. # from nm to um
            
    for i, ii in tqdm(zip(ids[:-1], ids[1:]), disable=True):
        track = xytf[i:ii]
        try:
            if len(track) >= min_len and len(track) <= max_len:
                track_sorted_by_frame = track[np.argsort(track[:,3])]
                tracks.append(track_sorted_by_frame)
        except Exception as e:
            print(min_len)
            raise e

    print(f'{len(tracks)}  tracks grouped with exp time {exposure_ms} ms and lengths between {min_len} and {max_len}')
    return tracks

def make_spoton_dataset_from_swift(data_path):
    tracks = pd.read_csv(data_path)
    rep = group_tracks_swift(tracks, by='seg.id', min_len=5, max_len=30)
    return rep

def show_pops(path):
    print(path)
    tracks = pd.read_csv(path)
    pops = select_populations(tracks)
    n_segs = {}
    for k in pops.keys():
        n = plot_mjd_hist(pops[k], label=k)
        n_segs[k] = n
    plt.legend()
    plt.show()
    return n_segs

def plot_mjd_hist(tracks, use_column='seg.mjd', weight_by_column='seg.mjd_n', bins=30, range=(0,250), label=''):
    h, _, _ = plt.hist(
        tracks[use_column], 
        weights=tracks[weight_by_column], 
        bins=bins, 
        range=range, 
        label=label,
        alpha=0.8,
        lw=1
        )
    plt.xlabel(f'{use_column} weighted by {weight_by_column}')
    plt.ylabel(f'counts')
    sum_h = sum(h)
    print(f'Sum for {label} = {sum_h}')
    return sum_h

def count_unique_segments(sub_tracks):
    return len(np.unique(sub_tracks["seg.id"]))

def select_populations(tracks, from_column='seg.dynamics', keywords=['static', 'free'], equalize_min_len_by='seg.loc_count'):
    '''
    Selects localizations with keywords dynamics and returns a dictionary
    '''
    out = {}
    min_len = 0
    for k in keywords:
        sub_tracks = tracks[tracks[from_column] == k ]
        n_tracks = len(sub_tracks)
        print(f'{k} : {n_tracks} localizations, {count_unique_segments(sub_tracks)} unique tracks')
        min_len = max(min_len, min(sub_tracks[equalize_min_len_by]))
        out[k] = sub_tracks
    
    print(f'min_len: {min_len}')
    for k in keywords:
        out[k] = out[k][out[k][equalize_min_len_by] >= min_len]
       
    return out

def extract_bound_molecules_which_photobleach(tracks_from_swift:pd.DataFrame, limit_seg_count=1):
    '''
    Idea is to select tracks exclusively consisted of bound molecules.
    track.seg_count == 1
    dynamics == 'static'
    '''
    selected_populations = select_populations(tracks_from_swift, keywords=['static', 'free'])
#     print(selected_populations.keys())
    bound_segments, _ = selected_populations.values()
    if limit_seg_count:
        bound_segments = bound_segments[bound_segments['track.seg_count'] == limit_seg_count]
        print(f'{count_unique_segments(bound_segments)} tracks with single segment')
    return bound_segments
    
def get_lengths_of_bound_tracks(bound_molecules_with_single_segment, min_len=20):
    bound_tracks = group_tracks_swift(bound_molecules_with_single_segment, max_len=np.inf, min_len=min_len)
    print('after grouping total ', len(bound_tracks), ' tracks')
    lengths = list(map(len, bound_tracks))
    return lengths

def get_lengths_of_bound_tracks_from_path(data_path, seg_count=None, min_len=5, plot=False):
    print(data_path)
    tracks = pd.read_csv(data_path)
    bleaching_bound_molecules = extract_bound_molecules_which_photobleach(tracks, limit_seg_count=seg_count)
    lengths = get_lengths_of_bound_tracks(bleaching_bound_molecules, min_len=min_len)
    if plot:
        plt.hist(lengths, density=True)
    return lengths

def compute_switching_rate(tracks_swift:pd.DataFrame, frame_rate=None):
    '''
    Computes koff, kon rate from swift table with dynamics column.
    
    p(diff -> bound | diffusive) = N(diff -> bound) / N(diffusive)
    
    Return:
    -------
    a dictionary with all the shit computed
    '''

    out = {}
    bound_locs, diff_locs = select_populations(tracks_swift, keywords=['static', 'free']).values()
    bound_tracks = group_tracks_swift(bound_locs, max_len=np.inf)
    diff_tracks = group_tracks_swift(diff_locs, max_len=np.inf)
    n_bound_locs = len(bound_locs) - len(bound_tracks)
    n_diff_locs = len(diff_locs) - len(diff_tracks)
    
    n_bound_to_diff = 0
    n_diff_to_bound = 0
    # n_no_switch = 0
    
    switching_locs = tracks_swift[tracks_swift['track.seg_count'] > 1]
    
    for track_id in np.unique(switching_locs['track.id']):
#         print('track.id ', track_id)
        single_track = switching_locs[switching_locs['track.id'] == track_id]
        states = [single_track[single_track['seg.id'] == seg_id]['seg.dynamics'].values[0] for seg_id in np.unique(single_track['seg.id'])]
        states_short = '-'.join(states)
        
        n_bound_to_diff += len(re.findall('static-free', states_short)) 
        n_diff_to_bound += len(re.findall('free-static', states_short))

        u_rate = n_bound_to_diff/n_bound_locs
        b_rate = n_diff_to_bound/n_diff_locs
    print(f'b -> u : {n_bound_to_diff},  \
        per {n_bound_locs} bound locs, i.e. \
            unbinding rate {u_rate:.2E} per frame')
    out['b -> u'] = {
        'n_events': n_bound_to_diff,
        'n_locs': n_bound_locs,
        'rate_per_frame': u_rate
    }
    if frame_rate:
        u_rate_sec = u_rate * frame_rate
        print(f'{u_rate_sec:.1E} per second')
        out['b -> u']['rate_per_second'] = u_rate_sec

    print(f'u -> b : {n_diff_to_bound},  per {n_diff_locs} diffusive locs, i.e. binding rate {b_rate:.2E}.')
    out['u -> b'] = {
        'n_events': n_diff_to_bound,
        'n_locs': n_diff_locs,
        'rate_per_frame': b_rate
    }
    if frame_rate:
        b_rate_sec = b_rate * frame_rate
        print(f'{b_rate * frame_rate:.1E} per second')
        out['u -> b']['rate_per_second'] = b_rate_sec

    bound_fraction = compute_expected_bound_fraction(u_rate, b_rate)
    print(f'bound fraction = {bound_fraction:.1%}')
    out['bound fraction'] = bound_fraction
        
    return out

def process_switching_rate(data_path, frame_rate=None):
    """
    example:
        rates = list(map(process_switching_rate_from_swift, data_paths))
    """
    print(data_path)
    tracks = pd.read_csv(data_path)
    return compute_switching_rate(tracks, frame_rate)


def compute_expected_bound_fraction(koff, kon):
    return 1. / (1. + koff / kon)

def make_table_with_rates(rates):
    '''
    Creates oandas Dataframe with rates and bound fractions.
    Parameters:
        rates for n experiments are (n, 2) numpy array or list with first column Koff, second - Kon rate.
    Returns: 
        rates_df: formatted pandas DataFrame table with third column containing bound fraction.
        
    Example:
        rates = list(map(process_switching_rate_from_swift, data_paths))
        rates_df = make_table_with_rates(rates)
        rates_df
            		koff		kon			bound fraction
            dataset			
            0		0.009074	0.001253	0.121338
            1		0.006121	0.000777	0.112576
            2		0.005812	0.000404	0.065015
            3		0.005241	0.000340	0.061006
            4		0.000000	0.000794	1.000000
    '''
    rates_df = pd.DataFrame(columns=['koff', 'kon', 'bound fraction'])
    rates_df.index.name = 'dataset'
    for i, (koff, kon)  in enumerate(rates):
        rates_df.loc[f'{i}'] = [koff, kon, compute_expected_bound_fraction(koff, kon)]

    return rates_df



def exponent(x, a, c):
        return a*np.exp(-x * c)


def fit_exp(values, vector, p0=None):
    '''
    returns:
    fit_result, a, c: c - decay rate
    '''
    fit_result, popt = tracklen.fit_exponent(values, vector, fun=exponent, p0=p0)
    a, c = popt
    print(f'Fit result: {a:.2f} * e^(-x * {c:.2f})')
    return fit_result, a, c
    

def bins2range(bins):
    '''
    Converts bin edges to bin centers.
    !!! Only works for even spacing !!!

    Arguments:
    bins (1D array): bin edges from histogram function

    Returns:
    bin_range (1D array): centers of bins
    bin_step (float): size of the bin

    '''
    bin_step = bins[1] - bins[0]
    bin_range = bins[:-1] + bin_step / 2
    return bin_range , bin_step
