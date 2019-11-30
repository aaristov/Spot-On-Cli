import scipy.io
import os
import numpy as np
from fastspt import matimport
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

tp.ignore_logging()


def fuse_lists(*args):
    fused_lists = reduce(lambda x, y: x+y, args)
    return fused_lists


def open_ts_table(path, verbose=0):
    ts_table = pd.read_csv(path)
    if verbose:
        ts_table.head()
    return ts_table


def add_suffix(path, suffix):
    abs_name, ext = os.path.splitext(path)
    return abs_name + suffix + ext


def get_extension(path):
    _, ext = os.path.splitext(path)
    return ext


def save_csv(df: pd.DataFrame, path, suffix=''):
    '''
    Saves pandas DataFrame to csv.
    Provided path complimented with suffix before .csv extension if provided.
    '''
    ext = get_extension(path)
    if ext == '.csv':
        new_path = add_suffix(path, suffix)
        df.to_csv(new_path, index=False)
        print(f'Saved csv to {new_path}')
        return True
    else:
        print('saving failed')
        raise ValueError(f'Wrong extension. Expected .csv, got {ext}')


def get_loc_numbers(ts_table: pd.DataFrame, do_plot=True):
    '''
    Analyses number of particles per frame, returns the list
    with the length of number of frames
    '''
    _, idx = np.unique(ts_table.frame, return_index=True)
    num_locs_per_frame = idx[1:] - idx[:-1]
    if do_plot:
        plt.plot(num_locs_per_frame)
    return num_locs_per_frame


def convert_grouped_tracks_to_df(grouped_tracks):
    fused_tracks = fuse_lists(*grouped_tracks)
    columns = ['x [nm]', 'y [nm]', 'time', 'frame']
    df = pd.DataFrame(columns=columns, data=fused_tracks)
    return df


def open_and_link_ts_table(
    path,
    min_frame=None,
    max_frame=None,
    exposure_ms=60,
    link_distance_um=0.3,
    link_memory=1,
    verbose=0,
    loc_num_plot=True,
    save_csv_with_track_id=True,
    save_linked_localizations=True,
    suffix_for_table_with_track_id='_pytracked',
    suffix_for_only_linked_locs='_pytracked_linked_locs',
    force=False
):

    path = r'{}'.format(path)

    tracked_path = add_suffix(path, suffix_for_table_with_track_id)

    if not os.path.exists(tracked_path) or force:
        ts_table = open_ts_table(path)
        tracks = link_ts_table(
            ts_table,
            min_frame,
            max_frame,
            exposure_ms,
            link_distance_um,
            link_memory,
            verbose,
            loc_num_plot
            )
        if save_csv_with_track_id:
            try:
                _ = save_csv(tracks, tracked_path)
            except ValueError as e:
                print('Unable to save tracks into csv, continue')
                print(e.args)

    else:
        print(f'tracks found in {tracked_path}, use force option to reanalyse')
        tracks = open_ts_table(tracked_path)

    grouped_tracks = matimport.group_tracks(
        tracks, min_len=3, max_len=20, exposure_ms=exposure_ms)

    if save_linked_localizations:
        try:
            df = convert_grouped_tracks_to_df(grouped_tracks)
            _ = save_csv(df, path, suffix='_pytracked_linked_locs')
        except ValueError:
            print('Unable to save tracks into csv, continue')

    return grouped_tracks


def link_ts_table(
    ts_table: pd.DataFrame,
    min_frame=None,
    max_frame=None,
    exposure_ms=60,
    link_distance_um=0.3,
    link_memory=1,
    verbose=0,
    loc_num_plot=True,
):
    '''
    links particles using 'x [nm]', 'y [nm]', 'frame' collumns with trackpy
    returns Dataframe with 'x', 'y', 'frame', 'particle' columns
    '''
    df = pd.DataFrame(
        columns=['x', 'y', 'frame'],
        data=ts_table[['x [nm]', 'y [nm]', 'frame']].values)

    df.x = df.x / 1000  # nm -> um
    df.y = df.y / 1000  # nm -> um
    if min_frame:
        df = df[df.frame >= min_frame]
    if max_frame:
        df = df[df.frame <= min_frame]
    if verbose:
        df.head()
    print(f'Linking with max distance {link_distance_um} um')
    tracks = tp.link_df(df, search_range=link_distance_um, memory=link_memory)
    if verbose:
        print(tracks.head())

    return tracks


def get_low_density_frame(num_locs_per_frame: list, max_locs=200):
    assert len(num_locs_per_frame) > 10

    if max(num_locs_per_frame) > max_locs:
        peak = np.argmax(num_locs_per_frame)
        # print(peak)
        indices_with_fewer_locs = np.where(
            num_locs_per_frame[peak:] > max_locs)[0]
        # print(indices_with_fewer_locs)
        try:
            thr = indices_with_fewer_locs[-1]
        except IndexError:
            thr = 0
        return thr + peak
    else:
        return 0


def load_matlab_dataset_from_path(path):
    """Returns a dataset object from a Matlab file"""
    mat = scipy.io.loadmat(path)
    return np.asarray(mat['trackedPar'][0])
