import os
from fastspt import readers, tools
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re
import numpy as np


def get_exposure_ms_from_path(path, pattern=r'ch_(\d*?)ms'):
    import re

    regx = re.compile(pattern)
    found = regx.findall(path)
    if found:
        try:
            return float(found[0])
        except ValueError as e:
            print(e)
    else:
        return None
    
assert get_exposure_ms_from_path('prebleach_30ms_strobo_15ms_1_') == 30


fit_params = dict(states=2,
                 iterations=1,
                 CDF=False,
                 CDF1 = True,
                 Frac_Bound = [0, 1],
                 D_Free = [0.02, 0.2],
                 D_Med = [0.005, 0.5],
                 D_Bound = [0.0, 0.005],
                 sigma = 0.025,
                 sigma_bound = [0.005, 0.1],
                 fit_sigma=True,
                 dT=0.06,
                 useZcorr=False,
                 plot_hist=False,
                 plot_result=True)

def parallel_fit(data_paths, fit_params, override=False):
    '''
    Runs process_xml_with_automatic_fps in parrallel
    '''

    from multiprocessing import Pool, cpu_count
    from functools import partial

    fun = partial(process_xml_with_automatic_fps, fit_params=fit_params, override=override)
    
    fits = []

    try:
        n_cpu = cpu_count()
        p = Pool(n_cpu)
        print(f"Running on {n_cpu} cores")
        fits = p.map(fun, data_paths)
        p.close()
        p.join()

    except Exception as e:
        print(f"Problem with Pool {e}. Fall back to sequential fit")
        fits = list(map(process_xml_with_automatic_fps, data_paths[:]))
    
    return fits


def process_xml_with_automatic_fps(path, fit_params=fit_params, override=False):
    print(path)
    
    if os.path.exists(path + ".fit.json") and not override:
        stats = pd.read_json(path + ".fit.json")
        print('Found previous fit, recovering stats')
        return stats

    exp = get_exposure_ms_from_path(path)
    exposure = exp / 1000.
    time_stamp = os.path.getmtime(path)
    print(exposure, ' ms')
    
    try:
        tracks = readers.read_trackmate_xml(path)
    except KeyError:
        print(f'Unable to read {path}')
        return False
    print(len(tracks), ' tracks')
    cell_spt = readers.to_fastSPT(tracks, from_json=False)
    fit_result = tools.auto_fit(cell_spt,
                                fit_params=fit_params)
    fit_result.params.add('num_tracks', len(tracks))
    fit_result.params.add('time_stamp', time_stamp)
    
    
    
    stats = get_stats(fit_result, names=[path])
    stats.to_json(path + ".fit.json")
    print('Saving fit_result')

    return  stats

def get_stats(*reps_fits, names=None):
    fit_stats = pd.DataFrame(columns=list(reps_fits[0].best_values.keys()) + ['chi2', 'num_tracks', 'time_stamp'])
    fit_stats.index.name = 'Dataset'
    for i, fit_result in enumerate(reps_fits):
        if names:
            name = names[i].split("/")[-1]
        else:
            name = f'rep {i+1}'
        fit_stats.loc[f'{name}'] = list(fit_result.best_values.values()) + [
            fit_result.chisqr, 
            fit_result.params['num_tracks'].value,
            fit_result.params['time_stamp'].value
        ]

#     fit_stats.loc['mean'] = fit_stats.mean(axis=0)
#     fit_stats.loc['std'] = fit_stats.std(axis=0)

    #fit_stats.to_json(folder + '\stats.json')

    return fit_stats


def modification_date(filename):
    tif_path = get_tif_path_from(filename)
    t = os.path.getmtime(tif_path)
    return datetime.datetime.fromtimestamp(t)

    

def get_tif_path_from(xml_path, extension='.tif'):
    span = re.search(extension, xml_path)
    end_of_span = span.span()[1]
    tif_path = xml_path[:end_of_span]
    return tif_path

assert get_tif_path_from('/mnt/c/Users/Andrey/data/2019/0822-AV51-OD0.2-DCS1mM-PI-diluted-plated-14h30/tracking_488_prebleach_60ms_no_strobo_1/tracking_488_prebleach_60ms_no_strobo_1_MMStack_Pos0.ome.tif.Tracks.xml') == '/mnt/c/Users/Andrey/data/2019/0822-AV51-OD0.2-DCS1mM-PI-diluted-plated-14h30/tracking_488_prebleach_60ms_no_strobo_1/tracking_488_prebleach_60ms_no_strobo_1_MMStack_Pos0.ome.tif'

def put_results_to_dataframe(times, D_frees, F_bounds, num_tracks):

    df = pd.DataFrame(
        #index=times, 
        columns=['minutes', 'D_free', 'F_bound', 'num_tracks'],
        data=np.array([times, D_frees, F_bounds, num_tracks]).T
    )
    return df

def plot_stats(stats, data_paths, start_time=None, save_json_prefix='', json_name='stats.json'):
    '''
    Plots D_free, F_bound, num_tracks over time
    Parameters:
    -----------
    stats: list of pd.Dataframes
        List of fitting result from process_xml_with_automatic_fps
    data_paths: list of str
        List of original paths of the tables
    start_time: str
        `15:34` for example or None
    Return:
    -------
    Nothing
    '''

    stats_good = list(filter(lambda a: isinstance(a, pd.DataFrame), stats))
    if len(stats_good) < len(stats):
        print(f'{len(stats) - len(stats_good)} items removed')
    stats_fused = pd.concat(stats_good)
    # stats = get_stats(*fits, names=data_paths)
    time_stamp = [modification_date( p) for p in data_paths]
    if start_time:
        h, m = list(map(int, start_time.split(':')))
        t0 = time_stamp[0].replace(hour=h, minute=m)
        time_stamp = [float((t - t0).total_seconds()) / 60. for t in time_stamp]
        x_label = 'minutes'
    else:
        time_stamp = [t.time() for t in time_stamp]
        x_label = 'time'
    print(time_stamp)
    D_frees = stats_fused.D_free.values
    F_bound = stats_fused.F_bound.values
    num_tracks = stats_fused.num_tracks.values
#     time_stamp = stats.time_stamp.values

    if save_json_prefix:
        df = put_results_to_dataframe(time_stamp, D_frees, F_bound, num_tracks)
        path_for_stats = save_json_prefix + json_name
        print(f'Saving stats to {path_for_stats}')
        df.to_json(path_for_stats)
    else:
        print('Warning, stats are not saved')
    

    plt.plot(time_stamp, D_frees, 'ro')
    plt.title('D_free vs time')
    plt.xlabel(x_label)
    plt.grid()
    plt.show()

    plt.plot(time_stamp, F_bound, 'ro')
    plt.title('F_bound vs time')
    plt.xlabel(x_label)
    plt.grid()
    plt.show()

    plt.plot(time_stamp, num_tracks, 'ro')
    plt.title('num_tracks vs time')
    plt.xlabel(x_label)
    plt.grid()
    plt.show()


def moving_average(df, data_column='D_free', time_column='minutes', window=10, step=10):
    time = df[time_column].values
    data = df[data_column].values
    min_time = time.min()
    max_time = time.max()
    time_range = np.arange(min_time, max_time, step)
    def fetch_data(t0): 
        f = np.logical_and(time < t0 + window, time > t0)
        return data[f]
    get_mean_std = lambda t0: [t0, fetch_data(t0).mean(), fetch_data(t0).std()]
    av_data = list(map(get_mean_std, time_range))
    return pd.DataFrame(
        columns=[
            time_column, 
            data_column + '_mean', 
            data_column + '_std'
        ], data=np.array(av_data)
    )