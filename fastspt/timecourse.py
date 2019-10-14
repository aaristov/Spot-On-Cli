import os
from fastspt import readers, tools
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re
import numpy as np
from glob import glob


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
    if exp:
        exposure = exp / 1000.
    else: 
        exposure = fit_params['dT']
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

    

def get_tif_path_from(xml_path, extension='.ome.tif'):
    folder = os.path.dirname(xml_path)
    # print(folder)
    flist = glob(folder + os.path.sep + '*' + extension)
    try:
        tif_path = flist[0]
    except IndexError:
        print(f'index error folder {folder}')
        print(f'flist: {flist}')
        return xml_path
    # print(tif_path)
    return tif_path

# assert get_tif_path_from('/mnt/c/Users/Andrey/data/2019/0822-AV51-OD0.2-DCS1mM-PI-diluted-plated-14h30/tracking_488_prebleach_60ms_no_strobo_1/tracking_488_prebleach_60ms_no_strobo_1_MMStack_Pos0.ome.tif.Tracks.xml') == '/mnt/c/Users/Andrey/data/2019/0822-AV51-OD0.2-DCS1mM-PI-diluted-plated-14h30/tracking_488_prebleach_60ms_no_strobo_1/tracking_488_prebleach_60ms_no_strobo_1_MMStack_Pos0.ome.tif', get_tif_path_from('/mnt/c/Users/Andrey/data/2019/0822-AV51-OD0.2-DCS1mM-PI-diluted-plated-14h30/tracking_488_prebleach_60ms_no_strobo_1/tracking_488_prebleach_60ms_no_strobo_1_MMStack_Pos0.ome.tif.Tracks.xml')

def put_results_to_dataframe(**dict_items):

    columns = dict_items.keys()
    data = [dict_items[k] for k in columns]
    try:
        df = pd.DataFrame(
            #index=times, 
            columns=columns,
            data=np.array(data).T
        )
    except ValueError as e:
        print('data :', data)
        raise e
    return df

def get_interval_minutes(*time_stamp, start_time:str="00:00"):
    h, m = list(map(int, start_time.split(':')))
    t0 = time_stamp[0].replace(hour=h, minute=m)
    time_interval_mins = [float((t - t0).total_seconds()) / 60. for t in time_stamp]
    return time_interval_mins

def plot_stats(
    stats, 
    data_paths, 
    start_time=None, 
    save_folder='', 
    save_title='',
    json_name='stats.json',
    csv_name='stats.csv',
    minutes=None
    ):
    '''
    Plots D_free, F_bound, num_tracks over time
    saves csv, json, pdf if save_folder provided
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
        time_stamp = get_interval_minutes(*time_stamp, start_time=start_time)
        x_label = 'minutes'
    else:
        time_stamp = [t.time() for t in time_stamp]
        x_label = 'time'
    print(time_stamp)
    D_frees = stats_fused.D_free.values
    F_bound = stats_fused.F_bound.values
    num_tracks = stats_fused.num_tracks.values
#     time_stamp = stats.time_stamp.values
    out = {
        'minutes': time_stamp, 
        'D_free': D_frees,
        'F_bound': F_bound, 
        'num_tracks': num_tracks
        }

    if minutes:
        groups = find_regexp_in_the_paths(data_paths, minutes)
        if groups:
            print(f'Found minutes: {groups}')
            out['sample_preparation_minutes'] = groups

    if save_folder:
        df = put_results_to_dataframe(**out)
        df = df.sort_values('minutes')
        path_for_stats = save_folder + json_name
        print(f'Saving stats to {path_for_stats}')
        df.to_json(path_for_stats)
        path_for_csv = save_folder + csv_name
        print(f'Saving stats to {path_for_csv}')
        df.to_csv(path_for_csv, index=False)
    else:
        print('Warning, stats are not saved')
    
    fig_params = dict(dpi=72, figsize=(4, 3), facecolor='white')

    plt.figure(**fig_params)
    plt.plot(time_stamp, D_frees, 'r.')
    # plt.title('D_free vs time')
    plt.xlabel(x_label)
    plt.ylabel(r'$D_{free}, \mu m^2 / sec$')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_folder + 'D_free_minutes.pdf')
    plt.show()

    plt.figure(**fig_params)
    plt.plot(time_stamp, F_bound * 100., 'r.')
    # plt.title('F_bound vs time')
    plt.xlabel(x_label)
    plt.ylabel(r'$F_{bound}, \%$')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_folder + 'F_bound_minutes.pdf')
    plt.show()

    plt.figure(**fig_params)
    plt.plot(time_stamp, num_tracks, 'r.')
    plt.title('num_tracks vs time')
    plt.xlabel(x_label)
    plt.grid()
    plt.show()

def find_regexp_in_the_paths(paths, regexp, convert_to=int):
    # print(paths)
    rem = re.compile(regexp)
    # print([rem.findall(p) for p in paths])
    try:
        items = list(map(lambda p: convert_to(rem.findall(p)[0]), paths))
        return items
    except IndexError:
        print(f'ERROR: Nothing found with {regexp} in the paths')
        return [None] * len(paths)

    

def group_paths(list_of_str, by_regexp=r'/.*slide_(\d+?)m', convert_to=int, verbose=True):
    '''
    Groups string items in the list by unique accurances of regexp
    
    Return:
    -------
    (groups, grouped_items): 
        list of occurances, list of grouped items
           
    '''
    
    times = find_regexp_in_the_paths(list_of_str, by_regexp)
    groups = list(np.unique(times))
    if verbose:
        print(f'Found following unique accurances: {groups}')
    ind = [list((times == u).nonzero()[0]) for u in groups]
    grouped_items = [[list_of_str[i] for i in ii] for ii in ind]
    if verbose:
        print(grouped_items)
    return groups, grouped_items

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