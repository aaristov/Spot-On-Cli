import os
from fastspt import readers, tools
import matplotlib.pyplot as plt

def get_exposure_ms_from_path(path, pattern='ch_(\d*?)ms'):
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


def process_xml_with_automatic_fps(path, fit_params=fit_params):
    
    exp = get_exposure_ms_from_path(path)
    exposure = exp / 1000.
    time_stamp = os.path.getmtime(path)
    print(exposure, ' ms')
    
    tracks = readers.read_trackmate_xml(path)
    print(len(tracks), ' tracks')
    cell_spt = readers.to_fastSPT(tracks, from_json=False)
    fit_result = tools.auto_fit(cell_spt,
                                fit_params=fit_params)
    fit_result.params.add('num_tracks', len(tracks))
    fit_result.params.add('time_stamp', time_stamp)
    
    with open(path + '.fit_result.txt', 'w') as fh:
        fh.write(fit_result.fit_report())
    print('Saving fit_result')
    return fit_result

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

import datetime
def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)

def plot_stats(fits):
    stats = get_stats(*fits, names=data_paths)
    time_stamp = [modification_date( p) for p in data_paths]
    print(time_stamp)
    D_frees = stats.D_free.values
    F_bound = stats.F_bound.values
    num_tracks = stats.num_tracks.values
#     time_stamp = stats.time_stamp.values

    plt.plot(time_stamp, D_frees, 'ro')
    plt.title('D_free vs time')
    plt.grid()
    plt.show()

    plt.plot(time_stamp, F_bound, 'ro')
    plt.title('F_bound vs time')
    plt.grid()
    plt.show()

    plt.plot(time_stamp, num_tracks, 'ro')
    plt.title('num_tracks vs time')
    plt.grid()
    plt.show()