import lmfit
import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt
from fastspt import bayes
from tqdm.auto import tqdm
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class JumpLengthHistogram:
    '''
    Object containing histogram values, bin edges, centers of bins 
    and time lag
    '''
    def __init__(self, hist, bin_edges, lag:int):
        self.hist = hist
        assert int(lag) > 0
        self.lag = int(lag)
        self.bin_edges = bin_edges
        self.vector = bin_edges[:-1] + np.diff(bin_edges[:2])[0] / 2
        assert len(self.vector) == len(self.hist)
    
    def __repr__(self):
        return f"""JumpLengthHistogram: {self.lag} lag, {len(self.vector)} elements"""
    
    def __len__(self):
        return len(self.vector)


def fit_spoton_2_0(
    tracks=None, 
    path=None, 
    n_lags=1, 
    plot=True, 
    dt=0.06,
    D=(0, 0.1), 
    fit_D=(False, True), 
    F=(0.3, 0.7), 
    fit_F=(True, True),
    sigma=(0.02,), 
    fit_sigma=(True,),
    n_bins=50,
    max_um=0.6,
    verbose=False,
    return_fit_result=False,
    return_hists=False,
    **kwargs
) -> dict:

    logger.info(f'fit_spoton_2_0: Fit path: {path}')

    # tracks1 = list(filter(lambda t: len(t) > n_lags + 1, tracks))
    n_tracks = len(tracks)

    logger.info(f'Total {n_tracks} tracks')

    hists = get_jds_histograms(tracks, n_lags, bins=n_bins, max_um=max_um, disable_tqdm=not verbose) 

    
    fit_result = fit_jd_hist(
        hists=hists, 
        dt=dt, 
        D=D, 
        fit_D=fit_D, 
        F=F, 
        fit_F=fit_F,
        sigma=sigma, 
        fit_sigma=fit_sigma,
        verbose=verbose
    )

    values = fit_result.params.valuesdict()
    dt = values['dt']
    sigma = [values[f'sigma{i}'] for i in range(len(sigma))]
    D_all = [values[f'D{i}'] for i in range(len(D))]
    F_all = [values[f'F{i}'] for i in range(len(F))]
    order = np.argsort(D_all)
    D_all = np.array(D_all)[order]
    F_all = np.array(F_all)[order]
    

    if plot:
        
        logger.info(f'fit_spoton_2_0: Plot fit for (D, F): {(D_all, F_all)}, {n_lags} lags')
        _ = [get_error_histogram_vs_model(
            h, 
            dt, 
            sigma, 
            D_all, 
            F_all, 
            plot=True
         ) for h in hists]

    out = {
        'sigma': list(sigma), 
        'D': list(D_all),
        'F': list(F_all),
        'dt': dt,
        'n_tracks': n_tracks, 
        'chi2': fit_result.chisqr, 
        'chi2_norm': fit_result.chisqr / n_bins / n_lags,
        'n_iter': fit_result.nfev,
        'path': path,
        **kwargs}

    if return_fit_result:
        out['fit_result'] = fit_result

    if return_hists:
        out['hists'] = hists
    
    return out


def result_2_table(*results:[dict]):
    new_dict = {}
    for r, res in enumerate(results):
        for k, v in res.items():
            if isinstance(v, (float, int, str)):
                try:
                    new_dict[k][r] = v
                except KeyError:
                    new_dict[k] = {}
                    new_dict[k][r] = v
            elif isinstance(v, (list, np.ndarray)):
                for i, vv in enumerate(v):
                    try:
                        new_dict[f'{k}_{i}'][r] = vv
                    except KeyError:
                        new_dict[f'{k}_{i}'] = {}
                        new_dict[f'{k}_{i}'][r] = vv
            else:
                pass
#                 print(f'skip {k}')
    df = pd.DataFrame.from_dict(new_dict)
    df.index.name = 'replicate'
    return df

def fit_jd_hist(
    hists:list, 
    dt:float, 
    D:list, 
    fit_D:list, 
    F:list,
    fit_F:list,
    sigma:float, 
    fit_sigma:bool,
    verbose=False, 
    
):
    
    '''
    Fits jd probability functions to a jd histograms. 
    
    Parameters:
    hist (list): histogram values
    D (list): init values for MSD
    F (list): fractions for D, sum = 1
    sigma (float): localization precision guess
    funcs (dict): dictionary with functions sigma, gamma, center, amplitude
    
    Returns:
    popt (lmfit.minimizerResult): optimized parameters
    '''

    from lmfit import Parameters, Parameter, fit_report, minimize
            
    def residual(fit_params, data):
        res = cumulative_error_jd_hist(fit_params, data, len(D))
        return res
    
    fit_params = Parameters()
    
    # fit_params.add('sigma', value=sigma, vary=fit_sigma, min=0.)
    fit_params.add('dt', value=dt, vary=False)
    try:
        fit_params.add('max_lag', value=max([h.lag for h in hists]), vary=False)
    except TypeError as e:
        logger.error(f'problem with `hists`: expected `list`, got `{type(hists)}`')
        raise e
    
    for i, (d, f_d, f, f_f) in enumerate(zip(D, fit_D, F, fit_F)):
        fit_params.add(f'D{i}', value=d, vary=f_d, min=0.)
        fit_params.add(f'F{i}', value=f, min=0., max=1., vary=f_f)

    f_expr = '1'
    for i, f in enumerate(F[:-1]):
        f_expr += f' - F{i}'

    fit_params[f'F{i+1}'] = Parameter(name=f'F{i+1}', min=0., max=1., expr=f_expr) 
    

    for i, (s, f_s, min_s, max_s) in enumerate(
        zip(
            sigma, 
            fit_sigma,
            (0, sigma[0]),
            (3*sigma[0], D[-1])
        )
    ):
        fit_params.add(f'sigma{i}', value=s, min=min_s, max=max_s, vary=f_s)
        
    
    # fit_params.pretty_print()
    logger.debug('start minimize')
    
    minimizer_result  = minimize(residual, fit_params, args=(hists, ))#, **solverparams)

    if verbose:
        logger.info(f'completed in {minimizer_result.nfev} steps')
        minimizer_result.params.pretty_print()

    return minimizer_result
       
        
def get_jds_histograms(tracks, max_lag, max_um=0.6, bins=100, disable_tqdm=False):
    '''
    For every lag in 1..max_lag compute density histogram
    
    '''
    def single_hist(i):
        lag = i + 1
        _jds = [
            bayes.get_jd(t.xy, lag, filter_frame_intevals=t.frame) for t in tracks
        ]
        jds = np.concatenate(_jds, axis=0)
        h, edges = np.histogram(jds, bins=bins, range=(0, max_um), density=True)
        return JumpLengthHistogram(h, edges, lag)
        
    hists = list(
        map(
            single_hist, 
            tqdm(
                range(max_lag), 
                desc=f'jds hists for {max_lag} lags', 
                disable=disable_tqdm
            )
        )
    )
    
    return hists

# def convert_value_to_vector(value, length, dtype):
    
#     if isinstance(value, dtype):
#         value = [value] * length
#     elif len(value) == 1 and isinstance(value, dtype):
#         value = value * length
#     else:
#         assert len(value) == length, f'Bad value vector of len {len(value)}, while D len {length}'
#     return value


def get_error_histogram_vs_model(
    hist:JumpLengthHistogram, 
    dt:float, 
    sigma:list, 
    D:list, 
    F:list, 
    p_density=bayes.p_jd, 
    plot=True,
) -> np.ndarray:

    assert len(D) == len(F), f'D and F vector should of the same length. Got {len(D)} and {len(F)}'
    assert isinstance(hist, JumpLengthHistogram)

    vector = hist.vector
    values = hist.hist
    lag = hist.lag
    model = np.zeros_like(vector)
    
    for d, f, s, in zip_longest(D, F, sigma, fillvalue=sigma[0]):
        # print('_D,_F, sigma: ', d, f, s)
        model = model + p_density(dt * lag, s, d)(vector) * f
    
    if plot:
        plt.figure(figsize=(10, 1))
        for i, (_D, _F, s) in enumerate(zip_longest(D, F, sigma, fillvalue=sigma[0])):
            name = 'D'

            plt.plot(
                vector, 
                p_density(dt * lag, s, _D)(vector) * _F, 
                alpha=0.5, 
                label=f'{name}$_{i}$: {_D:.2f}, σ: {s:.3f}, fraction {_F:.0%}'
            )
        plt.plot(vector, model, 'r-', label='sum model')

        plot_hist(
            vector, 
            values, 
            label=f'jd {lag} lag', 
            fill=None,
            alpha=0.8
        )

        plot_hist(
            vector, 
            model - values,
            label=f'residuals', 
            fill='red'
        )
                
        plt.title('sigma ' + ', '.join([f'{s:.3f}' for s in sigma]))
        plt.legend(loc=(1, 0))
        plt.show()
        
    return model - values

def cumulative_error_jd_hist(
    fit_params:lmfit.Parameters, 
    hist_list:list, 
    num_states:int
    ) -> np.ndarray:
    
    p = fit_params.valuesdict()
    # print(p)
    
    sigma = []
    try:
        for i in range(2):
            sigma.append(p[f'sigma{i}'])
    except:
        pass

    cum = [
        get_error_histogram_vs_model(
            h, 
            dt=p["dt"], 
            sigma=sigma,
            D=list(p[f'D{i}'] for i in range(num_states)), 
            F=list(p[f'F{i}'] for i in range(num_states)),
            plot=False
        ) 
        for h in hist_list
    ]
    return np.concatenate(cum, axis=0)
    

def plot_hist(vector, values, label, **kwargs):
    '''
    Simple barplot with autimatic calculation of bin width
    '''
    plt.bar(
        vector, 
        values, 
        width=np.diff(vector)[0],
        **kwargs
    )