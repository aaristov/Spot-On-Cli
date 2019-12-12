import lmfit
import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt
from fastspt import bayes, core
from tqdm.auto import tqdm
import pandas as pd
import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)


class JumpLengthHistogram:
    """
    Object containing histogram values, bin edges, centers of bins
    and time lag
    """

    def __init__(self, hist, bin_edges, lag: int):
        self.hist = hist
        assert int(lag) > 0
        self.lag = int(lag)
        self.bin_edges = bin_edges
        self.width = np.diff(bin_edges)[0]
        self.vector = bin_edges[:-1] + self.width / 2
        assert len(self.vector) == len(self.hist)

    def __repr__(self):
        return f"""JumpLengthHistogram: {self.lag} lag, {len(self.vector)} elements"""

    def __len__(self):
        return len(self.vector)


def fit_spoton_2_0(
    tracks: [core.track.Track] = None,
    path: str = None,
    n_lags: int = 1,
    plot: bool = True,
    dt: float = 0.06,
    D: (float, float,) = (0, 0.1),
    fit_D: (bool, bool,) = (True, True),
    F: (float, float,) = (0.3, 0.7),
    fit_F: (bool, bool) = (True, True),
    sigma: (float,) = (0.02,),
    fit_sigma: (bool,) = (True,),
    n_bins: int = 50,
    max_um: float = 0.6,
    verbose: bool = False,
    return_fit_result: bool = False,
    return_hists: bool = False,
    **kwargs,
) -> dict:

    if tracks is None:
        logger.error("No tracks. You must provide some.")
        raise ValueError("No tracks. You must provide some.")

    logger.info(f"fit_spoton_2_0: Fit path: {path}, fit_D: {fit_D}")

    # tracks1 = list(filter(lambda t: len(t) > n_lags + 1, tracks))
    try:
        n_tracks = len(tracks)
    except TypeError:
        raise TypeError(
            f"Expected list of `core.track.Track` objects, got {type(tracks)}"
        )

    logger.info(f"Total {n_tracks} tracks")

    hists = get_jds_histograms(
        tracks, n_lags, bins=n_bins, max_um=max_um, disable_tqdm=not verbose
    )

    fit_result = fit_jd_hist(
        hists=hists,
        dt=dt,
        D=D,
        fit_D=fit_D,
        F=F,
        fit_F=fit_F,
        sigma=sigma,
        fit_sigma=fit_sigma,
        verbose=verbose,
    )

    values = fit_result.params.valuesdict()
    dt = values["dt"]
    sigma = [values[f"sigma{i}"] for i in range(len(sigma))]
    D_all = [values[f"D{i}"] for i in range(len(D))]
    F_all = [values[f"F{i}"] for i in range(len(F))]
    order = np.argsort(D_all)
    D_all = np.array(D_all)[order]
    F_all = np.array(F_all)[order]

    if plot:
        logger.info(
            f"fit_spoton_2_0: Plot fit for (D, F): {(D_all, F_all)}, \
            {n_lags} lags"
        )
        _ = [
            get_error_histogram_vs_model(h, dt, sigma, D_all, F_all, plot=True)
            for h in hists
        ]

    out = {
        "sigma": list(sigma),
        "D": list(D_all),
        "F": list(F_all),
        "dt": dt,
        "n_tracks": n_tracks,
        "chi2": fit_result.chisqr,
        "chi2_norm": fit_result.chisqr / n_bins / n_lags,
        "n_iter": fit_result.nfev,
        "path": path,
        **kwargs,
    }

    if return_fit_result:
        out["fit_result"] = fit_result

    if return_hists:
        out["hists"] = hists

    return out


def result_2_table(*results: [dict], add_columns=[], add_values=[]):
    new_dict = {}
    for replicate, result in enumerate(results):
        for key, value in result.items():
            if isinstance(value, (float, int, str)):
                try:
                    new_dict[key][replicate] = value
                except KeyError:
                    new_dict[key] = {}
                    new_dict[key][replicate] = value
            elif isinstance(value, (list, np.ndarray)):
                for i, sub_value in enumerate(value):
                    try:
                        new_dict[f"{key}_{i}"][replicate] = sub_value
                    except KeyError:
                        new_dict[f"{key}_{i}"] = {}
                        new_dict[f"{key}_{i}"][replicate] = sub_value
            else:
                pass
    #                 print(f'skip {k}')
    df = pd.DataFrame.from_dict(new_dict)
    df.index.name = "replicate"
    for title, value in zip(add_columns, add_values):
        df[title] = value
    return df


def fit_jd_hist(
    hists: list,
    dt: float,
    D: list,
    fit_D: list,
    F: list,
    fit_F: list,
    sigma: float,
    fit_sigma: bool,
    verbose=False,
):

    """
    Fits jd probability functions to a jd histograms.

    Parameters:
    hist (list): histogram values
    D (list): init values for MSD
    F (list): fractions for D, sum = 1
    sigma (float): localization precision guess
    funcs (dict): dictionary with functions sigma, gamma, center, amplitude

    Returns:
    popt (lmfit.minimizerResult): optimized parameters
    """

    from lmfit import Parameters, Parameter, minimize

    def residual(fit_params, data):
        res = cumulative_error_jd_hist(fit_params, data, len(D))
        return res

    fit_params = Parameters()
    # fit_params.add('sigma', value=sigma, vary=fit_sigma, min=0.)
    fit_params.add("dt", value=dt, vary=False)
    try:
        fit_params.add("max_lag", value=max([h.lag for h in hists]), vary=False)
    except TypeError as e:
        logger.error(
            f"problem with `hists`: expected `list`,\
            got `{type(hists)}`"
        )
        raise e

    for i, (d, f_d, f, f_f) in enumerate(zip(D, fit_D, F, fit_F)):
        fit_params.add(f"D{i}", value=d, vary=f_d, min=0.0)
        fit_params.add(f"F{i}", value=f, min=0.0, max=1.0, vary=f_f)

    f_expr = "1"
    for i, f in enumerate(F[:-1]):
        f_expr += f" - F{i}"

    fit_params[f"F{i+1}"] = Parameter(name=f"F{i+1}", min=0.0, max=1.0, expr=f_expr)

    for i, (s, f_s, min_s, max_s) in enumerate(
        zip(sigma, fit_sigma, (0, sigma[0]), (3 * sigma[0], D[-1]))
    ):
        fit_params.add(f"sigma{i}", value=s, min=min_s, max=max_s, vary=f_s)

    logger.debug("start minimize")

    minimizer_result = minimize(residual, fit_params, args=(hists,))

    if verbose:
        logger.info(f"completed in {minimizer_result.nfev} steps")
        minimizer_result.params.pretty_print()

    return minimizer_result


def get_jds_histograms(tracks, max_lag, max_um=0.6, bins=100, disable_tqdm=False):
    """
    For every lag in 1..max_lag compute density histogram

    """

    def single_hist(i):
        lag = i + 1
        _jds = [bayes.get_jd(t.xy, lag, filter_frame_intevals=t.frame) for t in tracks]
        jds = np.concatenate(_jds, axis=0)
        h, edges = np.histogram(jds, bins=bins, range=(0, max_um), density=True)
        return JumpLengthHistogram(h, edges, lag)

    hists = list(
        map(
            single_hist,
            tqdm(
                range(max_lag),
                desc=f"jds hists for {max_lag} lags",
                disable=disable_tqdm,
            ),
        )
    )

    return hists


def get_error_histogram_vs_model(
    hist: JumpLengthHistogram,
    dt: float,
    sigma: list,
    D: list,
    F: list,
    p_density=bayes.p_jd,
    plot=True,
) -> np.ndarray:

    assert len(D) == len(
        F
    ), f"D and F vector should of the same length. \
        Got {len(D)} and {len(F)}"
    assert isinstance(hist, JumpLengthHistogram)

    vector = hist.vector
    values = hist.hist
    width = hist.width
    lag = hist.lag
    model = np.zeros_like(vector)

    for d, f, s, in zip_longest(D, F, sigma, fillvalue=sigma[0]):
        # print('_D,_F, sigma: ', d, f, s)
        model = model + p_density(dt * lag, s, d)(vector) * f

    if plot:
        plt.figure(figsize=(10, 1))
        for i, (_D, _F, s) in enumerate(zip_longest(D, F, sigma, fillvalue=sigma[0])):
            name = "D"

            plt.plot(
                vector,
                p_density(dt * lag, s, _D)(vector) * _F,
                alpha=0.5,
                label=f"{name}$_{i}$: {_D:.2f}, σ: {s:.3f}, fraction {_F:.0%}",
            )
        plt.plot(vector, model, "r-", label="sum model")

        plt.bar(vector, values, width=width, label=f"jd {lag} lag", fill=None, alpha=0.8)

        plt.bar(vector, model - values, width=width, label=f"residuals", fill="red", alpha=0.8)

        plt.xlabel('jump distance, μm')
        plt.title(f"{lag} Δt")
        plt.legend(loc=(1, 0))
        plt.show()

    return model - values


def cumulative_error_jd_hist(
    fit_params: lmfit.Parameters, hist_list: list, num_states: int
) -> np.ndarray:

    p = fit_params.valuesdict()
    # print(p)
    sigma = []
    try:
        for i in range(2):
            sigma.append(p[f"sigma{i}"])
    except KeyError:
        pass

    cum = [
        get_error_histogram_vs_model(
            h,
            dt=p["dt"],
            sigma=sigma,
            D=list(p[f"D{i}"] for i in range(num_states)),
            F=list(p[f"F{i}"] for i in range(num_states)),
            plot=False,
        )
        for h in hist_list
    ]
    return np.concatenate(cum, axis=0)

