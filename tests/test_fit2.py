import pytest
from fastspt import fit2, simulate
import numpy as np
import pandas as pd


def test_sim_tracks():
    tracks = simulate.tracks(num_tracks=100)
    assert len(tracks) == 100


def test_fit_tracks():

    sim_params = dict(
        num_tracks=4e3,
        dt=0.06,
        D_bound=0.0,
        D_free=0.05,
        loc_error=0.03,
        p_binding=0.0001,
        p_unbinding=0.001,
        p_bleaching=1e-1,
        p_out_of_focus=1e-5,
        min_len=5,
        use_tqdm=False,
    )

    fit_params = dict(
        n_lags=7,
        plot=False,
        dt=0.06,
        D=(0.0, 0.1),
        fit_D=(True, True),
        F=(0.5, 0.5),
        fit_F=(True, True),
        sigma=(0.2,),
        fit_sigma=(True,),
        n_bins=100,
        max_um=2,
        verbose=False,
    )

    F_bound = sim_params["p_binding"] / (
        sim_params["p_binding"] + sim_params["p_unbinding"]
    )

    tracks = simulate.tracks(**sim_params)

    assert len(tracks) == sim_params["num_tracks"]

    fit = fit2.fit_spoton_2_0(tracks, **fit_params)

    np.testing.assert_almost_equal(
        fit["D"][1],
        (sim_params["D_free"]),
        decimal=0,
        err_msg=f"D_free input: {sim_params['D_free']}, fit: {fit['D'][1]}",
    )

    np.testing.assert_almost_equal(fit["sigma"][0], sim_params["loc_error"], decimal=2)

    np.testing.assert_almost_equal(
        fit["F"][0],
        F_bound,
        decimal=1,
        err_msg=f"Fitted F_bound {fit['F'][0]:.2f}, \
            simulated {F_bound:.2f}",
    )


def test_fit_jd_hist_bad_input():
    with pytest.raises(TypeError):
        fit2.fit_jd_hist(None, 0, [0, 0], [1, 1], [0, 1], [1, 1], 0, 1, 0)

    with pytest.raises(AttributeError):
        fit2.fit_jd_hist([None], 0, [0, 0], [1, 1], [0, 1], [1, 1], 0, 1, 0)


def test_fit_2_sigmas(sigmas=[0.01, 0.03], n_tracks=5000, D_free=0.1, plot=False):
    """
    Simulate confined state
    """
    sim1 = simulate.tracks(
        n_tracks=n_tracks, D_free=D_free, loc_error=sigmas[0], use_tqdm=False
    )

    sim2 = simulate.tracks(
        n_tracks=n_tracks, D_free=D_free, loc_error=sigmas[1], use_tqdm=False
    )

    sim = sim1 + sim2

    fit_2_sigmas = fit2.fit_spoton_2_0(
        sim,
        D=[0, 0.0, 0.1],
        F=[0.3, 0.3, 0.4],
        fit_D=[False, False, True],
        fit_F=[1, 1, 1],
        sigma=[0.01, 0.04],
        fit_sigma=[True, True],
        n_bins=100,
        dt=0.06,
        n_lags=3,
        plot=plot,
    )

    fit_1_sigmas = fit2.fit_spoton_2_0(
        sim,
        D=[0, 0.1],
        F=[0.3, 0.7],
        fit_D=[False, True],
        fit_F=[1, 1],
        sigma=[0.01],
        fit_sigma=[True, True],
        n_bins=100,
        dt=0.06,
        n_lags=3,
        plot=plot,
    )

    for s, S in zip(sigmas, fit_2_sigmas["sigma"]):
        np.testing.assert_almost_equal(S, s, 2)

    assert fit_1_sigmas["chi2_norm"] > fit_2_sigmas["chi2_norm"]

    return fit_2_sigmas


def test_fit_high_lags():
    tracks = simulate.tracks(num_tracks=100, min_len=3)
    fit = fit2.fit_spoton_2_0(tracks, n_lags=5, plot=False)
    assert isinstance(fit, dict)


def test_fit_return_hists():
    tracks = simulate.tracks(num_tracks=100, min_len=3)
    fit = fit2.fit_spoton_2_0(tracks, n_lags=5, plot=False, return_hists=True)
    hists = fit["hists"]
    assert len(hists) == 5
    assert [isinstance(h, fit2.JumpLengthHistogram) for h in hists]


def test_fit_return_fit():
    tracks = simulate.tracks(num_tracks=100, min_len=3)
    fit = fit2.fit_spoton_2_0(
        tracks, n_lags=5, plot=False, return_hists=False, return_fit_result=True
    )
    fit_result = fit["fit_result"]
    assert isinstance(fit_result, fit2.lmfit.minimizer.MinimizerResult)


def test_result_2_table():
    data = [simulate.tracks(100) for _ in range(2)]
    fits = [fit2.fit_spoton_2_0(d, plot=False) for d in data]
    table = fit2.result_2_table(*fits)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 2
