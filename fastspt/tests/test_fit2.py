import pytest
from fastspt import fit2, simulate
import numpy as np

sim_params = dict(
    num_tracks=1e3, 
    dt=0.06, 
    D_free=0.06, 
    loc_error=0.02, 
    p_binding=3e-2, 
    p_unbinding=1e-1, 
    p_bleaching=1e-1, 
    p_out_of_focus=1e-5, 
    min_len=5, 
    use_tqdm=False
    )

sim_params['F_bound'] = sim_params['p_binding']/(sim_params['p_binding'] + sim_params['p_unbinding'])

fit_params = dict(
    n_lags=1, 
    plot=False, 
    dt=0.06, 
    D=(0, 0.1), 
    fit_D=(False, True), 
    F=(0.5, 1 - 0.5), 
    fit_F=(True, True), 
    sigma=(0.1,), 
    fit_sigma=(True,), 
    n_bins=50, 
    max_um=0.6, 
    verbose=False
)

tracks = simulate.tracks(**sim_params)

fit = fit2.fit_spoton_2_0(tracks, **fit_params)


def test_sim_tracks():
    assert len(tracks) == sim_params['num_tracks']

def test_fit_tracks():

    np.testing.assert_almost_equal(
        fit['D'][1], sim_params['D_free'], 
        decimal=2,
        err_msg=f"D_free input: {sim_params['D_free']}, fit: {fit['D'][1]}")

    np.testing.assert_almost_equal(fit['sigma'][0], sim_params['loc_error'], decimal=2)

    np.testing.assert_almost_equal(
        fit['F'][0], sim_params['F_bound'],         
        decimal=1, 
        err_msg=f"Fitted F_bound {fit['F'][0]:.2f}, simulated {sim_params['F_bound']:.2f}"
    )

def test_fit_jd_hist_bad_input():
    with pytest.raises(TypeError):
        fit2.fit_jd_hist(None, 0, [0,0], [1,1], [0,1], [1,1], 0, 1, 0)

    with pytest.raises(AttributeError):
        fit2.fit_jd_hist([None], 0, [0,0], [1,1], [0,1], [1,1], 0, 1, 0)


def test_fit_2_sigmas(sigmas = [0.01, 0.03], n_tracks=5000, D_free=0.1, plot=False):
    '''
    Simulate confined state
    '''
    sim1 = simulate.tracks(
        n_tracks=n_tracks, 
        D_free=D_free,
        loc_error=sigmas[0],
        use_tqdm=False)

    sim2 = simulate.tracks(
        n_tracks=n_tracks, 
        D_free=D_free,
        loc_error=sigmas[1],
        use_tqdm=False)

    sim = sim1 + sim2

    fit_2_sigmas = fit2.fit_spoton_2_0(
        sim,
        D=[0, 0.0, 0.1],
        F=[0.3, 0.3, 0.4],
        fit_D=[False, False, True],
        fit_F=[1,1,1],
        sigma=[0.01, 0.04],
        fit_sigma=[True, True],
        n_bins=100,
        dt=0.06,
        n_lags=3,
        plot=plot)

    fit_1_sigmas = fit2.fit_spoton_2_0(
        sim,
        D=[0, 0.1],
        F=[0.3, 0.7],
        fit_D=[False, True],
        fit_F=[1,1],
        sigma=[0.01],
        fit_sigma=[True, True],
        n_bins=100,
        dt=0.06,
        n_lags=3,
        plot=plot)

    for s, S in zip(sigmas, fit_2_sigmas['sigma']):
        np.testing.assert_almost_equal(S, s, 2)
    
    assert fit_1_sigmas['chi2_norm'] > fit_2_sigmas['chi2_norm']

    return fit_2_sigmas

