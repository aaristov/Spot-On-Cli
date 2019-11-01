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
    sigma=0.1, 
    fit_sigma=True, 
    n_bins=50, 
    max_um=0.6, 
    verbose=False
)

tracks = simulate.tracks(**sim_params)

fit = fit2.fit_spoton_2_0(tracks, **fit_params)

class TestFit:

    
    def test_sim_tracks(self):
        assert len(tracks) == sim_params['num_tracks']

    def test_fit_tracks(self):

        np.testing.assert_almost_equal(
            fit['D_free'], sim_params['D_free'], 
            decimal=2,
            err_msg=f"D_free input: {sim_params['D_free']}, fit: {fit['D_free']}")

        np.testing.assert_almost_equal(fit['sigma'], sim_params['loc_error'], decimal=2)

        np.testing.assert_almost_equal(
            fit['F_bound'], sim_params['F_bound'],         
            decimal=1, 
            err_msg=f"Fitted F_bound {fit['F_bound']:.2f}, simulated {sim_params['F_bound']:.2f}"
        )

    def test_fit_jd_hist_bad_input(self):
        with pytest.raises(TypeError):
            fit2.fit_jd_hist(None, 0, [0,0], [1,1], [0,1], [1,1], 0, 1, 0)

        with pytest.raises(AttributeError):
            fit2.fit_jd_hist([None], 0, [0,0], [1,1], [0,1], [1,1], 0, 1, 0)

        