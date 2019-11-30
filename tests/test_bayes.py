from fastspt import bayes, simulate, plot
import pytest
import numpy as np


def test_2_states_classification(Ds=[0.0, .08], num_tracks=1000):

    tracks = simulate.tracks(
        num_tracks=num_tracks,
        D_bound=Ds[0],
        D_free=Ds[1],
        loc_error=0.02,
        p_binding=1e-2,
        p_unbinding=4e-2,
        p_out_of_focus=1e-5,
        p_bleaching=0.1,
        use_tqdm=False
    )

    bayes_filter = bayes.BayesFilter(
        **{'D': Ds, 'F': [0.2, 0.8], 'sigma': [0.02], 'dt': 0.06})

    predictions = [bayes_filter.predict_states(
        track,
        max_lag=4,
        smooth=0
    ) for track in tracks]

    n_bound = sum([sum(track.free == 0) for track in tracks])[0]
    total = sum([len(track) for track in tracks])
    n_predicted = sum([sum(prediction == 0) for prediction in predictions])

    print(f'bound fraction sim: {n_bound/total:.1%}, \
        predicted: {n_predicted/total:.1%}')
    np.testing.assert_almost_equal(n_predicted/total, n_bound/total, 1)

    r = np.arange(0.01, 0.3, 0.01)
    fig1 = bayes_filter.plot_bayes(r)
    fig2 = bayes_filter.plot_jd(r)

    assert isinstance(fig1, plot.plt.Figure)
    assert isinstance(fig2, plot.plt.Figure)

    assert isinstance(repr(bayes_filter), str)
    assert isinstance(repr(bayes_filter[0]), str)

    with pytest.raises(AssertionError):
        bayes_filter(True)

    with pytest.raises(AssertionError):
        bayes_filter([1, 2], 0)


def test_2_states_classification_smooth(Ds=[0.0, .08], num_tracks=1000):

    tracks = simulate.tracks(
        num_tracks=num_tracks,
        D_bound=Ds[0],
        D_free=Ds[1],
        loc_error=0.02,
        p_binding=1e-2,
        p_unbinding=4e-2,
        p_out_of_focus=1e-5,
        p_bleaching=0.1,
        use_tqdm=False
    )

    bayes_filter = bayes.BayesFilter(
        **{'D': Ds, 'F': [0.2, 0.8], 'sigma': [0.02], 'dt': 0.06})

    predictions = [bayes_filter.predict_states(
        track,
        max_lag=4,
        smooth=1
    ) for track in tracks]

    n_bound = sum([sum(track.free == 0) for track in tracks])[0]
    total = sum([len(track) for track in tracks])
    n_predicted = sum([sum(prediction == 0) for prediction in predictions])

    print(f'bound fraction sim: {n_bound/total:.1%}, \
        predicted: {n_predicted/total:.1%}')
    np.testing.assert_almost_equal(n_predicted/total, n_bound/total, 1)


def test_p_jd():
    step = 0.01
    r = np.arange(0.01, 0.3, step)

    dist = bayes.p_jd(time=0.06, sigma=0.02, D=0)(r)

    np.testing.assert_almost_equal(sum(dist) * step, 1, 2)

    with pytest.raises(AssertionError):
        dist = bayes.p_jd(0, 0.02, 0)(r)

    with pytest.raises(AssertionError):
        dist = bayes.p_jd(0.1, 0, 0)(r)

    with pytest.raises(AssertionError):
        dist = bayes.p_jd(0.1, 0.2, -1)(r)

    with pytest.raises(TypeError):
        dist = bayes.p_jd(0.1, 0.2, [0, 5])(r)


def test_switching_rates():

    tracks = simulate.tracks(
        num_tracks=1e3,
        dt=0.06,
        D_bound=0,
        D_free=0.06,
        loc_error=0.02,
        p_binding=1e-4,
        p_unbinding=1e-3,
        p_bleaching=1e-1,
        p_out_of_focus=1e-5,
        min_len=5,
        use_tqdm=False
    )

    out = bayes.get_switching_rates(tracks, fps=15, column='free')

    assert isinstance(out, dict)

    np.testing.assert_almost_equal(out['u_rate_frame'], 1e-3, 2)
    np.testing.assert_almost_equal(out['b_rate_frame'], 1e-4, 3)


def test_big_lag():
    track = simulate.track()
    assert bayes.get_jd(track, lag=len(track)) == []
