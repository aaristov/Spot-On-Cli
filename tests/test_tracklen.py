from fastspt import simulate, tracklen
import numpy as np


def test_decay(num_tracks=2000, p_bleaching=0.1, p_out_of_focus=1e-5):
    tracks = simulate.tracks(
        num_tracks=num_tracks, p_bleaching=p_bleaching, p_out_of_focus=p_out_of_focus,
    )

    fit = tracklen.get_track_lengths_dist(tracks, max_len=50, plot=False)
    decay_rate = fit["decay rate"]
    np.testing.assert_almost_equal(decay_rate, p_bleaching, 1)
