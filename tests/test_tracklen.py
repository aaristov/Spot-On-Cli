from fastspt import simulate, tracklen
import numpy as np


def test_decay_rate(num_tracks=1000, p_bleaching=0.05, p_out_of_focus=1e-5):
    tracks = simulate.tracks(
        num_tracks=num_tracks, p_bleaching=p_bleaching, p_out_of_focus=p_out_of_focus, use_tqdm=False
    )

    decay_rate = tracklen.get_track_lengths_dist(tracks, plot=False)
    np.testing.assert_almost_equal(decay_rate, p_bleaching, 1)
