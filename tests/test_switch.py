from fastspt import simulate
from fastspt import switch
import numpy as np
import pytest


def test_get_switching_matrix(num_tracks=1000, p_binding=0.01, p_unbinding=0.1):
    tracks = simulate.tracks(
        num_tracks=num_tracks,
        p_binding=p_binding,
        p_unbinding=p_unbinding,
        use_tqdm=False,
    )

    switching_matrix = switch.get_switching_matrix(tracks, column="free", n_states=2)
    assert switching_matrix.shape == (2, 2)

    np.testing.assert_almost_equal(
        switching_matrix[1, 0], p_binding, decimal=abs(np.log10(p_binding).astype(int))
    )
    np.testing.assert_almost_equal(
        switching_matrix[0, 1],
        p_unbinding,
        decimal=abs(np.log10(p_unbinding).astype(int)),
    )

    switching_matrix, raw_matrix = switch.get_switching_matrix(
        tracks, column="free", n_states=2, return_raw_counts=True
    )
    assert raw_matrix.shape == (2, 2)

    with pytest.raises(ValueError):
        tr = simulate.track(min_len=3)
        tr1 = tr.add_column("states", np.arange(len(tr)), "test")
        switch.get_switching_matrix([tr1], column="states", n_states=2)
