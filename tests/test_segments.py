from fastspt import simulate, segments
import numpy as np


def test_select_pops(
    p_binding=1e-2,
    p_unbinding=4e-2,
    num_tracks=100,
    p_bleaching=0.05
):

    tracks = simulate.tracks(
        num_tracks, p_binding=p_binding, p_unbinding=p_unbinding,
        p_bleaching=p_bleaching)

    pops = segments.get_populations(
        tracks, column_with_states='free', values=(0, 1), min_len=3)

    assert len(pops) == 2

    for p in pops:
        assert len(p) > 0

    n_bound = sum([len(p) for p in pops[0]])
    n_free = sum([len(p) for p in pops[1]])
    bound_f = n_bound / (n_bound + n_free)
    bound_sim = p_binding / (p_unbinding + p_binding)

    np.testing.assert_almost_equal(bound_f, bound_sim, 1)


def test_add_seg_id_to_track():
    track = simulate.track()
    new_track = segments.add_seg_id_to_track(
        track, column_with_states='free',
        start_id=0, new_column='seg_id', return_new_id=False)
    assert isinstance(new_track.seg_id, np.ndarray)
