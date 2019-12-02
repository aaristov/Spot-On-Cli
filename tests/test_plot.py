from fastspt import plot, core
import numpy as np


def test_select_states():
    track = core.Track(
        np.random.rand(10, 2), columns=['x', 'y'],
        units=['um', 'um'])

    track = track.add_column('s0', np.zeros(10), units='')

    track = track.add_column('s1', [0]*5 + [1]*5, units='')

    states = plot.select_states(
        track,
        cols=['s0', 's1'],
        states={'free': [1, 1], 'med': [0, 1], 'bound': [0, 0]}
    )

    assert [
        len(states['free'] == 0),
        len(states['med']) == 5,
        len(states['bound']) == 5]
    return True
