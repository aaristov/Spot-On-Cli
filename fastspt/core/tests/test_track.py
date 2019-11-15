from fastspt import simulate, plot, core
from operator import __add__
from functools import reduce
import numpy as np
import pytest

def test_track_add():
    tracks = simulate.tracks(20)
    sum_tracks = reduce(__add__, tracks)

    assert isinstance(sum_tracks, core.Track)
    assert len(sum_tracks) == sum(map(len, tracks))

def test_bad_column():
    track = simulate.track()

    with pytest.raises(ValueError):
        track.xyz

def test_set_item():
    
    track = simulate.track()

    frames = np.arange(10, len(track)+10)
    track[:,3] = frames

    np.testing.assert_array_equal(track.frame.flat, frames)

    return track


def test_bad_set_item():
    
    track = simulate.track()

    sigma = np.random.randn(len(track), 2)

    with pytest.raises(ValueError):
        track[:, 4] = sigma

def test_get_item_vector():
    
    track = simulate.track()

    sub_track = track[track.frame < 3]

    assert len(sub_track) == 3

def test_repr():
    
    track = simulate.track()

    assert isinstance(repr(track), str)