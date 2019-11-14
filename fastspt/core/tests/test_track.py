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