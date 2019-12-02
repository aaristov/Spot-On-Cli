from fastspt import core
import numpy as np
from functools import reduce
from tqdm.auto import tqdm


# tools to break tracks into segments
# based on predefined column with states


def get_populations(tracks, column_with_states="free", values=(0, 1), min_len=3):
    """
    Breaks tracks into segments based on predefined column with states

    Parameters:
    -----------
    tracks: list of simulate.Track objects
        list of tracks to be analyzed
    column_with_states: string, default `free`
        title of the column containing state information
    values: tuple of any type, default (0, 1)
        Whatever values you expect in the `column_with_states`
    min_len: int, default 3
        Minimal length of output segments.
        All segments shorter than this number will be regected.

    Return:
    -------
    tuple of lists of segments with the good states.

    """
    ttt = add_seg_id_to_tracks(tracks, column_with_states)
    segments = reduce(lambda a, b: a + b, map(break_into_segments, ttt))
    segments_longer = list(filter(lambda t: len(t) > min_len, segments))
    pops = list(
        list(filter(lambda t: t.col(column_with_states).mean() == v, segments_longer))
        for v in values
    )

    return pops


def assign_seg_id(states, id0=1):
    n = id0
    ids = [
        n,
    ]
    for v2, v1 in zip(states[1:], states[:-1]):
        dif = v2 - v1
        if dif != 0:
            n = n + 1
        ids.append(n)
    return ids


def add_seg_id_to_track(
    track: core.Track,
    column_with_states="free",
    start_id=0,
    new_column="seg_id",
    return_new_id=False,
) -> core.Track:

    states = track.col(column_with_states)
    ids = assign_seg_id(states, start_id)
    new_track = track.add_column(new_column, ids, "")
    if return_new_id:
        return new_track, max(ids)
    return new_track


def add_seg_id_to_tracks(tracks: list, column_with_states="free", new_column="seg_id"):
    cur_id = 0
    new_tracks = []
    for t in tqdm(tracks):
        new_track, i = add_seg_id_to_track(
            t, column_with_states, cur_id, new_column, return_new_id=True
        )
        new_tracks.append(new_track)
        cur_id = i + 1
    return new_tracks


def break_into_segments(track, column_with_seg_id="seg_id"):
    _, indices = np.unique(track.col(column_with_seg_id), return_index=True)
    indices = list(indices)
    indices.append(None)
    out = []
    for i1, i2 in zip(indices[:-1], indices[1:]):
        out.append(track.crop_frames(i1, i2))
    return out
