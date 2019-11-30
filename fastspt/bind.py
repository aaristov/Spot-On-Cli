import numpy as np
from fastspt import plot


def get_sqr_displacement(track_xytf):
    track = np.array(track_xytf)
    xy = track[:, :2]
    dxy = xy[1:] - xy[:-1]
    sqr_displacement = (dxy ** 2).sum(axis=1)
    return sqr_displacement


def sliding_window(array1d, size=3):
    '''
    cuts array1d into pieces by size and returns vstack
    '''
    len_ = len(array1d)
    n = len_ - size + 1
    out = np.empty((n, size), dtype=array1d.dtype)
    for i in range(n):
        out[i] = array1d[i:i+size]
    return out


def match_sequence(array2d, template):
    out = np.apply_along_axis(np.array_equal, 1, array2d, template)
    return out


def count_match(match_array_1d):
    return len(match_array_1d.nonzero()[0])


def count_on_off(
    track_xytf, threshold=0.005, on_seq=[1, 0, 0, 0], off_seq=[0, 0, 0, 1]
):
    '''
    Counts on/off cases inside the track.
    Track [[x1, y1, time1, frame1], [...]] 
    is converted do square displacements.
    Displacements are converted to 0 and 1: 
    jumps below threshold get 0, otherwise: 1.
    Then on_seq and off_seq occurances are counted 
    inside the sequence (kon, koff).
    
    Arguments:

    track_xytf: 2d nd.array

    Threshold: float, optional
        Threshold for square displacements. Lower values become 0, higher: 1. 

    on_seq: list of 0 and 1, optional
        sequence of binding event

    on_seq: list of 0 and 1, optional
        sequence of unbinding event
    
    Returned: 
    
    (kon, koff) : tuple (int, int)
        number of binidn g and unbinding event per track
    '''
    assert len(on_seq) == len(off_seq)
    if len(track_xytf) <= len(on_seq):
        return (0, 0)
    sd = get_sqr_displacement(track_xytf)
    categories = plot.threshold_sqr_displacement(sd, thr=threshold)
    slides = sliding_window(categories, size=len(on_seq))
    on_match = match_sequence(slides, on_seq)
    off_match = match_sequence(slides, off_seq)
    kon = count_match(on_match)
    koff = count_match(off_match)
    return (kon, koff)


def count_on_off_tracks(
    tracks, min_len=7, threshold=0.005, 
    on_seq=[1, 0, 0, 0], off_seq=[0, 0, 0, 1]
):
    '''
    Counts template occurance in the list of treack with format  
    [[x1, y1, time1, frame1], [...]].
    Return (kon, koff) for all tracks.
    '''
    if min_len:
        tracks = list(filter(lambda t: len(t) >= min_len, tracks))
    
    def get_counts(track): 
        return count_on_off(track, threshold, on_seq, off_seq)
    
    matched_counts = list(map(get_counts, tracks))
    return np.array(matched_counts).sum(axis=0)