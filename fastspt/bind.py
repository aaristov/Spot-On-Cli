import numpy as np
from fastspt import plot


def sliding_window(array1d, size=3):
    '''
    cuts array1d into pieces by size and returns vstack
    '''
    l = len(array1d)
    n = l - size + 1
    out = np.empty((n, size), dtype='int')
    for i in range(n):
        out[i] = array1d[i:i+size]
    return out


def match_sequence(array2d, template):
    out = np.apply_along_axis(np.array_equal, 1, array2d, template)
    return out

def count_match(match_array_1d):
    return len(match_array_1d.nonzero()[0])

def count_on_off(track_xytf, threshold=0.005, on_seq=[1,0,0,0], off_seq=[0,0,0,1]):
    '''
    Counts on/off cases inside the track.
    Track [[x1, y1, time1, frame1], [...]] is converted do square displacements.
    Displacements are converted to 0 and 1: jumps below threshold get 0, otherwise: 1.
    Then on_seq and off_seq occurances are counted inside the sequence (kon, koff).
    Returned: tuple(kon, koff)
    '''
    assert len(on_seq) == len(off_seq)
    if len(track_xytf) <= len(on_seq):
        return False
    sd = plot.get_sqr_displacement(track_xytf)
    categories = plot.threshold_sqr_displacement(sd, thr=threshold)
    slides = sliding_window(categories, size=len(on_seq))
    on_match = match_sequence(slides, on_seq)
    off_match = match_sequence(slides, off_seq)
    kon = count_match(on_match)
    koff = count_match(off_match)
    return (kon, koff)
