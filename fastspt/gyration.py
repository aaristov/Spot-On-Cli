import numpy as np
from fastspt import bind

def get_gyration_radius_track(track_xytf):
    xy = track_xytf[:, :2]
    cxy = xy.mean(axis=0).reshape((1,2))
    dxy = xy - cxy
    dxy_sq = np.math.pow(dxy, 2)
    d_sq = dxy_sq.sum(axis=1)
    
    rg = np.sqrt(
        np.sum(d_sq) / len(d_sq)
    )
    return rg


def get_gyration_radius_sd(sd):
    s = np.sum(sd)
    rg = np.sqrt( s / (2 * (len(sd) + 1) ** 2))
    return rg
    