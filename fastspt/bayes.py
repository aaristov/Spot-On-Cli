import numpy as np
import fastspt.simulate as sim
from scipy.ndimage import gaussian_filter1d as gf1

d_ = lambda time, sigma, D: D * time + sigma ** 2
p_jd = lambda time, sigma, D: lambda r: r / (2 * d_( time, sigma, D)) * np.exp(- r ** 2 / ( 4 * d_( time, sigma, D)))
deriv = lambda time, sigma, D: lambda r: 1 / (2 *  d_( time, sigma, D)) * (1 -  r ** 2 / (2 * d_( time, sigma, D))) * np.exp(- r **2 / (4 * d_( time, sigma, D)))   
r_max = lambda time, sigma, D: np.sqrt(2 * d_(time, sigma, D))
p_jd_max =lambda time, sigma, D:  p_jd(time, sigma, D)(r_max(time, sigma, D))
p_jd_norm =  lambda time, sigma, D: lambda r: p_jd(time, sigma, D)(r) / p_jd_max( time, sigma, D)

def cdf_bound(sigma, r):
    jd_max = r_max(1, sigma, 0)
    if r <= jd_max:
        return 1
    else:
        return p_jd_norm(1, sigma, 0)(r)

def cdf_unbound(sigma, r):
    return 1 - cdf_bound(sigma, r)

def get_jd(xy:np.ndarray, lag=1, extrapolate=False):
#     xy = [[x1, y1]
#           [x2, y2]
#             ...   ]   

    if len(xy) > lag:
        dxy = xy[lag:] - xy[:-lag] 
        jd = np.sqrt((dxy ** 2).sum(axis=1))
        if extrapolate:
            while len(jd) < len(xy):
                jd = np.concatenate(([jd[0]], jd[:]))
#                 even = lag % 2 == 0
                if len(jd) < len(xy):
                    jd = np.concatenate((jd[:], [jd[-1]]))

        return jd
    else:
        return []

def classify_bound_segments(track:sim.Track, sigma:float, max_lag:int=4):
    jds = [get_jd(track.xy, lag=l, extrapolate=1) for l in range(1,max_lag+1)]

    jds = list(filter(len, jds))

    p_unbinds = list(map(lambda i: list(map(lambda x: cdf_unbound(sigma, x), jds[i])), range(len(jds))))
#     print(p_unbinds)
    try:
        bound_vector = gf1(np.median(p_unbinds, axis=0), max_lag) > 0.5
        
    except TypeError as e:
        print(track)
        print(jds)
        bound_vector = [None] * len(track) 
        raise e
    
#     new_track = np.insert(track, track.shape[1], bound_vector, axis=1)
    new_track = track.add_column('prediction', bound_vector, int)
    return new_track