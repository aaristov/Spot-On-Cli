import numpy as np
import fastspt.simulate as sim
from scipy.ndimage import gaussian_filter1d as gf1
from functools import reduce

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

def classify_bound_segments(track:sim.Track, sigma:float, max_lag:int=4, col_name='prediction', extrapolate_edges=True, verbose=False, return_p_unbinds=False):
    jds = [get_jd(track.xy, lag=l, extrapolate=1) for l in range(1,max_lag+1)]

    jds = list(filter(len, jds))

    p_unbinds = list(map(lambda i: list(map(lambda x: cdf_unbound(sigma, x), jds[i])), range(len(jds))))
#     print(p_unbinds)
    if verbose:
        print('jds: ', jds)
        print(p_unbinds)
    try:
        hl = max_lag // 2
        bound_vector = gf1(np.median(p_unbinds, axis=0), hl) > 0.5

        if verbose:
            print('half lag: ', hl)
            print('bound_vector: ', bound_vector)

        if extrapolate_edges:
            bound_vector[:hl] = bound_vector[hl]
            bound_vector[-hl:] = bound_vector[-hl - 1]
            
            if verbose:
                print('after extrapolation: bound_vector: ', bound_vector)
        
        
    except TypeError as e:
        print(track)
        print(jds)
        bound_vector = [None] * len(track) 
        raise e
    
#     new_track = np.insert(track, track.shape[1], bound_vector, axis=1)
    new_track = track.add_column(col_name, bound_vector, int)

    if return_p_unbinds:
        return new_track, p_unbinds
    else:
        return new_track


sum_list = lambda l: reduce(lambda a, b: a + b, l)



def get_switching_rates(xytfu:sim.Track, fps:float, column='free'):

    n_bound_spots = sum_list(map(lambda a: sum(a.col(column)[:] == 0), xytfu))
    n_bound_spots_for_rates = sum_list(map(lambda a: sum(a.col(column)[:-1] == 0), xytfu))
    n_unbound_spots_for_rates = sum_list(map(lambda a: sum(a.col(column)[:-1] == 1), xytfu))
    print(n_bound_spots, n_bound_spots_for_rates)
    n_total_spots = sum_list(map(lambda a: len(a), xytfu))

    
#     n_total_segments = n_total_spots - len(xytfu)
#     n_bound_segments = n_bound_spots - len(bound)

    print( f'bound fraction based on number of spots: {n_bound_spots} / {n_total_spots} = {n_bound_spots / n_total_spots:.1%}')
#     print( f' bound fraction based on number of segments: {n_bound_segments} / {n_total_segments} = {n_bound_segments / n_total_segments:.1%}')
    
    get_n_switch_unbind = lambda xytfu: sum_list(map(lambda a: sum(a.col(column)[1:] - a.col(column)[:-1] == 1), xytfu))
    get_n_switch_bind = lambda xytfu: sum_list(map(lambda a: sum(a.col(column)[1:] - a.col(column)[:-1] == -1), xytfu))

    n_switch_unbind = get_n_switch_unbind(xytfu)
    n_switch_bind = get_n_switch_bind(xytfu)

    print(f'{n_switch_bind} binding events, {n_switch_unbind} unbinding events')
    fps = 15
    u_rate_frame = n_switch_unbind / n_bound_spots_for_rates
    b_rate_frame = n_switch_bind / n_unbound_spots_for_rates
    print(f'Unbinding switching rates: {u_rate_frame:.1%} per frame, {u_rate_frame * fps:.1%} per second {fps} fps')
    print(f'Binding switching rates: {b_rate_frame:.1%} per frame, {b_rate_frame * fps:.1%} per second {fps} fps')
    print(f'Bound fraction based on switching rates: {b_rate_frame / (b_rate_frame + u_rate_frame):.1%}')
    
    return {'F_bound': n_bound_spots / n_total_spots, 'u_rate_frame': u_rate_frame, 'b_rate_frame': b_rate_frame }
    
