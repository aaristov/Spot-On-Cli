import numpy as np
from functools import partial
from tqdm.auto import tqdm
import pandas as pd

class Track:

    def __init__(
        self, 
        array:np.array, 
        columns=['x', 'y', 't', 'f', 'state', 'id'], 
        units=['um', 'um', 'sec', int, int, int]
        ):
        array = np.array(array)
        assert len(columns) == array.shape[1]
        self.columns = columns
        self.array = array
        self.units = units

    def sub_columns(self, ind):
        return self.array[:, ind]

    def col(self, name):
        return np.ravel(self.__getattr__(name))

    
    def __len__(self):
        return len(self.array)

    @property
    def values(self):
        return self.array

    def __getattr__(self, item):
        try:
            ind = self.columns.index(item)
            return self.sub_columns((ind,))
        except ValueError:
            if item == 'xy':
                ind = tuple(self.columns.index(it) for it in item)
                return self.sub_columns(ind)
            else:
                return []
    
    def __getitem__(self, key):
        try:
            return self.array[key]
        except IndexError:
            return self.array[np.ravel(key)]
    
    def __setitem__(self, key, value):
        try:
            arr = self.array.copy()
            arr[key] = value
            return Track(arr, self.columns, self.units)
        except Exception as e:
            print('unable to set the new values: ', *e.args)
            return self
            
            
    
    def __repr__(self):
        return repr(pd.DataFrame(columns=self.columns, data=self.array))
    
    def add_column(self, title:str, values, units):
        assert len(values) == len(self.array)
        cols = self.columns.copy()
        cols.append(title)

        units = self.units.copy()
        units.append(units)

        new_array = np.insert(self.array, self.array.shape[1], values, axis=1)

        return Track(new_array, cols, units)


def track(
    track_id=0,
    start_time=0,
    dt = 0.06, 
    D_free=0.06,
    loc_error=0.02, 
    p_binding=1e-4, 
    p_unbinding=1e-3, 
    p_bleaching=1e-1, 
    p_out_of_focus=1e-2, 
    min_len=5
    ):
    '''
    Uses physical parameters to get timepoints xy diffusing or bound in space. 
    All localizations are equally affected by localization error Sigma.
    Parameters:
    -----------

    Return:
    -------
    np.array with each line containing [x, y, time, sigma, bound?, id]
    '''

    bound_fraction = p_binding / (p_binding + p_unbinding)
    
    bound = np.random.random() < bound_fraction
    stop = False
    steps = 0
    dxytsbi = [] # time step for x, y, time, sigma, bound?, id
    
    while not stop:
        
#         sigma = np.random.normal(*loc_error)
            
        if bound:
            stop = np.random.random() < p_bleaching and steps >= min_len
            D_ = 0
            switch = np.random.random() < p_unbinding
        else:
            stop = np.random.rand(1)[0] < p_bleaching + p_out_of_focus and steps >= min_len
#             D_free_ = np.random.normal(*D_free)
            D_ = np.sqrt( 2 * D_free  * dt)
            switch = np.random.random() < p_binding
        
        if switch:
            bound = not bound
#             print(['u', 'b'][int(bound)], end='')
        dxy = np.random.standard_normal(size=(2,)) * D_
        
#         dxytsb.loc[steps] = [xy[0], xy[1], steps * dt + start_time, sigma, int(bound)]
        dxytsbi.append([dxy[0], dxy[1], steps * dt + start_time, 0, steps, int(not bound), track_id])
        steps += 1
    
    out = np.array(dxytsbi)
    out[:, :2] = np.cumsum(out[:, :2], axis=0)
    sigma = .001 * np.random.standard_gamma(loc_error * 1000., size=(len(out), 2))
    out[:, :2] = out[:, :2] + sigma * np.random.standard_normal(size=(len(out), 2))
    out[:, 3] = sigma.mean(axis=1)
  
    return Track(out, columns=['x', 'y', 't', 'sigma', 'frame', 'free', 'id'])

def tracks(
    num_tracks = 1e3,
    dt = 0.06, 
    D_free=0.06,
    loc_error=0.02, 
    p_binding=1e-4, 
    p_unbinding=1e-3, 
    p_bleaching=1e-1, 
    p_out_of_focus=1e-1, 
    min_len=5,
    fun=track,
    use_tqdm=True
):
    print(f'Simulating {num_tracks} tracks')
    tracks = list(
        map(
            lambda id: fun(
                track_id=id,
                start_time=id, 
                dt=dt, 
                D_free=D_free, 
                loc_error=loc_error, 
                p_binding=p_binding, 
                p_unbinding=p_unbinding, 
                p_bleaching=p_bleaching,
                p_out_of_focus=p_out_of_focus,
                min_len=min_len
                ),
            tqdm(range(num_tracks), disable=not use_tqdm)
            )
        )
    
    
    return tracks


        