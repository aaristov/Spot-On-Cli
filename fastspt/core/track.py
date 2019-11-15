import numpy as np
from functools import partial
from tqdm.auto import tqdm
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class Track:
    '''
    Master class to storing tracks.
    Raise:
    ------
    ValueError if column not found
    '''
    
    def __init__(
        self, 
        array:np.array, 
        columns=['x', 'y', 't', 'f', 'state', 'id'], 
        units=['um', 'um', 'sec', '', '', '']
        ):
        array = np.array(array)
        assert len(columns) == array.shape[1]
        self.columns = columns
        self.array = array
        self.units = units

    def sub_columns(self, ind):
        return self.array[:, ind]

    def crop_frames(self, start, end):
        return Track(self.array[start:end], self.columns, self.units)
        
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
                raise ValueError(f'column `{item}` not found, available columns: `{self.columns}`')

    def __getitem__(self, key):
        try:
            return self.array[tuple(key)]
        except IndexError:
            return self.array[np.ravel(key)]
    
    def __setitem__(self, key, value):
        try:
            arr = self.array#.copy()
            arr[key] = value
            # return Track(arr, self.columns, self.units)
            return self
        except ValueError as e:
            logger.warning('unable to set the new values: ', *e.args)
            raise e
            
            
    
    def __repr__(self):
        return repr(pd.DataFrame(columns=[f'{c} [{u}]' for c, u in zip(self.columns, self.units)], data=self.array))
    
    def add_column(self, title:str, values, units):
        assert len(values) == len(self.array)
        cols = self.columns.copy()
        cols.append(title)

        _units = self.units.copy()
        _units.append(units)

        new_array = np.insert(self.array, self.array.shape[1], values, axis=1)

        return Track(new_array, cols, _units)

    def __add__(self, *another_track):
        for track in another_track:
            assert isinstance(track, Track), TypeError(f'type mismatch: {type(track)} != {type(self)}')

            for c, s in zip(self.columns, track.columns):
                assert c == s , f'columns mismatch: {c} != {s}'
        cols = self.columns
        units = self.units
        vals = self.array
        another_vals = [t.values  for t in another_track]
        new_vals = np.concatenate((vals, *another_vals), axis=0)
        return Track(new_vals, cols, units)