## readers.py is part of the fastspt library
## By MW, GPLv3+, Oct. 2017
## readers.py imports many file formats widespread in SPT analysis
## Imported from Spot-On

from fastspt import format4DN
import scipy.io, os, json, xmltodict
import numpy as np
import pandas as pd
from fastspt.core import Track


def get_exposure_ms_from_path(path, pattern='bleach_(.*?)ms_'):
    import re

    regx = re.compile(pattern)
    found = regx.findall(path)
    if found:
        return float(found[0])
    else:
        return None
    
assert get_exposure_ms_from_path('bleach_1.7ms_no_strobo') == 1.7
assert get_exposure_ms_from_path('bleach_60ms_strobo_10ms') == 60


## ==== evalSPT
def read_evalspt(fn, framerate, pixelsize):
    return read_arbitrary_csv(fn, col_traj=3, col_x=0, col_y=1, col_frame=2,
                              framerate=framerate/1000., pixelsize=pixelsize/1000.,
                              sep="\t", header=None)

## ==== 4DN format
def read_4DN(fn, return_header=False, return_pandas=False):
    """Read the 4D nucleome SPT format
    Documented at: https://docs.google.com/document/d/1SKljQyuTNtKQOxOD5AC9ZBqZZETDXtUz1BImGZ99Z3M/edit
    """
    return format4DN.read_4DN(fn, return_header, return_pandas)

## ==== MOSAIC suite file format
def read_mosaic(fn, framerate, pixelsize):
    return read_arbitrary_csv(fn, col_traj="Trajectory", col_x="x", col_y="y", col_frame="Frame", framerate=framerate/1000., pixelsize=pixelsize/1000.)

## ==== TrackMate file format
def read_trackmate_csv(fn, framerate):
    """Do not call directly, wrapped into `read_trackmate`"""
    def cb(da):
        try:
            return da[da.TRACK_ID!="None"]
        except:
            return da[da.TRACK_ID!=np.nan]
    return read_arbitrary_csv(fn, col_traj="TRACK_ID", col_x="POSITION_X", col_y="POSITION_Y", col_frame="FRAME", framerate=framerate/1000., cb=cb)

def read_trackmate_xml(path, min_len=3):
    """Converts xml to [x, y, time, frame] table"""
    data = xmltodict.parse(open(path, 'r').read(), encoding='utf-8')
    # Checks
    spaceunit = data['Tracks']['@spaceUnits']
    if spaceunit not in ('micron', 'um', 'µm', 'Âµm'):
        raise IOError("Spatial unit not recognized: {}".format(spaceunit))
    if data['Tracks']['@timeUnits'] != 'ms':
        raise IOError("Time unit not recognized")
    
    # parameters
    framerate = float(data['Tracks']['@frameInterval'])/1000. # framerate in ms
    traces = []
    
    try:
        for i, particle in enumerate(data['Tracks']['particle']):
            track = [(float(d['@x']), float(d['@y']), float(d['@t'])*framerate, int(d['@t']), i) for d in particle['detection']]
            if len(track) >=min_len:
                traces.append(Track(array=np.array(track), columns=['x', 'y', 't', 'frame', 'track.id'], units=['um', 'um', 'sec', '', '']))
    except KeyError as e:
        print(f'problem with {path}')
        raise e
    return traces

def remove_edge_tracks(tracks):
    ''' 
    Removes the tracks toucihng the edge of the frame.
    Trackmate rally bugs at the edges, returning localizations
    stick to the pixels. These localizations produce spikes 
    in jump length distributions. With this function spikes are removed.
    '''
    x_min = min([t.x.min() for t in tracks])
    x_max = max([t.x.max() for t in tracks])
    y_min = min([t.y.min() for t in tracks])
    y_max = max([t.y.max() for t in tracks])
    return list(filter(lambda t: t.x.min() > x_min and t.x.max() < x_max and t.y.min() > y_min and t.y.max() < y_max , tracks))

## ==== CSV file format
def read_csv(fn):
    return read_arbitrary_csv(fn, col_traj="trajectory", col_x="x", col_y="y", col_frame="frame", col_t="t")

## ==== Anders' file format
def read_anders(fn, new_format=True):
    """The file format sent by Anders. I don't really know where it 
    comes from.
    new_format tells whether we should perform weird column manipulations
    to get it working again...

        traces_header = ('x','y','t','f')

    """
    
    def _new_format(cel):
        """Converts between the old and the new Matlab format. To do so, it 
        swaps columns 1 and 2 of the detections and transposes the matrices"""
        cell = cel.copy()
        for i in range(len(cell)):
            f = cell[i][2].copy()
            cell[i][2] = cell[i][1].T.copy()
            cell[i][1] = f.T
        return cell
    
    ## Sanity checks
    if not os.path.isfile(fn):
        raise IOError("File not found: {}".format(fn))

    try:
        mat = scipy.io.loadmat(fn)
        m=np.asarray(mat['trackedPar'])
    except:
        raise IOError("The file does not seem to be a .mat file ({})".format(fn))

    if new_format:
        m[0] = _new_format(m[0])
    
    ## Make the conversion
    traces = []
    for tr in m[0]:
        x = [float(i) for i in tr[0][:,0]]
        y = [float(i) for i in tr[0][:,1]]
        t = [float(i) for i in tr[1][0]]
        f = [int(i) for i in tr[2][0]]
        traces.append(zip(x,y,t,f))
    return traces

## ==== 4DN file format


## ==== Format for fastSPT
def to_fastSPT(f, from_json=False):
    """Converts [x, y, time, frame] table to [[[x1,y1], [x2,y2], ...], [t1, t2, ...], [f1, f2, ...]]] """

    if from_json:
        da = json.loads(f.read()) ## Load data
    else:
        da = f

    ## Create the object
    dt = np.dtype([('xy', 'O'), ('TimeStamp', 'O'), ('Frame', 'O')]) # dtype
    # DT = np.dtype('<f8', '<f8', 'uint16')
    trackedPar = []
    for i in da:
        xy = []
        TimeStamp = []
        Frame = []
        for p in i:
            xy.append([p[0],p[1]])
            TimeStamp.append(p[2])
            Frame.append(p[3])
        trackedPar.append((np.array(xy, dtype='<f8'),
                           np.array([TimeStamp], dtype='<f8'),
                           np.array([Frame], dtype='uint16')))
    return np.asarray(trackedPar, dtype=dt)

##
## ==== This are some helper functions
##

def traces_to_csv(traces):
    """Returns a CSV file with the format 
    trajectory,x,y,t,frame
    """
    print("WARNING: deprecated use of 'traces_to_csv' in fastspt.readers, use it in 'fastspt.writers.traces_to_csv instead'.")
    csv = "trajectory,x,y,t,frame\n"
    for (tr_n, tr) in enumerate(traces):
        for pt in tr:
            csv +="{},{},{},{},{}\n".format(tr_n, pt[0],pt[1],pt[2],pt[3])
    return csv

def read_arbitrary_csv(fn, col_x="", col_y="", col_frame="", col_t="t",
                       col_traj="", framerate=None, pixelsize=None, cb=None,
                       sep=",", header='infer'):
    """This function takes the file name of a CSV file as input and parses it to
    the list of list format required by Spot-On. This function is called by various
    CSV importers and it is advised not to call it directly."""
    
    da = pd.read_csv(fn, sep=sep, header=header) # Read file
    
    # Check that all the columns are present:
    cols = da.columns
    if (not (col_traj in cols and col_x in cols and col_y in cols and col_frame in cols)) or (not (col_t in cols) and framerate==None):
        raise IOError("Missing columns in the file, or wrong header")
        
    # Correct units if needed
    if framerate is not None:
        da[col_t]=da[col_frame]*framerate
    if pixelsize is not None:
        da[col_x] *= pixelsize
        da[col_y] *= pixelsize
        
    # Apply potential callback
    if cb is not None:
        da = cb(da)

    # Split by traj
    out = pandas_to_fastSPT(da, col_traj, col_x, col_y, col_t, col_frame)
    return out

def pandas_to_fastSPT(da, col_traj, col_x, col_y, col_t, col_frame):
    out = []
    for (_, t) in da.sort_values(col_traj).groupby(col_traj):
        tr = [(tt[1][col_x], tt[1][col_y], tt[1][col_t], int(tt[1][col_frame]))
              for tt in t.sort_values(col_frame).iterrows()]  # Order by trace, then by frame
        out.append(tr)
    return out
