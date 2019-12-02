import scipy.io
import os
import xmltodict
import numpy as np
import pandas as pd
from fastspt.core import Track


def get_exposure_ms_from_path(path, pattern="bleach_(.*?)ms_"):
    import re

    regx = re.compile(pattern)
    found = regx.findall(path)
    if found:
        return float(found[0])
    else:
        return None


assert get_exposure_ms_from_path("bleach_1.7ms_no_strobo") == 1.7
assert get_exposure_ms_from_path("bleach_60ms_strobo_10ms") == 60


def read_trackmate_xml(path, min_len=3):
    """Converts xml to [x, y, time, frame] table"""
    data = xmltodict.parse(open(path, "r").read(), encoding="utf-8")
    # Checks
    spaceunit = data["Tracks"]["@spaceUnits"]
    if spaceunit not in ("micron", "um", "µm", "Âµm"):
        raise IOError("Spatial unit not recognized: {}".format(spaceunit))
    if data["Tracks"]["@timeUnits"] != "ms":
        raise IOError("Time unit not recognized")

    # parameters
    framerate = float(data["Tracks"]["@frameInterval"]) / 1000.0
    traces = []

    try:
        for i, particle in enumerate(data["Tracks"]["particle"]):
            track = [
                (
                    float(d["@x"]),
                    float(d["@y"]),
                    float(d["@t"]) * framerate,
                    int(d["@t"]),
                    i,
                )
                for d in particle["detection"]
            ]
            if len(track) >= min_len:
                traces.append(
                    Track(
                        array=np.array(track),
                        columns=["x", "y", "t", "frame", "track.id"],
                        units=["um", "um", "sec", "", ""],
                    )
                )
    except KeyError as e:
        print(f"problem with {path}")
        raise e
    return traces


def remove_edge_tracks(tracks):
    """
    Removes the tracks toucihng the edge of the frame.
    Trackmate rally bugs at the edges, returning localizations
    stick to the pixels. These localizations produce spikes
    in jump length distributions. With this function spikes are removed.
    """
    x_min = min([t.x.min() for t in tracks])
    x_max = max([t.x.max() for t in tracks])
    y_min = min([t.y.min() for t in tracks])
    y_max = max([t.y.max() for t in tracks])
    return list(
        filter(
            lambda t: t.x.min() > x_min
            and t.x.max() < x_max
            and t.y.min() > y_min
            and t.y.max() < y_max,
            tracks,
        )
    )


def read_csv(fn):
    return read_arbitrary_csv(
        fn, col_traj="trajectory", col_x="x", col_y="y", col_frame="frame", col_t="t"
    )


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

    # Sanity checks
    if not os.path.isfile(fn):
        raise IOError("File not found: {}".format(fn))

    try:
        mat = scipy.io.loadmat(fn)
        m = np.asarray(mat["trackedPar"])
    except IOError:
        raise IOError("The file does not seem to be a .mat file ({})".format(fn))

    if new_format:
        m[0] = _new_format(m[0])

    # Make the conversion
    traces = []
    for tr in m[0]:
        x = [float(i) for i in tr[0][:, 0]]
        y = [float(i) for i in tr[0][:, 1]]
        t = [float(i) for i in tr[1][0]]
        f = [int(i) for i in tr[2][0]]
        traces.append(zip(x, y, t, f))
    return traces


def read_arbitrary_csv(
    fn,
    col_x="",
    col_y="",
    col_frame="",
    col_t="t",
    col_traj="",
    framerate=None,
    pixelsize=None,
    cb=None,
    sep=",",
    header="infer",
):
    """This function takes the file name of a CSV file as input and parses it to
    the list of list format required by Spot-On.
    This function is called by various
    CSV importers and it is advised not to call it directly."""

    da = pd.read_csv(fn, sep=sep, header=header)  # Read file

    # Check that all the columns are present:
    cols = da.columns
    if (
        not (col_traj in cols and col_x in cols and col_y in cols and col_frame in cols)
    ) or (not (col_t in cols) and framerate is None):
        raise IOError("Missing columns in the file, or wrong header")

    # Correct units if needed
    if framerate is not None:
        da[col_t] = da[col_frame] * framerate
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
        tr = [
            (tt[1][col_x], tt[1][col_y], tt[1][col_t], int(tt[1][col_frame]))
            for tt in t.sort_values(col_frame).iterrows()
        ]
        # Order by trace, then by frame
        out.append(tr)
    return out
