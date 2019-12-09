import os
import datetime
from glob import glob
import json
import logging

logger = logging.getLogger(__name__)


def get_exposure_ms_from_path(path, pattern=r"ch_(\d*?)ms"):
    import re

    regx = re.compile(pattern)
    found = regx.findall(path)
    if found:
        try:
            return float(found[0])
        except ValueError as e:
            print(e)
    else:
        return None


assert get_exposure_ms_from_path("prebleach_30ms_strobo_15ms_1_") == 30


def get_timestamp(path:str) -> float:
    return os.path.getmtime(path)


def get_modification_date(filename: str) -> datetime.datetime:
    tif_path = get_tif_path_from(filename)
    timestamp = get_timestamp(tif_path)
    return datetime.datetime.fromtimestamp(timestamp)


def get_tif_path_from(xml_path, extension=".ome.tif"):
    '''
    Finds source file path with extension in the same folder as xml_path
    '''

    folder = os.path.dirname(xml_path)
    flist = glob(folder + os.path.sep + "*" + extension)
    try:
        tif_path = flist[0]
    except IndexError:
        logger.error(f"index error folder {folder}")
        logger.error(f"flist: {flist}")
        return xml_path
    return tif_path


def get_interval_minutes(
    *time_stamp: datetime.datetime, start_time: str = "00:00"
) -> float:
    h, m = list(map(int, start_time.split(":")))
    t0 = time_stamp[0].replace(hour=h, minute=m)
    time_interval_mins = [float((t - t0).total_seconds()) / 60.0 for t in time_stamp]
    return time_interval_mins


def save_timestamps_to_json(tif_paths:[str], fname="timestamps.json"):
    '''
    Finds timestamps of tif file paths form the list.
    Finds common path.
    Creates `fname`.
    Saves json with tif_path: timestamp
    '''
    out = {os.path.basename(p): get_timestamp(p) for p in tif_paths}
    prefix = os.path.commonpath(tif_paths)
    fname = os.path.join(prefix, fname)
    with open(fname, "w") as f:
        json.dump(out, f)
    logger.info(f"Saved {fname}")
    return fname
