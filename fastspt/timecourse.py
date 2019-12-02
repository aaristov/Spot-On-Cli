import os
import datetime
from glob import glob


def get_exposure_ms_from_path(path, pattern=r'ch_(\d*?)ms'):
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


assert get_exposure_ms_from_path('prebleach_30ms_strobo_15ms_1_') == 30


def get_modification_date(filename):
    tif_path = get_tif_path_from(filename)
    t = os.path.getmtime(tif_path)
    return datetime.datetime.fromtimestamp(t)


def get_tif_path_from(xml_path, extension='.ome.tif'):
    folder = os.path.dirname(xml_path)
    # print(folder)
    flist = glob(folder + os.path.sep + '*' + extension)
    try:
        tif_path = flist[0]
    except IndexError:
        print(f'index error folder {folder}')
        print(f'flist: {flist}')
        return xml_path
    # print(tif_path)
    return tif_path


def get_interval_minutes(*time_stamp, start_time: str = "00:00"):
    h, m = list(map(int, start_time.split(':')))
    t0 = time_stamp[0].replace(hour=h, minute=m)
    time_interval_mins = [
        float((t - t0).total_seconds()) / 60. for t in time_stamp]
    return time_interval_mins
