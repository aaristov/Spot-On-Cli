from fastspt import core
import numpy as np
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


def track(
    track_id=0,
    start_time=0,
    dt=0.06,
    D_bound=0,
    D_free=0.06,
    loc_error=0.02,
    p_binding=1e-4,
    p_unbinding=1e-3,
    p_bleaching=1e-1,
    p_out_of_focus=1e-2,
    min_len=5,
):
    """
    Uses physical parameters to get timepoints xy diffusing or bound in space.
    All localizations are equally affected by localization error Sigma.
    Parameters:
    -----------

    Return:
    -------
    np.array with each line containing [x, y, time, sigma, bound?, id]
    """

    bound_fraction = p_binding / (p_binding + p_unbinding)

    bound = np.random.random() < bound_fraction
    stop = False
    steps = 0
    dxytsbi = []  # time step for x, y, time, sigma, bound?, id

    while not stop:

        if bound:
            stop = np.random.random() < p_bleaching and steps >= min_len
            D_ = D_bound
            switch = np.random.random() < p_unbinding
        else:
            stop = (
                np.random.rand(1)[0] < p_bleaching + p_out_of_focus and steps >= min_len
            )
            D_ = np.sqrt(2 * D_free * dt)
            switch = np.random.random() < p_binding

        if switch:
            bound = not bound
        dxy = np.random.standard_normal(size=(2,)) * D_

        dxytsbi.append(
            [
                dxy[0],
                dxy[1],
                steps * dt + start_time,
                steps,
                0,
                int(not bound),
                track_id,
            ]
        )
        steps += 1

    out = np.array(dxytsbi)
    out[:, :2] = np.cumsum(out[:, :2], axis=0)
    sigma = 0.001 * np.random.standard_gamma(loc_error * 1000.0, size=(len(out), 2))
    out[:, :2] = out[:, :2] + sigma * np.random.standard_normal(size=(len(out), 2))
    out[:, 4] = sigma.mean(axis=1)

    track = core.Track(
        out,
        columns=["x", "y", "t", "frame", "sigma", "free", "id"],
        units=["um", "um", "sec", "", "um", "", ""],
    )

    return track


def tracks(
    num_tracks=1e3,
    dt=0.06,
    D_bound=0,
    D_free=0.06,
    loc_error=0.02,
    p_binding=1e-4,
    p_unbinding=1e-3,
    p_bleaching=1e-1,
    p_out_of_focus=1e-5,
    min_len=5,
    fun=track,
    use_tqdm=True,
    **kwargs,
):
    logger.info(f"Simulating {num_tracks} tracks")
    tracks = list(
        map(
            lambda id: fun(
                track_id=id,
                start_time=id,
                dt=dt,
                D_bound=D_bound,
                D_free=D_free,
                loc_error=loc_error,
                p_binding=p_binding,
                p_unbinding=p_unbinding,
                p_bleaching=p_bleaching,
                p_out_of_focus=p_out_of_focus,
                min_len=min_len,
            ),
            tqdm(range(int(num_tracks)), disable=not use_tqdm),
        )
    )
    return tracks
