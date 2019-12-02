import matplotlib.pyplot as plt
import numpy as np
from fastspt import bind, bayes, core


def threshold_sqr_displacement(sd, thr=0.005):
    """
    Returns 0 if below threshold, 1 otherwise
    """
    return np.where(sd < thr, 0, 1)


def select_states(
    track, cols=["s0", "s1"], states={"free": [1, 1], "med": [0, 1], "bound": [0, 0]},
):
    segs = {}
    for label, values in states.items():
        sel = [track.col(col) == v for col, v in zip(cols, values)]
        if len(sel) > 1:
            sel = np.logical_and(*sel)
        segs[label] = track[sel]
    return segs


def plot_track_multistates(
    track: core.Track,
    cols=["s0", "s1"],
    states={"free": [1, 1], "med": [1, 0], "bound": [0, 0]},
    exclude="free",
    msd=False,
    lim=0.5,
    jd_lim=0.3,
    n_lags=3,
    title="",
):

    segs = select_states(track, cols=cols, states=states)

    #     labels = states.keys()

    if msd:
        base = 130
        figsize = (15, 4)
    else:
        base = 120
        figsize = (10, 4)

    fig = plt.figure(figsize=figsize)

    fig.add_subplot(base + 1)
    plt.plot(track.x, track.y, ".-", label="trajectory xy")

    plt.plot(track.x[0], track.y[0], "o", label="start")
    plt.plot(track.x[-1], track.y[-1], "o", label="end")

    for label, seg in segs.items():
        if len(seg) and label is not exclude:
            plt.plot(seg[:, 0], seg[:, 1], ".", label=label)

    plt.xlim(track[:, 0].mean() - lim, track[:, 0].mean() + lim)
    plt.ylim(track[:, 1].mean() - lim, track[:, 1].mean() + lim)
    # plt.axis('equal')

    plt.grid()
    plt.legend()
    plt.title(title)

    fig.add_subplot(base + 2)

    def get_jds(track):
        return [bayes.get_jd(track.xy, lag=l + 1, extrapolate=1) for l in range(n_lags)]

    jds = get_jds(track)

    # frames = np.arange(len(jds[0]))
    try:
        [
            plt.plot(track.frame, jd, label=f"jump length {i + 1} Δt", alpha=0.5)
            for i, jd in enumerate(jds)
        ]

        for label, seg in segs.items():
            if len(seg) and label is not exclude:
                plt.plot(seg[:, 3], np.zeros(len(seg)), "o", label=label, alpha=0.5)
    except ValueError:
        pass

    try:
        plt.plot(track.frame, track.col("uncertainty_xy [nm]") * 3, label="3 * sigma")
    except Exception:
        pass

    try:
        swift_id = track.col("seg.id")
        plt.plot(track.frame, (swift_id - min(swift_id)) * jd_lim, label="swift id")
    except Exception:
        pass

    plt.ylim(-0.01, jd_lim * 1.05)
    plt.legend(loc=(1, 0.5))

    if msd:
        fig.add_subplot(base + 3)

        msd = [
            np.mean(bayes.get_jd(track.xy, lag=l + 1) ** 2)
            for l in range(len(track) - 2)
        ]
        plt.plot(msd)
        plt.xlabel("frame lag")
        plt.title("MSD")

    plt.tight_layout()


def plot_track_xy_sd(
    track_xytf, figsize=(8, 4), xy_radius=0.3, sd_max=0.2, bound_sd=0.01, title=None
):

    sd = bind.get_sqr_displacement(track_xytf)
    track = np.array(track_xytf)
    xy = track[:, :2]
    x_center, y_center = xy.mean(axis=0)
    dt = track[:-1, 3]

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(121)
    plt.title(f"track {title} xy")
    plt.plot(xy[:, 0], xy[:, 1], ".-")
    plt.xlabel("x, um")
    plt.ylabel("y, um")
    plt.axis("square")
    plt.xlim(x_center - xy_radius, x_center + xy_radius)
    plt.ylim(y_center - xy_radius, y_center + xy_radius)
    fig.add_subplot(122)
    plt.plot(dt, sd, ".-", label="free")

    if bound_sd:
        bound = np.where(sd < bound_sd)
        plt.plot(dt[bound], sd[bound], "ro", label="bound")
    plt.title(f"track {title} sqr displacement")
    plt.xlabel("frame")
    plt.ylabel("sqr displacement")
    plt.ylim(0, sd_max)
    plt.tight_layout()
    plt.legend()
    plt.show()

    return True
