import numpy as np
from fastspt import core
from scipy.ndimage import gaussian_filter1d as gf1
from functools import reduce
from itertools import zip_longest
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

matplotlib.logging.getLogger().setLevel(logging.INFO)


class SingleProb:
    def __init__(self, D, F, dt, sigma):
        self.D = D
        self.F = F
        self.dt = dt
        self.sigma = sigma

    def __call__(self, r, lag=1):
        return self.F * p_jd(self.dt * lag, self.sigma, self.D)(r)

    def __repr__(self):
        return f"Single prob: \t D: {self.D:.3f} \
            \t F: {self.F:.0%} \t dt: {self.dt}, \
            \t sigma: {self.sigma:.3f}"


class BayesFilter:
    def __init__(self, D=[0, 0.2], F=[0.5, 0.5], dt=0.06, sigma=[0.02], **kwargs):
        """
        Helper class to classify states of segments of a track using
        bayesian inference
        Inititalize bayes probabilities for given kinetic constants.
        D and F must be equal length.
        Sigma vector can be shorter than number of states. In this case,
        later states are using sigma[0] if len(sigma > 1)

        Parameters:
        -----------
        D : list
            Diffusion constants
        F : list
            Fractions
        dt : float
            time interval of acquisition
        sigma: list
            localization errors for different states
        **kwargs: dict
            ignored
        Example:
        --------
        >>> track = simulate.track(D_free=0.1,
        loc_error=0.02, p_binding=1e-2, p_unbinding=3e-2,
        p_out_of_focus=1e-5, p_bleaching=0.05)

        >>> bayes_filter = BayesFilter(**{'D': [0, .1], 'F': [.2, .8],
        'sigma': [0.02], 'dt': 0.06})

        >>> prediction = bayes_filter.predict_states(
            track,
            max_lag=4,
            smooth=0
        )

        >>> plot.plot_track_multistates(track, cols=['free'],
        states={'free': [1], 'true bound': [0]}, lim=2)
        >>> plt.show()
        >>> plt.plot(prediction, '.-', label='prediction')
        >>> plt.plot(track.free, '.-', label='true state')
        >>> plt.ylim(-0.1, 1.1)
        >>> plt.legend()
        >>> plt.show()

        # inititalizing three states with second state having
        # confinement with sigm 0.03
            probs = BayesFilter(D=[0, 0.01, 0.07], F=[0.16, 0.40, 0.44],
            sigma=[0.017, 0.03])
            _ = probs.plot_bayes(r)

        """
        assert len(D) == len(F) > 1
        assert sum(F) == 1

        self.probs = []
        self.D = D
        self.F = F
        self.dt = dt
        self.sigma = sigma
        for d, f, s in zip_longest(D, F, sigma, fillvalue=sigma[0]):
            p = SingleProb(d, f, dt, s)
            self.probs.append(p)
        self.__repr__()

    def _sum_prob(self, r, lag=1):
        return sum([p(r, lag) for p in self.probs])

    def __getitem__(self, value):
        return self.probs[value]

    def __call__(self, r: list, lag: int = 1):
        """
        Computes probabilities for vector `r` of jumping distances.
        First dimension: state, second: distances.
        """
        assert isinstance(
            r, (list, np.ndarray, tuple)
        ), f"`r` must be vector, \
            got {type(r)}"
        assert isinstance(
            lag, int
        ), f"`lag` must be positive integer, \
            not {lag}"
        assert lag > 0, f"`lag` must be positive integer, not {lag}"

        r = np.array(r)
        return [p(r, lag) / self._sum_prob(r, lag) for p in self.probs]

    def __repr__(self):
        return (
            f"Bayes prob: {self.n_states} states"
            + "\n"
            + "\n".join([f"{i} state: {p}" for i, p in enumerate(self.probs)])
        )

    @property
    def n_states(self):
        return len(self.probs)

    __len__ = n_states

    def predict_states(self, track: core.Track, max_lag=4, smooth: float = 0):
        """
        Bayesian filter for segments of the track. Depending on jumping
        distances for lag = 1..max_lag, assigns state index to each
        localization.

        Parameters:
        -----------
        track: core.Track
            track to analyze
        max_lag: int
            how many jumps to consider for every time point
        smooth: float
            if not zero, applies 1d gaussian filter to the output and casts
            back to integers. Allows to minimize jiggling

        Return:
        -------
        states: list
            vector of the same size as track length. Possible values span in
            range(n_states).

        """
        lags = range(1, min(max_lag + 1, len(track)))

        jds = [get_jd(track.xy, lag=l, extrapolate=1) for l in lags]

        probs_lag_states_jd = np.array(
            [self.__call__(jd, lag) for jd, lag in zip(jds, lags)]
        )

        probs_states_jd = probs_lag_states_jd.mean(axis=0)
        assert probs_states_jd.shape[0] == self.n_states

        current_state_jd = np.argmax(probs_states_jd, axis=0)
        assert len(current_state_jd) == len(track)

        if smooth:
            current_state_jd = np.round(gf1(current_state_jd, smooth), 0).astype(int)

        return current_state_jd

    def add_states_single(
        self, track: core.Track, col="states", comments="", max_lag=4, smooth: float = 0
    ) -> core.Track:
        """
        Computes the states for single track and stores them
        into `col` with [`comments`]
        """
        states = self.predict_states(track, max_lag, smooth)
        new_track = track.add_column(col, states, comments)
        return new_track

    def add_states_many(
        self,
        tracks: [core.Track],
        col="states",
        comments="",
        max_lag=4,
        smooth: float = 0,
    ) -> [core.Track]:

        """
        Computes the states for list of tracks and stores them
        into `col` with [`comments`]
        """
        tracks_with_states = [
            self.add_states_single(t, col, comments, max_lag, smooth)
            for t in tqdm(tracks)
        ]
        return tracks_with_states

    def plot_jd(self, x, lag=1):
        """
        Shows probability density functions for jump lengths
        `x` and time lag `lag`.

        Parameters:
        -----------
        x: list or 1d array
            Vector of jumping distances.
        lag: int
            Time lag for computation.
        Return:
        -------
        fig: matplotlib.pyplot.figure
            Figure object


        Example:
        --------
        r = np.arange(0.001, 0.3, 0.001)
        bayes_filter = BayesProb(D=[0, 0.2], F=[0.5, 0.5],
        dt=0.06, sigma=[0.02])
        bayes_filter.plot_jd(r)
        """
        fig = plt.figure()
        plt.title(f"sigma: {self.sigma}, Δt: {self.dt}")

        [
            plt.plot(
                x,
                p(x, lag) / self._sum_prob(x, lag),
                label=f"D: {p.D} ({p.F:.0%}), {lag} * Δt",
            )
            for p in self.probs
        ]
        plt.legend(loc=(1, 0))

        return fig

    def plot_bayes(self, x, lag=1, fig=True):
        """
        Shows bayesian probabilities normalized by total probability
        as function of jump lengths `x` and time lag `lag`

        Parameters:
        -----------
        x: list or 1d array
            Vector of jumping distances.
        lag: int
            Time lag for computation.
        Return:
        -------
        fig: matplotlib.pyplot.figure
            Figure object

        Example:
        --------
        r = np.arange(0.001, 0.3, 0.001)
        bayes_filter = BayesProb(D=[0, 0.2], F=[0.5, 0.5],
        dt=0.06, sigma=[0.02])
        bayes_filter.plot_bayes(r)
        """
        if fig:
            fig = plt.figure()
        plt.title(f"sigma: {self.sigma}, Δt: {self.dt}")
        [
            plt.plot(
                x,
                p(x, lag) / self._sum_prob(x, lag),
                label=f"D: {p.D} ({p.F:.0%}), {lag} * Δt",
            )
            for p in self.probs
        ]
        plt.legend(loc=(1, 0))
        plt.xlabel("jump distance, μm")
        plt.ylabel("probability")
        if fig:
            return fig


def d_(time, sigma, D):
    return D * time + sigma ** 2


def p_jd(time, sigma, D):
    assert time > 0
    assert sigma > 0
    assert D >= 0
    d = d_(time, sigma, D)
    return lambda r: r / (2 * d) * np.exp(-(r ** 2) / (4 * d))


def get_jd(xy: np.array, lag=1, extrapolate=False, filter_frame_intevals=None):

    # TODO: send frame jumps to higher lags

    """
    Computes jumping distances out of xy coordinates array.
    Parameters:
    -----------
    xy = [[x1, y1]
        [x2, y2]
            ---  ]

    lag: int
        jump interval in frames, default 1

    extrapolate: bool
        If True, returns the same size by filling edges with the same values.
        Default False.

    filter_frame_intevals: list
        if not None, will use only frame intervals equal to lag, default None

    """

    if len(xy) <= lag:
        return []

    dxy = xy[lag:] - xy[:-lag]

    if filter_frame_intevals is not None:
        frames = np.ravel(filter_frame_intevals)
        d_frames = frames[lag:] - frames[:-lag]
        dxy = dxy[d_frames == lag]

    jd = np.sqrt((dxy ** 2).sum(axis=1))

    if extrapolate:
        while len(jd) < len(xy):
            jd = np.concatenate(([jd[0]], jd[:]))
            #                 even = lag % 2 == 0
            if len(jd) < len(xy):
                jd = np.concatenate((jd[:], [jd[-1]]))

    return jd


def sum_list(l):
    return reduce(lambda a, b: a + b, l)


def get_switching_rates(xytfu: [core.Track], fps: float, column: str = "free") -> dict:
    """
    Parameters:
    -----------
    xytfu: list of core.Track objects
    fps: float
        Framerate
    lag: int
        if lag is more than 1, halpf of this length will be cut off the ends
        of the track to avoid artefacts
    column: str
        where to look for the label
    Return:
    -------
    stats: dict
        {'F_bound': n_bound_spots / n_total_spots,
        'u_rate_frame': u_rate_frame, 'b_rate_frame': b_rate_frame }
    """

    n_bound_spots = sum_list(map(lambda a: sum(a.col(column)[:] == 0), xytfu))
    n_bound_spots_for_rates = sum_list(
        map(lambda a: sum(a.col(column)[:-1] == 0), xytfu)
    )
    n_unbound_spots_for_rates = sum_list(
        map(lambda a: sum(a.col(column)[:-1] == 1), xytfu)
    )
    # print(n_bound_spots, n_bound_spots_for_rates)
    n_total_spots = sum_list(map(lambda a: len(a), xytfu))

    #     n_total_segments = n_total_spots - len(xytfu)
    #     n_bound_segments = n_bound_spots - len(bound)

    print(
        f"bound fraction based on number of spots: {n_bound_spots} / \
        {n_total_spots} = {n_bound_spots / n_total_spots:.1%}"
    )

    def get_n_switch_unbind(xytfu):
        return sum_list(
            map(lambda a: sum(a.col(column)[1:] - a.col(column)[:-1] == 1), xytfu)
        )

    def get_n_switch_bind(xytfu):
        return sum_list(
            map(lambda a: sum(a.col(column)[1:] - a.col(column)[:-1] == -1), xytfu)
        )

    n_switch_unbind = get_n_switch_unbind(xytfu)
    n_switch_bind = get_n_switch_bind(xytfu)

    print(f"{n_switch_bind} binding events, {n_switch_unbind} unbinding events")
    u_rate_frame = n_switch_unbind / n_bound_spots_for_rates
    b_rate_frame = n_switch_bind / n_unbound_spots_for_rates
    print(
        f"Unbinding switching rates: {u_rate_frame:.1%} per frame, \
            {u_rate_frame * fps:.1%} per second {fps} fps"
    )
    print(
        f"Binding switching rates: {b_rate_frame:.1%} per frame, \
            {b_rate_frame * fps:.1%} per second {fps} fps"
    )
    print(
        f"Bound fraction based on switching rates: \
        {b_rate_frame / (b_rate_frame + u_rate_frame): 0.1%}"
    )

    return {
        "F_bound": n_bound_spots / n_total_spots,
        "u_rate_frame": u_rate_frame,
        "b_rate_frame": b_rate_frame,
        "F_bound_from_rates": b_rate_frame / (u_rate_frame + b_rate_frame),
    }
