import matplotlib.pyplot as plt
import numpy as np
from fastspt import core
import logging

logger = logging.getLogger(__name__)


def get_track_lengths_dist(tracks: [core.Track], plot=True):
    """
    Fits exponential function to track length distribution
    Returns decay rate
    """
    hist, bins = get_hist(tracks)
    if plot:
        plot_hist(hist, bins)   
    try:
        fit_result, popt = fit_exponent(hist, bins)
        a, c = popt
        decay_rate = 1 / c
        logger.info(f"Fit result: {a:.0f} * e^(-{decay_rate:.2f}x)")
        if plot:
            plot_fit(bins, fit_result, popt)
            plt.show()
        return decay_rate
    except RuntimeError as e:
        logger.error(f"Fit Failed: {e.args}")
        
        return None


def get_hist(tracks:[core.Track]):
    '''
    Generates histogram and vector from lengths of tracks
    '''
    track_lengths = list(map(len, tracks))
    logger.info(f"{len(tracks)} tracks, {sum(track_lengths)} localizations")
    bins = np.arange(min(track_lengths), max(track_lengths))
    hist, _ = np.histogram(track_lengths, bins=bins)
    return hist, bins[:-1]


def exponent(x, a, c):
    return a * np.exp(-x / c)


def fit_exponent(hist, bins, fun=exponent, p0=(100, 10)):

    from scipy.optimize import curve_fit

    popt, _ = curve_fit(fun, bins, hist, p0)
    fit_result = fun(bins, *popt)
    return fit_result, popt


def plot_hist(hist, bins):
    plt.bar(bins, hist, fill=None, label=f"track length")
    plt.title("Track length distribution")
    plt.xlabel("track length")
    plt.ylabel("counts")

def plot_fit(bins, fit_result, popt):
    a, c = popt
    plt.plot(bins, fit_result, 'r-', label=f"${a:.0f} * e^{{-{1/c:.2f} x}}$")
    plt.legend()
    plt.show()
