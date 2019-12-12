import matplotlib.pyplot as plt
import numpy as np
from fastspt import core


def get_track_lengths_dist(tracks: [core.Track], plot=True):
    """
    Fits exponential function to track length distribution
    Returns decay rate
    """
    hist, bins = get_hist(tracks)
    try:
        fit_result, popt = fit_exponent(hist, bins)
        a, c = popt
        decay_rate = 1 / c
        print(f"Fit result: {a:.0f} * e^(-{decay_rate:.2f}x)")
        if plot:
            plot_hist_fit(hist, bins, fit_result, popt)
        return decay_rate
    except RuntimeError as e:
        print("Fit Failed", e)
        if plot:
            plt.hist(tracks)
            plt.show()
        return None


def get_hist(cell):
    track_lengths = list(map(len, cell))
    print(f"{len(cell)} tracks, {sum(track_lengths)} localizations")
    bins = np.arange(min(track_lengths), max(track_lengths))
    hist, _ = np.histogram(track_lengths, bins=bins)
    return hist, bins[:-1]


def exponent(x, a, c):
    return a * np.exp(-x / c)


def fit_exponent(hist, bins, fun=exponent, p0=None):

    from scipy.optimize import curve_fit

    popt, _ = curve_fit(fun, bins, hist, p0)
    fit_result = fun(bins, *popt)
    return fit_result, popt


def plot_hist_fit(hist, bins, fit_result, popt):
    a, c = popt
    plt.bar(bins, hist, fill=None, label=f"track length")
    plt.plot(bins, fit_result, label=f"${a:.0f} * e^{{-{1/c:.2f} x}}$")
    plt.title("Track length distribution")
    plt.xlabel("track length")
    plt.ylabel("counts")
    plt.legend()
    plt.show()
