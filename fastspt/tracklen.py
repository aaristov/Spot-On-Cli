import matplotlib.pyplot as plt
import numpy as np


def get_track_lengths_dist(tracks, plot=True, max_len=20):
    """
    Generates histogram of lengths and fits exponent with offset.

    hist (x) = a * exp(-x/c) + d
    """
    assert len(tracks) > 0
    assert len(tracks[0]) > 0

    hist, bins = get_hist(tracks, max_len=max_len)
    try:
        fit_result, popt = fit_exponent(hist, bins)
        a, c, d = popt
        print(f"Fit result: {a:.2f} * e^(-x/{c:.2f}) + {d:.2f}")
        if plot:
            plot_hist_fit(hist, bins, fit_result, popt)
        return {"decay rate": 1 / c}
    except RuntimeError as e:
        print("Fit Failed", e)
        if plot:
            plt.hist(tracks)
            plt.show()
        return False


def get_hist(cell, max_len):
    track_lengths = list(map(len, cell))
    print(f"{len(cell)} tracks, {sum(track_lengths)} localizations")
    bins = np.arange(min(track_lengths), max_len)
    hist, _ = np.histogram(track_lengths, bins=bins)
    return hist, bins[:-1]


def exponent(x, a, c, d):
    return a * np.exp(-x / c) + d


def fit_exponent(hist, bins, fun=exponent, p0=None):

    from scipy.optimize import curve_fit

    popt, _ = curve_fit(fun, bins, hist, p0)
    fit_result = fun(bins, *popt)
    return fit_result, popt


def plot_hist_fit(hist, bins, fit_result, popt):
    a, c, d = popt
    plt.bar(
        bins,
        hist,
        fill=None,
        label=f"Weighted mean = {np.average(bins, weights=hist):.1f}",
    )
    plt.plot(bins, fit_result, label=f"{a:.2f}*np.exp(-x/{c:.2f})+{d:.2f}")
    plt.title("Track length distribution")
    plt.xlabel("track length")
    plt.ylabel("counts")
    plt.legend()
    plt.show()
