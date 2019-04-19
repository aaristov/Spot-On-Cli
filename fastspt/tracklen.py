import matplotlib.pyplot as plt
import numpy as np

def get_track_lengths_dist(cell, plot=True):
    hist, bins = get_hist(cell)
    fit_result, popt = fit_exponent(hist, bins)
    if plot:
        plot_hist_fit(hist, bins, fit_result, popt)
    return True
    
def get_hist(cell):
    track_lengths = list(map(len, cell))
    print(f'{len(cell)} tracks, {sum(track_lengths)} localizations')
    bins = np.arange(min(track_lengths), 20)
    hist, bins_edges = np.histogram(track_lengths, bins=bins)
    return hist, bins[:-1]

def exponent(x, a, c, d):
    return a*np.exp(-x/c)+d
    
def fit_exponent(hist, bins):

    from scipy.optimize import curve_fit


    popt, pcov = curve_fit(exponent, bins, hist)
    a, c, d = list(map(lambda x: np.round(x, 2), popt))
    #print(a, c, d)
    fit_result = exponent(bins, *popt)
    return fit_result, popt

def plot_hist_fit(hist, bins, fit_result, popt):
    a, c, d = popt
    print(len(bins), len(hist))
    print(bins.shape, hist.shape)
    plt.bar(bins, hist, fill=None, label=f'Weighted mean = {np.average(bins, weights=hist):.1f}')
    plt.plot(bins, fit_result, label=f'{a:.2f}*np.exp(-x/{c:.2f})+{d:.2f}')
    plt.title('Track length distribution')
    plt.xlabel('track length')
    plt.ylabel('counts')
    plt.xlim(2.5,19)
    plt.legend()
    plt.show()