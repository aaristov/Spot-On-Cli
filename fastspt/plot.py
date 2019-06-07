# A packaged version of the fastSPT code by Anders Sejr Hansen, Feb. 2016
# Python rewriting by MW, March 2017
#
# In this module we put all the plotting functions
# 
# History: For the history of the script see the related CHANGELOG file.

## ==== Imports
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from fastspt import fit, plot
import lmfit

def plot_kinetics_fit(jump_hist,
                      fit_result:lmfit.model.ModelResult, 
                      CDF1=True,
                      states=2, 
                      **fit_params) -> bool:
    
    h1=jump_hist

    if CDF1:
        HistVecJumps = h1[2]
        JumpProb = h1[3]
        HistVecJumpsCDF = h1[0]
        JumpProbCDF = h1[1]
    else:
        HistVecJumps = h1[0]
        JumpProb = h1[1]
        HistVecJumpsCDF = h1[0]
        JumpProbCDF = h1[1]

    fit2states_dict = {2 : True, 3 : False}
    fit2states = fit2states_dict[states]

    ## Generate the PDF corresponding to the fitted parameters
    y = fit.generate_jump_length_distribution(fit_results=fit_result.params, 
                                              JumpProb = JumpProbCDF, 
                                              HistVecJumps=HistVecJumpsCDF,
                                              fit2states=fit2states,
                                              norm=True, 
                                              **fit_params
                                              )
    ## Normalization does not work for PDF yet (see commented line in fastspt.py)
    if CDF1:
        y = y * float(len(HistVecJumpsCDF))/float(len(HistVecJumps))
    #plt.figure(figsize=(18,8)) # Initialize the plot
    plot.plot_histogram(HistVecJumps, JumpProb, HistVecJumpsCDF, y, ) ## Read the documentation of this function to learn how to populate all the 'na' fields
    return True


def plot_histogram(HistVecJumps, emp_hist, HistVecJumpsCDF=None, sim_hist=None,
                   TimeGap=None, SampleName=None, CellNumb=None,
                   len_trackedPar=None, Min3Traj=None, CellLocs=None,
                   CellFrames=None, CellJumps=None, ModelFit=None,
                   D_free=None, D_bound=None, F_bound=None, figsize=(18,8) ):
    """Function that plots an empirical histogram of jump lengths,
    with an optional overlay of simulated/theoretical histogram of 
    jump lengths"""

    ## Parameter parsing for text labels
    if CellLocs != None and CellFrames != None:
        locs_per_frame = round(CellLocs/CellFrames*1000)/1000
    else:
        locs_per_frame = 'na'    
    if SampleName == None:
        SampleName = 'na'
    if CellNumb == None:
        CellNumb = 'na'
    if len_trackedPar == None:
        len_trackedPar = 'na'
    if Min3Traj == None:
        Min3Traj = 'na'
    if CellLocs == None:
        CellLocs = 'na'
    if CellFrames == None:
        CellFrames = 'na'
    if CellJumps == None:
        CellJumps = 'na'
    if ModelFit == None:
        ModelFit = 'na'
    if D_free == None:
        D_free = 'na'
    if D_bound == None:
        D_bound = 'na'
    if F_bound == None:
        F_bound = 'na'

    ## Do something
    JumpProb = emp_hist
    scaled_y = sim_hist
    
    histogram_spacer = 0.055
    number = JumpProb.shape[0]
    cmap = plt.get_cmap('viridis')
    colour = [cmap(i) for i in np.linspace(0, 1, number)]

    plt.figure(figsize=figsize)
    
    for i in range(JumpProb.shape[0]-1, -1, -1):
        new_level = (i)*histogram_spacer
        colour_element = colour[i] #colour[round(i/size(JumpProb,1)*size(colour,1)),:]
        plt.plot(HistVecJumps, (new_level)*np.ones(HistVecJumps.shape[0]), 'k-', linewidth=1)
        for j in range(1, JumpProb.shape[1]): ## Looks like we are manually building an histogram. Why so?
            x1 = HistVecJumps[j-1]
            x2 = HistVecJumps[j]
            y1 = new_level
            y2 = JumpProb[i,j-1]+new_level
            plt.fill([x1, x1, x2, x2], [y1, y2, y2, y1], color=colour_element) # /!\ TODO MW: Should use different colours
        if type(sim_hist) != type(None): ## HistVecJumpsCDF should also be provided
            plt.plot(HistVecJumpsCDF, scaled_y[i,:]+new_level, 'k-', linewidth=2)
        if TimeGap != None:
            plt.text(0.6*max(HistVecJumps), new_level+0.3*histogram_spacer, '$\Delta t$ : {} ms'.format(TimeGap*(i+1)))
        else:
            plt.text(0.6*max(HistVecJumps), new_level+0.3*histogram_spacer, '${} \Delta t$'.format(i+1))

    plt.xlim(0,HistVecJumps.max())
    plt.ylabel('Probability')
    plt.xlabel('jump length ($\mu m$)')
    if type(sim_hist) != type(None):
        plt.title('{}; Cell number {}; Fit Type = {}; Dfree = {}; Dbound = {}; FracBound = {}, Total trajectories: {}; => Length 3 trajectories: {}, \nLocs = {}, Locs/Frame = {}; jumps: {}'
          .format(
              SampleName, CellNumb, ModelFit,
              D_free, D_bound, F_bound,
              len_trackedPar, Min3Traj, CellLocs,
              locs_per_frame,
              CellJumps))
    else:
        plt.title('{}; Cell number {}; Total trajectories: {}; => Length 3 trajectories: {}, \nLocs = {}, Locs/Frame = {}; jumps: {}'
          .format(
              SampleName, CellNumb,
              len_trackedPar, Min3Traj, CellLocs,
              locs_per_frame,
              CellJumps))
    plt.yticks([])
    plt.show()

    return True
