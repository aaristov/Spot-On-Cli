## A simple list of datasets
## See specific license for the datasets
## Unless specified, datasets acquired by Anders Sejr Hansen

import os

def list_path(path, string=True):
    """ A simple function that return the list of datasets """
    d1 = {"description": "20160216_C87_Halo-mCTCF_5nM_PA-JF549_FastTracking: 5 ms camera, 1 ms 561 nm, 6.75 ms per frame.",
        "path": '/Users/AndersSejrHansen/Dropbox/MatLab/Lab/Microscopy/SingleParticleTracking/Analysis/FastTrackingData/20160216_C87_Halo-mCTCF_5nM_PA-JF549_FastTracking/',
        "workspaces": ['20160216_5ms_PA-JF549_Pulse1_L100_Constant405-1.mat',
                       '20160216_5ms_PA-JF549_Pulse1_L100_Constant405-2.mat'],
        "SampleName": 'SampleName'}

    d2 = {"description": "20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646",
        'path': './20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646/',
        'workspaces': ['20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell1.rpt_tracked_TrackedParticles.mat',              '20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell2.rpt_tracked_TrackedParticles.mat', '20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell3.rpt_tracked_TrackedParticles.mat', '20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell4.rpt_tracked_TrackedParticles.mat', '20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell5.rpt_tracked_TrackedParticles.mat', '20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell6.rpt_tracked_TrackedParticles.mat', '20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell7.rpt_tracked_TrackedParticles.mat', '20160526_mESC_C87_Halo-mCTCF_25nM_PA-JF646_1ms633_3-405_4msCam_cell8.rpt_tracked_TrackedParticles.mat'],
        'SampleName': '20160526 mESC C87 Halo-mCTCF 25 nM PA-JF646',
        'Include': [1,1,1,1,1,1,1,1]}

    da = [d1, d2]

    ## Checking if the datasets are present
    ok = []
    for d in da:
        okk = []
        for w in d['workspaces']:
            if os.path.exists(os.path.join(path, d['path'],w)):
                okk.append('Found')
            else:
                okk.append('Not found')
        ok.append(okk)

    ## Creating output
    out = "Found {} datasets\n".format(len(da))

    for (i,j) in enumerate(zip(da,ok)):
        (d,p) = j
        out += "\n-- Dataset {}: {} (path {})\nDescription: {}\n\n".format(i, d['SampleName'], d['path'], d["description"])
        for (ii,jj) in enumerate(zip(d['workspaces'],p)):
            (dd,pp) = jj
            out += "  Cell {}: {} {}\n".format(ii, pp.upper(), dd)

    if string:
        return out
    else:
        return (da, ok)

if __name__ == "__main__":
    print(list_path(path="."))
