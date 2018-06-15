#-*-coding:utf-8-*-
## writers.py is part of the fastspt library
## By MW, GPLv3+, Dec. 2017
## writers.py exports file formats widespread in SPT analysis


## ==== Imports
#import xml.etree.cElementTree as ElementTree
import scipy.io, pandas

## ==== Functions
def write_trackmate(da):
    """Experimental (not to say deprecated)"""
    #tree = ElementTree.Element('tmx', {'version': '1.4a'})
    Tracks = ElementTree.Element('Tracks', {'lol': 'oui'})
    ElementTree.SubElement(Tracks,'header',{'adminlang': 'EN',})
    ElementTree.SubElement(Tracks,'body')

    with open('/home/**/Bureau/myfile.xml', 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>')
        ElementTree.ElementTree(Tracks).write(f, 'utf-8')


def mat_to_csv(inF, outF):
    """Convert a .mat file (Anders's mat file) to a csv"""
    try:
        d = scipy.io.loadmat(inF)['trackedPar'][0]
    except:
        print("ERROR: "+inF)
        return

    t = []
    for (i, traj) in enumerate(d):
        for j in range(traj[0].shape[0]):
            t.append({'trajectory': i,
                      'x': traj[0][j, 0],
                      'y': traj[0][j, 1],
                      'frame': traj[1][j, 0]-1,
                      't': traj[2][j, 0]})
    pandas.DataFrame(t).to_csv(outF)


def write_4dn(df, hd):
    """"""
    raise NotImplementedError("Sorry :s")


if __name__ == "__main__":
    import sys
    print "Running standalone"
    #write_trackmate("")
    mat_to_csv(sys.argv[1], sys.argv[1].replace(".mat", ".csv"))
