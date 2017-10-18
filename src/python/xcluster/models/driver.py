import numpy as np
from PCA import PCA

def loadData(filename):
    pid = []
    cid = []
    p = []
    with open(filename, 'r') as fin:
        for string in fin:
            splt = string.strip().split("\t")
            if (len(splt) < 2):
                println("ERROR: Line: " + string)
            pid.append(splt[0])
            cid.append(splt[1])
            p.append([float(x) for x in splt[2:]])
    return pid, cid, p




def main():

    pid,cid,p = loadData("../../../../data/glass.tsv")
    #pid,cid,p = loadData("../../../../data/aloi.tsv")
    #pid,cid,p = loadData("../../../../data/speaker_whitened.tsv")

    d = np.shape(np.array(p))[1]
    print d
    print d/2

    pca = PCA(np.array(p), 5)
    print pca.shape

    #f= open("../../../../data/PCA_data.tsv", "w")
    f= open("../../../../data/PCA_data_0.25d.tsv", "w")
    #f= open("../../../../data/PCA_data_speaker.tsv", "w")
    for i in range(len(pid)):
        line = []
        line.append(pid[i])
        line.append(cid[i])
        line.extend(list(pca[i,:]))

        f.write("%s\n"%"\t".join([str(x) for x in line]))
        #print("%s\n"%"\t".join([str(x) for x in line]))

    #print pca
    f.close()
    
main()
