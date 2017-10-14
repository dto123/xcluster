import numpy as np
from PCA import PCA

def loadData(filename):
    pid = []
    cid = []
    p = []
    with open(filename, 'r') as fin:
        for string in fin:
            a = string.strip().split("\t")
            if (len(a) < 2)
                println("ERROR: Line: " + string)
            pid.append(a[0])
            cid.append(a[1])
            p.append([float(x) for x in a[2:]])
    return pid, cid, p




def main():

    pid,cid,p = loadData("../../../../data/glass.tsv")


    pca = PCA(np.array(p))
    print pca.shape

    f= open("../../../../data/PCA_data.tsv", "w")
    for i in range(len(pid)):
        line = []
        line.append(pid[i])
        line.append(cid[i])
        line.extend(list(pca[i,:]))

        f.write("%s\n"%"\t".join([str(x) for x in line]))

    #print pca
    f.close()
main()
