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

    #pid,cid,p = loadData("../../../../data/glass.tsv")
    #pid,cid,p = loadData("../../../../data/aloi.tsv")
    #pid,cid,p = loadData("../../../../data/speaker_whitened.tsv")
    #pid,cid,p = loadData("../../../../data/ilsvrc12_50k.tsv")
    pid,cid,p = loadData("../../../../data/imagenet_full_100k.tsv")

    d = np.shape(np.array(p))[1]
    print d
    #aloi has 128 dimensions
    #speaker has 600 dimensions
    #ilsvrc12_50k has 2048 dimensions
    #imagenet has 2048 dimensions

    pca = PCA(np.array(p), 512)
    print pca.shape

    f= open("../../../../data/PCA_data_imagenet_dim-512.tsv", "w")
    #f= open("../../../../data/PCA_data.tsv", "w")
    #f= open("../../../../data/PCA_data_dim-7.tsv", "w")
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
