import numpy as np
import PCA

def loadData(filename):
    a = np.loadtxt(filename)
    return a


def main():
    print "hi"
    data = loadData("../../../../data/glass.tsv")
    pca = PCA(data)
    print pca.shape
    print pca

main()
