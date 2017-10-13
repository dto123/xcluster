import numpy as np
from PCA import PCA

def loadData(filename):
    a = np.loadtxt(filename)
    return a


def main():

    data = loadData("../../../../data/glass.tsv")
    print data.shape

    pca = PCA(data)
    print pca.shape
    print pca

main()
