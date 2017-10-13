import numpy as np

def loadData(filename):
    a = np.loadtxt(filename)
    return a


def main():
    loadData("../../../../data/glass.tsv")
