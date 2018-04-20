import numpy as np
from PCA import PCA
from autoencoder import reduceData

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
    #a, c, p = loadData("../../../../data/speaker_whitened.tsv")
    pid,cid,p = loadData("../../../../data/glass.tsv")
    #pid,cid,p = loadData("../../../../data/aloi.scale.tsv")
    #pid,cid,p = loadData("../../../../data/speaker_whitened.tsv")
    #pid3,cid3,p3 = loadData("../../../../data/ilsvrc12_50k.tsv")
    #pid,cid,p = loadData("../../../../data/imagenet_full_100k.tsv")
    shape = np.shape(np.array(p))
    #points, dim = np.shape(np.array(p))
    #shape2 = np.shape(np.array(p2))
    #shape3 = np.shape(np.array(p3))
    #shape4 = np.shape(np.array(p4))
    #print (shape)


    #W = np.random.rand(dim,dim)
    #projectedP = np.matmul(p, W)


    #print (projectedP.shape)
    #print (shape2)
    #print (shape3)
    #print (shape4)
    #d = np.shape(np.array(p))[1]
    #print ("hey")
    #d = shape[0]*.75
    #print dog
    #print p.shape
    #d = (3*d)/4
    #d = d/2
    #print(d)
    #aloi has 128 dimensions
    #speaker has 600 dimensions
    #ilsvrc12_50k has 2048 dimensions
    #imagenet has 2048 dimensions



    #pca = PCA(np.array(p), 2)
    encoder = reduceData(np.array(p), 5)
    #encoder = reduceData(np.array(projectedP), 5)

    #print (encoder.shape)
    #print type(encoder)



    train_dim, new_dim =encoder.shape
    #print pca.shape
    #autoencoder = autoencoder(np.array(p), 2)
    #f= open("../../../../data/Autoencoder_data_ilsvrc_dim_2.tsv", "w")
    #f= open("../../../../data/Autoencoder_data_imagenet_dim_0.25d.tsv", "w")
    #f= open("../../../../data/AE_speaker_.5dim_ZM_dev.tsv", "w")
    #f= open("../../../../data/AE_glass_5d_epoch100_LR_0.01_ZM_withlossplot.tsv", "w")
    f= open("../../../../data/AE_glass_5d_epoch100_decoded.tsv", "w")
    #f= open("../../../../data/PCA_data_dim-3.tsv", "w")
    #f= open("../../../../data/PCA_data_dim-7.tsv", "w")
    #f= open("../../../../data/PCA_data_speaker.tsv", "w")
    for i in range(train_dim):
    #for i in range(projectedP.shape[0]):
        line = []
        line.append(pid[i])
        line.append(cid[i])
        #line.extend(list(pca[i,:]))
        #line.extend(list(encoder[i,:]))
        line.extend(list(encoder[i,:]))
        #line.extend(list(projectedP[i,:]))
        f.write("%s\n"%"\t".join([str(x) for x in line]))
        #print("%s\n"%"\t".join([str(x) for x in line]))

    #print pca
    f.close()

main()
