from deltasep_utils import gen_k_centers
import numpy as np

def main():
    dim = 2
    centers, delta = gen_k_centers(3, dim)
    list_of_data = []
    rotated_data = []
    clusterList = []
    for center in centers:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        corner1_x = (center[0] - (0.5*x*delta))
        corner1_y = (center[1] - (0.5*y*delta))
        corner2_x = (center[0] + (0.5*x*delta))
        corner2_y = (center[1] + (0.5*y*delta))

        datapoints_x = np.random.uniform(low=corner1_x, high = corner2_x, size = (25,))
        datapoints_y = np.random.uniform(low=corner1_y, high = corner2_y, size = (25,))

        random_datapoints = np.transpose(np.vstack((datapoints_x, datapoints_y)))
        #print (random_datapoints.shape)
        list_of_data.append(random_datapoints)
        cluster = 1
        for i in range(25):
            clusterList.append(np.array(cluster))
        cluster+=1





    #print (len(list_of_data))
    W = np.random.rand(dim,dim)
    for points in list_of_data:
        newpoints = np.matmul(points, W)
        rotated_data.append(newpoints)
    #print (len(rotated_data))

    final_data = np.vstack(list_of_data)
    final_rotated_data = np.vstack(rotated_data)

    d1, d2 = final_data.shape
    pidList = []
    for a in range(d1):
        one = np.array(a)
        pidList.append(one)

    clusterList = np.asarray(clusterList)[...,None]
    pidList = np.asarray(pidList)[...,None]


    print (final_data.shape)
    print (final_rotated_data.shape)

    print (pidList.shape)
    print (clusterList.shape)

main()
