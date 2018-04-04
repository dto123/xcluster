from deltasep_utils import gen_k_centers
import numpy as np

def main():
    centers, delta = gen_k_centers(2, 2)
    corners = []
    #datapoints = []
    for center in centers:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        corner1_x = (center[0] - (0.5*x*delta))
        corner1_y = (center[1] - (0.5*y*delta))
        corner2_x = (center[0] + (0.5*x*delta))
        corner2_y = (center[1] + (0.5*y*delta))

        datapoints_x = np.random.uniform(low=corner1_x, high = corner2_x, size = (25,))
        datapoints_y = np.random.uniform(low=corner1_y, high = corner2_y, size = (25,))

        datapoints = np.transpose(vstack((datapoints_x, datapoints_y)))
        print (datapoints.shape())

main()
