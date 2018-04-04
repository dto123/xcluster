from deltasep_utils import gen_k_centers
import random

def main():
    centers, delta = gen_k_centers(2, 2)
    corners = []
    for center in centers:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        corner1_x = (center[0] - (0.5*x*delta))
        corner1_y = (center[1] - (0.5*y*delta))
        corner2_x = (center[0] + (0.5*x*delta))
        corner2_y = (center[1] + (0.5*y*delta))
        corners.append((corner1_x, corner1_y), corner2_x, corner2_y)
    print (corners)
main()
