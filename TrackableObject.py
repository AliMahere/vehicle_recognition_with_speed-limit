# import the necessary packages
import numpy as np
from scipy.spatial import distance as dist


def calculateSpeeds(speeds,ontracking, frameTiem):
    
    for key, centroids in ontracking.items():
        
        if len(centroids)<2:
            speeds[key] = [0]
            continue
        D = dist.cdist(np.reshape(centroids[-2], (-1, 2)), np.reshape(centroids[-1], (-1, 2)))

        speed =  D[0][0]/frameTiem
        if key not in speeds.keys():
            speeds[key] = [speed]
        else :
            speeds[key].append(speed)
    return speeds
