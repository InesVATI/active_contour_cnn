import numpy as np
import skfmm as fmm

def perform_fast_marching_isotropic(W, start_points, periodic=False):
    """
        Implementation of the fast marching algorithm in 2Ds using the skfmm library
        W : 2D weight matrix, must be positive
        start_points : 2 x n array, start_points[:,i] is the ith starting point
    """
    D_temp = np.ones_like(W) # 0-levelsets indicate boundary from which we compute distance
    D_temp[start_points[0,:],start_points[1,:]] = 0
    return fmm.travel_time(D_temp, speed=W, periodic=periodic) + 1e-15
    
