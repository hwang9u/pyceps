import numpy as np

def upsample(x, ups = 5):
    return np.tile(x.reshape(1,-1), (ups,1)).T.ravel()
    
def find_closest_ind(x , ref):
    fftmat = np.tile(ref.reshape(-1,1), (1, x.size))
    x_mat = np.tile(  x.reshape(1,-1), (len(ref),1))
    x_ind = np.abs(fftmat - x_mat).argmin(axis=0)
    return x_ind