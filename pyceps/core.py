# ------------------------------------------ #
# Python Implementation of Cepstral Analysis #
# ------------------------------------------ #
import numpy as np
import librosa
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

from .utils import find_closest_ind


def rceps(D, sr=22050, qmin_ind=0):
    '''
    real cepstrum
    '''
    n_fft = int((D.shape[0]-1)*2)
    quef = np.arange(n_fft//2 + 1)/sr
    C = np.apply_along_axis(func1d=lambda x: np.fft.irfft(x).real, axis=0, arr=D)
    C = np.apply_along_axis(func1d=lambda x: x[ qmin_ind: x.size//2 +1], axis=0, arr=C) 
        
    return quef, C    

def cepsf0(D, sr=22050, fmax = 400, ss = 3, verbose=True, remove_outliers = False):
    '''
    f0 estimation from high quefrency of cepstrum
    '''
    n_fft = int((D.shape[0]-1)*2)
    quef = (np.arange(n_fft)/sr)[1:]
    
    qmin_ind = np.where( (1/quef) <= fmax)[0][0]
    _, C = rceps(D, sr=sr, qmin_ind=qmin_ind)
    
    if verbose:
        print("Search F0 in the range below {:.2f} [Hz] / quefrency: {:.7f}({})".format(1/quef[qmin_ind], quef[qmin_ind], qmin_ind))
    n_fft = int((C.shape[0]-1)*2)
    qf0_ind = qmin_ind + np.apply_along_axis(func1d=np.argmax, axis=0, arr=C)
    f0 = 1/quef[qf0_ind]
    if ss > 0:
        f0 = median_filter(f0, ss)
    lfrq = librosa.fft_frequencies(n_fft=n_fft, sr=sr)
    
    if remove_outliers:
        med_f0 = np.median(f0)
        iqr = np.diff(np.quantile(f0, (.25,.75)))
        outlier_ind = np.where( np.abs(f0 - med_f0) <= 1.5*iqr, 1, 0 )
        f0 *= outlier_ind
    f0_ind = find_closest_ind(f0, lfrq)    
    return f0, f0_ind


def cepsenv(C, lift_th=20):      
    '''
    spectral envelope estimation from low quefrency of cepstrum
    '''     
    env = C.copy()
    env = np.concatenate(  (env, env[1:][::-1]), axis=0 )
    env[lift_th:-lift_th, :] = 0.
    env = np.apply_along_axis(func1d=lambda x: np.fft.rfft(x).real[:C.shape[0]], axis=0, arr=env)
    return env
