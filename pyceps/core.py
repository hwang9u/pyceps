# ------------------------------------------ #
# Python Implementation of Cepstral Analysis #
# ------------------------------------------ #
import numpy as np
import librosa
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from .utils import find_closest_ind


def rceps(D, qmin_ind=0, qmax_ind = None, sr=22050):
    '''
    real cepstrum obtained from log spectrogram (dB),
    '''
    n_fft = int((D.shape[0]-1)*2)
    quef = np.arange(n_fft//2 + 1)/sr # n_fft//2 + 1개 sr/2 ~ 0
    if qmax_ind == None:
        qmax_ind = n_fft//2 +1
    C = np.apply_along_axis(func1d=lambda x: np.fft.irfft(x).real, axis=0, arr=D)
    C = C[qmin_ind: qmax_ind, :]
    quef = quef[qmin_ind: qmax_ind]
    return quef, C    





def find_max_harm_ind(x, n =3):
    max_cand = np.argsort(x)[::-1][:n]
    max_cand_harm_ind = np.tile(max_cand, (n,1)) * np.arange(1,n+1).reshape(-1,1) 
    max_cand_harm_ind = np.where(max_cand_harm_ind > len(x), np.argmin(x), max_cand_harm_ind )
    harm_sum = x[max_cand_harm_ind].sum(axis=0)
    return max_cand[harm_sum.argmax()]

def find_max_db_ind(x,Dt,quef, lfrq, n=3):
    max_cand = np.argsort(x)[::-1][:n]
    freq_cand = 1/quef[max_cand]
    f0_ind_cand = find_closest_ind(freq_cand, lfrq)
    mag_cand = Dt[f0_ind_cand]
    return max_cand[mag_cand.argmax()]




def cepsf0(D, sr=22050, fmax = 400, fmin=0, win_size= 3, verbose=True, remove_outliers = False):
    '''
    f0 estimation from high quefrency of cepstrum
    '''
    n_fft = int((D.shape[0]-1)*2)
    lfrq = librosa.fft_frequencies(n_fft=n_fft, sr=sr) # linear frequency
    quef = (np.arange(n_fft//2 + 2)/sr)[1:] # quefrency (f: sr/2 ~ 0) w/o inf
    qmin_ind = np.where( (1/quef) <= fmax)[0][0]
    qmax_ind = np.where( (1/quef) >= fmin)[0][-1] if fmin > 1 else D.shape[0]-1
    if verbose:
        print("Search range: F0 [Hz] [{:.2f}, {:.2f}] / quefrency(index): [{:.5f}({}), {:.5f}({})] ".format(1/quef[qmin_ind], 1/quef[qmax_ind],
                                                                                                           quef[qmin_ind], qmin_ind,
                                                                                                           quef[qmax_ind], qmax_ind
                                                                                                           ))
        
    _, C = rceps(D, sr=sr, qmin_ind=qmin_ind, qmax_ind=qmax_ind) # real cepstrum
    qf0_ind = qmin_ind +  np.array(list(map(lambda t: find_max_db_ind(C[:,t], Dt=D[:,t], quef=_, lfrq=lfrq) , range(C.shape[1]) ) ))
    f0 = 1/quef[qf0_ind] # quefrency -> frequency
    
    # remove outliers using IQR method
    if remove_outliers: 
        med_f0 = np.median(f0)
        iqr = np.diff(np.quantile(f0, (.25,.75)))
        outlier_ind = np.where( np.abs(f0 - med_f0) <= 1.5*iqr, 1, 0 )
        f0 *= outlier_ind
    
    # median filtering
    if win_size > 0:
        f0 = median_filter(f0, win_size) # smoothing

    # mapping to linear frequency index
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