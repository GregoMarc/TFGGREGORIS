@Translated by Marc Gregoris
import numpy as np
from scipy.stats import ranksums
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import warnings
import numpy as np
# from numba import jit
import BOLDFilters
import measures
from scipy.stats import pearsonr


def corrcoef(x, y):

    cov_matrix = np.cov(x.T, y.T)

    # Extreure les submatrius de covariança entre x e y
    cov_xx = cov_matrix[:x.shape[1], :x.shape[1]]
    cov_xy = cov_matrix[:x.shape[1], x.shape[1]:]
    cov_yx = cov_matrix[x.shape[1]:, :x.shape[1]]
    cov_yy = cov_matrix[x.shape[1]:, x.shape[1]:]

    # Calcular les desviacions estandar de x e y
    std_x = np.sqrt(np.diag(cov_xx))
    std_y = np.sqrt(np.diag(cov_yy))

    # Calcular la matriu de correlació
    corr_matrix = cov_xy / np.outer(std_x, std_y)

    return corr_matrix

N = 80
NLAG = 6




#Funció insideout cridada desde el from fmri
def InsideOUT (ts):
    FowRev = np.zeros((NLAG,))
    AsymFow = np.zeros((NLAG,))
    AsymRev = np.zeros((NLAG,))
    Tm = ts.shape[1]
    Ts0 = ts.shape[0]

    FCtau_foward = np.zeros((NLAG, Ts0,Ts0))
    FCtau_reversal = np.zeros((NLAG, Ts0, Ts0))

    for Tau in range(1,NLAG + 1):

        # Calcular la correlació cap endevant (forward)
        ts_1 = ts[:, :-Tau].T
        ts_2 = ts[:, Tau:].T
        FCtau_foward = corrcoef(ts_1, ts_2)
        


        # Calcular la correlació cap enrere (reversal)
        ts_11 = ts[:, Tm-1: Tau - 1: -1].T

        ts_22 = ts[:, Tm - Tau -1:: -1].T
        FCtau_reversal = corrcoef(ts_11, ts_22)
       

        # Squeeze per eliminar dimensions adicionals si es necesari
        FCtf = np.squeeze(FCtau_foward[:379,:379]) #NO CAL? PYCHARM JA TRACTA LA MATRIU COM 379X379
        FCtr = np.squeeze(FCtau_reversal[:379,:379])

        Itauf = -0.5 * np.log(1 - FCtf**2)
        Itaur = -0.5 * np.log(1 - FCtr**2)
        Reference = ((Itauf[:] - Itaur[:])**2).T
        threshold = np.quantile(Reference, 0.95)

        # Trobar els indexs a on la diferencia al quadrat es mayor que el percentil 95
        index = np.where(Reference > threshold)
        FowRev[Tau-1] = np.nanmean(Reference[index])
        AsymFow[Tau-1] = np.mean(np.abs(Itauf - Itauf.T))
        AsymRev[Tau-1] = np.mean(np.abs(Itaur - Itaur.T))

    return FowRev,AsymRev,AsymFow



# @jit(nopython=True)
def from_fMRI(signal, applyFilters=True, removeStrongArtefacts=True):
    if not np.isnan(signal).any():  
        if applyFilters:
            signal_filt = BOLDFilters.BandPassFilter(signal, removeStrongArtefacts=removeStrongArtefacts)
            sfiltT = signal_filt
        else:
            sfiltT = signal
        cc = InsideOUT(sfiltT)  

        return cc

    else:
        warnings.warn('############ Warning!!! FC.from_fMRI: NAN found ############')]
        return np.nan
