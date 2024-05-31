import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ranksums
from scipy.signal import correlate
import warnings
import numpy as np
# from numba import jit
import BOLDFilters
import matplotlib.pyplot as plt


# Cargar datos

NSUB = 36
NPARCELLS = 360
Tmax = 197
lambda_val = 0.18

# Parámetros de los datos
TR = 3 # Tiempo de Repetición (segundos)

# Configuración del filtro pasa banda
fnq = 1 / (2 * TR)  # Frecuencia de Nyquist
flp = 0.01  # Frecuencia de corte inferior del filtro (Hz)
fhi = 0.09  # Frecuencia de corte superior del filtro (Hz)
Wn = [flp / fnq, fhi / fnq]  # Frecuencia no dimensional del filtro Butterworth
k = 2  # Orden del filtro Butterworth
bfilt, afilt = butter(k, Wn, btype='band',analog=False)  # Construir el filtro






def Turbulencia(ts):
    # Inicialización de matrices para almacenar resultados
    enstrophy = np.zeros((NPARCELLS, Tmax))
    enstrophy_su = np.zeros((NPARCELLS, Tmax))
    signal_filt = np.zeros((NPARCELLS, Tmax))
    Phases = np.zeros((NPARCELLS, Tmax))
    Phases_su = np.zeros((NPARCELLS, Tmax))
    Rspatime = np.zeros(1)
    Rspa = np.zeros((1, NPARCELLS))
    Rtime = np.zeros((1, Tmax))
    acfspa = np.zeros((1, 101))
    acftime = np.zeros((1, 101))
    Rspatime_su = np.zeros(1)
    Rspa_su = np.zeros((1, NPARCELLS))
    Rtime_su = np.zeros((1, Tmax))
    acfspa_su = np.zeros((1, 101))
    acftime_su = np.zeros((1, 101))


    Schaefercog = np.loadtxt('glasser_coords.txt')
    # Cálculo de la matriz de distancias
    rr = np.zeros((NPARCELLS, NPARCELLS))
    for i in range(NPARCELLS):
        for j in range(NPARCELLS):
            rr[i, j] = np.linalg.norm(Schaefercog[i,:] - Schaefercog[j,:])

    # Construcción de la matriz de correlación exponencial
    Cexp = np.exp(-lambda_val * rr)
    np.fill_diagonal(Cexp, 1)



    for seed in range(NPARCELLS):
        ts[seed, :] = ts[seed, :] - np.mean(ts[seed,:])  # Detrending
        signal_filt[seed,:] = filtfilt(bfilt, afilt, ts[seed,:])
        Xanalytic = hilbert(signal_filt[seed,:])
        Xanalytic = Xanalytic - np.mean(Xanalytic)
        Phases[seed, :] = np.angle(Xanalytic)
        Phases_su[seed, :] = Phases[seed, np.random.permutation(Tmax)]

    Phases_su = Phases_su[np.random.permutation(NPARCELLS), :]

    for i in range(NPARCELLS):
        sumphases = np.nansum(np.tile(Cexp[i, :].T, (Tmax, 1)).T * np.exp(1j * Phases)) / np.nansum(Cexp[i, :])
        enstrophy[i] = np.abs(sumphases)
        sumphases = np.nansum(np.tile(Cexp[i, :].T, (Tmax, 1)).T * np.exp(1j * Phases_su)) / np.nansum(Cexp[i, :])
        enstrophy_su[i] = np.abs(sumphases)

    Rspatime = np.nanstd(enstrophy)
    Rspatime_su = np.nanstd(enstrophy_su)

    Rspa= np.nanstd(enstrophy,axis=1).T
    Rtime = np.nanstd(enstrophy,axis=0)
    acfspa = np.correlate(Rspa, Rspa, mode='full')[len(Rspa) - 1:]
    acftime = np.correlate(Rtime, Rtime, mode='full')[len(Rtime) - 1:]

    Rspa_su = np.nanstd(enstrophy_su, axis=1).T
    Rtime_su = np.nanstd(enstrophy_su,axis=0)
    acfspa_su = np.correlate(Rspa_su, Rspa_su, mode='full')[len(Rspa_su) - 1:]
    acftime_su = np.correlate(Rtime_su, Rtime_su, mode='full')[len(Rtime_su) - 1:]

    results = {
        'Rspatime': Rspatime,
        'Rspatime_su': Rspatime_su,
        'Rspa': Rspa,
        'Rtime': Rtime,
        'Rspa_su': Rspa_su,
        'Rtime_su': Rtime_su,
        'acfspa': acfspa,
        'acftime': acftime,
        'acfspa_su': acfspa_su,
        'acftime_su': acftime_su
    }



    return Rspatime,Rspatime_su,Rspa,Rtime,Rspa_su,Rtime_su,acfspa,acftime,acfspa_su,acftime_su



def from_fMRI(signal, applyFilters=False, removeStrongArtefacts=True):
    if not np.isnan(signal).any():  # No problems, go ahead!!!
        if applyFilters:
            signal_filt = BOLDFilters.BandPassFilter(signal, removeStrongArtefacts=removeStrongArtefacts)
            sfiltT = signal_filt
        else:
            sfiltT = signal
        Rspatime, Rspatime_su, Rspa, Rtime, Rspa_su, Rtime_su, acfspa, acftime, acfspa_su, acftime_su = Turbulencia(sfiltT)  # Pearson correlation coefficients

        return Rspatime,Rspatime_su,Rspa,Rtime,Rspa_su,Rtime_su,acfspa,acftime,acfspa_su,acftime_su

    else:
        warnings.warn('############ Warning!!! FC.from_fMRI: NAN found ############')
        # n = signal.shape[0]
        return np.nan
