import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import stft, istft
from sklearn.decomposition import NMF
from numpy.linalg import norm
import pandas as pd
from scipy.stats import kurtosis, entropy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt



# -------------------------- Spectrogram --------------------------------------

def compute_spectrogram(audio, sample_rate, nperseg=1024):

    """
    Compute the spectrogram of an audio signal using the 
    Short-Time Fourier Transform (STFT).

    Parameters
    ----------
    audio: numpy.ndarray
      audio signal as a 1D numpy array
    sample_rate: int
      sample rate
    nperseg: int (default: 1024)
      length of each segment for STFT

    Returns
    -------
    f: numpy.ndarray
      sample frequencies.
    t: numpy.ndarray
      segment times.
    magnitude: numpy.ndarray
      magnitude spectrogram (2D array).
    phase: numpy.ndarray
      phase spectrogram (2D array).
    """
    
    
    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg)

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    return f, t, magnitude, phase

def reconstruct_audio(magnitude, phase, sample_rate, nperseg=1024):

    """
    Reconstruct an audio signal from its magnitude and phase spectrograms 
    using the inverse STFT.

    Parameters
    ----------
    magnitude: numpy.ndarray
      magnitude spectrogram (2D array).
    phase: numpy.ndarray
      phase spectrogram (2D array).
    sample_rate: int
      sample rate
    nperseg: int (default: 1024)
      length of each segment for STFT

    Returns
    -------
    audio: numpy.ndarray
      reconstructed audio signal (1D array)
    """
    
    # Reconstruct the complex STFT matrix
    
    Zxx = magnitude * np.exp(1j * phase)

    # Compute the ISTFT
    _, reconstructed_audio = istft(Zxx, fs=sample_rate, nperseg=nperseg)

    return reconstructed_audio


# ------------------------------- NMF ------------------------------------------

def separate_sources(magnitude, n_components=3):

    """
    Separate audio sources using Non-negative Matrix Factorization (NMF).

    Parameters
    ----------
    magnitude: numpy.ndarray
      magnitude spectrogram (2D array).
    n_components: int (default: 3)
      number of components for NMF

    Returns
    -------
    W: numpy.ndarray
      of size (n_frequencies, n_components).
    H: numpy.ndarray
      activation matrix of size (n_components, n_times).
    """
    
    # Apply NMF to decompose the magnitude spectrogram
    nmf = NMF(n_components=n_components, init="random", random_state=42, max_iter=1000)
    W = nmf.fit_transform(magnitude) 
    H = nmf.components_ 

    return W, H
    
    
def reconstruct_sources(W, H):

    """
    Reconstruct the spectrogram for each source using the basis 
    vectors and activation matrix.

    Parameters
    ----------
    W: numpy.ndarray
        Basis vectors (spectral patterns) of size (n_frequencies, n_components).
    H: numpy.ndarray
        Activation matrix (time-varying gains) of size (n_components, n_times).

    Returns
    -------
    sources: list of numpy.ndarray
        List of reconstructed spectrograms for each component.
    """
    

    source_spectrograms = [W[:, i:i+1] @ H[i:i+1, :] for i in range(W.shape[1])]
    return source_spectrograms
    

def reconstruct_sources_idx(W, H, idx):
    '''
    Useful metric to reconstruct only the sources indicated by the index parameter.
    '''
    idx = list(idx)
    source_spectrograms = [W[:, i:i+1] @ H[i:i+1, :] for i in idx]
    return source_spectrograms

# -------------- Signal-to-reconstruction error (SRE)--------------------------

def SRE(signal, rec):

    """
    Signal-to-Reconstruction Error
    
    Parameters
    ----------
    
    signal: array-like
      original signal
    rec: array-like
      reconstructed signal
      
    Return
    ------
    
    SRE: float
      signal-to-reconstruction error 
    """
    
    # Ensure input signals are numpy arrays
    signal = np.asarray(signal)
    rec = np.asarray(rec)
    
    min_length = min(signal.shape[0], rec.shape[0])
    signal = signal[:min_length]
    rec = rec[:min_length]

    signal_power = np.sum(signal**2)
    error_power = np.sum((signal - rec) ** 2)
    
    if error_power == 0: # Avoid division by zero 
        return np.inf  

    sre = 10 * np.log10(signal_power / error_power)

    return sre
  
  
# ----------------------------------
# Additional functions
# -----------------------------------

def harmonic_peak_stats(w_col):
    peaks, _ = find_peaks(w_col, height=0.05)
    print(peaks)
    if len(peaks) < 2:
        return 0, len(peaks), peaks[0], peaks[0], 0, 0
    diffs = np.diff(peaks)
    mean_spacing = np.mean(diffs)
    max_ = np.max(peaks)
    min_ = np.min(peaks)
    sd = np.std(peaks)
    mean = np.mean(peaks)
    return (mean_spacing, len(peaks), min_, max_, mean, sd)

def analyze_W(W):
    spacings = []
    num_peaks = []
    max_peaks = []
    min_peaks = []
    sd_peaks = []
    mean_peaks = []


    for k in range(W.shape[1]):
        col = W[:, k]
        spacing, n_peaks, min_, max_, mean, sd = harmonic_peak_stats(col)
        spacings.append(spacing)
        num_peaks.append(n_peaks)
        max_peaks.append(max_)
        min_peaks.append(min_)
        sd_peaks.append(sd)
        mean_peaks.append(mean)

    df = pd.DataFrame({
        'Component': np.arange(W.shape[1]),
        'PeakSpacing': spacings,
        'NumPeaks': num_peaks,
        'MaxPeaks': max_peaks,
        'MinPeaks': min_peaks,
        'SD': sd_peaks,
        'MeanPeaks': mean_peaks
    })

    
    return df[df.NumPeaks >= 2][df.SD > 20][df.MeanPeaks > 50] # due to Harmonics, Violins should have multiple peaks
  
def plot_top_W(W, top_indices):
    plt.figure(figsize=(12, 6))  
    for i, idx in enumerate(top_indices):
        plt.plot(W[:, idx], label=f'Component {idx}')
    plt.title('Top Candidate Components')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.grid(True)


    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), 
               ncol=6, fontsize='medium', frameon=False)

    plt.tight_layout()
    plt.show()

def plot_WH(W, H):
    plt.figure(figsize=(12, 6)) 
    for i, idx in enumerate(range(W.shape[1])):
        plt.plot(W[:, idx], label=f'Component {idx}')
    plt.title('Components of W')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.grid(True)


    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), 
               ncol=6, fontsize='medium', frameon=False)

    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(range(W.shape[1])):
        plt.plot(H[idx, :], label=f'Component {idx}')
    plt.title('Components of H')
    plt.xlabel('Time Bin')
    plt.ylabel('Magnitude of Gain')
    plt.grid(True)


    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), 
               ncol=6, fontsize='medium', frameon=False)

    plt.tight_layout()
    plt.show()

    

# -------------------------------------------------------------------
# Launch script
# -------------------------------------------------------------------

if __name__ == '__main__':

    # Load audio
    sample_rate, audio = load_audio("./audio/mixed.wav")
    
    # Compute and display spectrogram
    freqs, t, magnitude, phase = compute_spectrogram(audio, sample_rate)
    # plot_spectrogram(freqs, t, magnitude, sample_rate)
    
    # NMF and reconstruction with 3 sources
    n_components = 3
    W, H = separate_sources(magnitude, n_components=n_components)
    sources = reconstruct_sources(W, H)
    synthesis_magn = np.sum(sources, axis=0)
    synthesis = reconstruct_audio(synthesis_magn, 
      phase, sample_rate, nperseg=1024)
    err = SRE(audio, synthesis)
    save_audio("./results/synthesis3.wav", synthesis, sample_rate)
    print("SRE with " + str(n_components) + " : " + str(err))
    
    # NMF and reconstruction with 30 sources
    n_components = 30
    W, H = separate_sources(magnitude, n_components=n_components)
    sources = reconstruct_sources(W, H)
    synthesis_magn = np.sum(sources, axis=0)
    synthesis = reconstruct_audio(synthesis_magn, 
      phase, sample_rate, nperseg=1024)
    err = SRE(audio, synthesis)
    save_audio("./results/synthesis30.wav", synthesis, sample_rate) 
    print("SRE with " + str(n_components) + " : " + str(err))
    plot_WH(W, H)
    
    df = analyze_W(W)
    print(df)

    # Plot top components by entropy
    top_components = df['Component'].values
    plot_top_W(W, top_components)
    sources = reconstruct_sources_idx(W, H, top_components)
    synthesis_magn = np.sum(sources, axis=0)
    synthesis = reconstruct_audio(synthesis_magn, 
      phase, sample_rate, nperseg=1024)
    err = SRE(audio, synthesis)
    save_audio("./results/synthesisViolin.wav", synthesis, sample_rate) 
    print("SRE with " + str(n_components) + " : " + str(err))
  




