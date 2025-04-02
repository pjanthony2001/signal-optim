import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import stft, istft
from sklearn.decomposition import NMF

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
    
    # A CODER

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
    
    # A CODER


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
    
    # A CODER
    
    
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
    
    # A CODER
    

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
    
    # A CODER
 
# -------------------------------------------------------------------
# Launch script
# -------------------------------------------------------------------

if __name__ == '__main__':

    # Load audio
    sample_rate, audio = load_audio("./audio/mixed.wav")
    
    # Compute and display spectrogram
    freqs, t, magnitude, phase = compute_spectrogram(audio, sample_rate)
    plot_spectrogram(freqs, t, magnitude, sample_rate)
    
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
    
    # Extraction of the violin pist  
    # A CODER

        
    
