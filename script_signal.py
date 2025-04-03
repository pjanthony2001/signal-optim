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
    nmf = NMF(n_components=n_components, init="random", random_state=42, max_iter=500)
    W = nmf.fit_transform(magnitude)  # Basis matrix
    print(f"SHAPE OF W: {W.shape}")
    H = nmf.components_  # Activation matrix

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
    
def reconstruct_sources_gain(W, H):
    separated_sources = reconstruct_sources(W, H)
    
    source_energies = np.sum(H, axis=1)  # Sum over time
    
    results = []
    sorted_idx = np.argsort(source_energies)
    for k in range(1, source_energies.shape[0]):
      print(np.array(separated_sources).shape)
      print(sorted_idx.shape)
      selected_sources = np.array(separated_sources)[sorted_idx[k:].ravel(), :, :]
      synthesis_magn = np.sum(selected_sources, axis=0)
      results.append(synthesis_magn)
    
    return results    

    
    
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

    # Compute SRE
    signal_power = np.sum(signal**2)
    error_power = np.sum((signal - rec) ** 2)
    
    # Avoid division by zero (if error_power is zero, return a high SRE)
    if error_power == 0:
        return np.inf  # Perfect reconstruction

    sre = 10 * np.log10(signal_power / error_power)

    return sre

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
    print(H.shape)
    


    # Reconstruct the spectrogram for the dominant source
    # NMF and reconstruction with 30 sources
    n_components = 1000
    W, H = separate_sources(magnitude, n_components=n_components)
    
    data = W.T
    import umap

    umap_model = umap.UMAP(n_components=2)  # Reduce to 2D for visualization
    reduced_data = umap_model.fit_transform(data)

    # Step 2: Plot the 2D UMAP projection
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title("UMAP 2D Visualization of Clusters")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()
    source_energies = np.sum(H, axis=1)  # Sum over time
    print(source_energies)
    
    sources = reconstruct_sources_gain(W, H)
    # for i, source in enumerate(sources):
    #   synthesis = reconstruct_audio(source, phase, sample_rate, nperseg=1024) 
    #   save_audio(f"./results/result_{i}.wav", synthesis, sample_rate) 


        
    
