import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# ------------------- Load/Save audio signal -----------------------------------

def load_audio(file_path):

    """
    Load an audio sample from a file and convert it to mono if it is stereo

    Parameters
    ----------
    file_path: string
      path to the audio file

    Returns
    -------
    sample_rate: int
      sample rate of the audio file
    audio: numpy.ndarray
      audio signal as a 1D numpy array
    """
    
    sample_rate, audio = wav.read(file_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print("Sample rate: " + str(sample_rate))
    return sample_rate, audio


def save_audio(file_path, audio, sample_rate):

    """
    Save the audio signal to a .wav file

    Parameters
    ----------
    file_path: string
      path to the audio file
    audio: numpy.ndarray
      audio signal as a 1D numpy array
    sample_rate: int
      sample rate
    """
    # Normalize and convert to int16
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)  
    wav.write(file_path, sample_rate, audio)
    
    
# --------------------- Display -----------------------------------------------s
   

def plot_audio_waveform(audio, sample_rate):
    """
    Plot the waveform of an audio signal.

    Parameters
    ----------
    audio: numpy.ndarray
        The audio signal as a 1D numpy array.
    sample_rate: int
        The sample rate of the audio signal.
    """
    # Create a time axis in seconds
    time = np.arange(len(audio)) / sample_rate

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, color='b')
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()



def plot_spectrogram(f, t, Zxx, sample_rate):

    """
    Plot the spectrogram of an audio signal.

    Parameters
    ----------
    f: numpy.ndarray
        Array of sample frequencies.
    t: numpy.ndarray
        Array of segment times.
    Zxx: numpy.ndarray
        STFT of the audio signal.
    sample_rate: int
        The sample rate of the audio signal.
    """
    # Compute the magnitude spectrogram (in dB)
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-9)  
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, magnitude_db, shading='gouraud', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, sample_rate / 2)  # Show frequencies up to the Nyquist frequency
    plt.show()
          

