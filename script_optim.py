import numpy as np
from scipy.linalg import norm
from sklearn.decomposition import NMF
from scipy.optimize import minimize
import casadi as ca
import time
import matplotlib.pyplot as plt 

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from utils import *
from script_signal import compute_spectrogram
from scipy.signal import stft


def nmf_APGD():

#    A CODER
    
    return W, H, i


# Examples:
if __name__ == "__main__":
    
    m, n, k = 100, 80, 3  # Example dimensions
    
    # # Example 1
    # S = np.random.random((m, n))
    
    # Example 2 Shepp Logan
    shape = (m, n)
    image = shepp_logan_phantom()
    S = resize(image, shape, anti_aliasing=True)
    
    # # Example 3 Spectrogram
    # # Load audio
    # sample_rate, audio = load_audio("../audio/mixed.wav")
    # # Compute and display spectrogram
    # freqs, t, magnitude, phase = compute_spectrogram(audio, sample_rate)
    # S = magnitude
    # S = resize(S, shape, anti_aliasing=True)   
    
    W0 = np.random.random((m, k));
    H0 = np.random.random((k, n));
    
    fig, ax = plt.subplots(2, 4, figsize=(10, 10))
    ax[0, 0].imshow(S,cmap = "hsv")
    ax[0, 0].set_title('Original')
    plt.axis('off')
    ax[1, 0].imshow(np.matmul(W0,H0), cmap="hsv")
    ax[1, 0].set_title('Initial Guess')
    plt.axis('off')
        
    start_time = time.time()
    W, H, nb_iter = nmf_APGD() # A CODER
    print("Factorization with Alternating Projected Gradient complete.")
    print("Durée : ", time.time()-start_time)
    print("Norm of the error:", norm(S - np.matmul(W, H), 'fro'))
    print("L1-norm of (W,H):", norm(W.ravel(),1)+norm(H.ravel(),1)) 
    print()
    ax[0, 1].imshow(np.matmul(W,H), cmap="hsv")
    ax[0, 1].set_title('Projected Gradient')





