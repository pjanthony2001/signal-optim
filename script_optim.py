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

def f_norm_sq(A: np.ndarray) -> float: # Given that the matrices are all real
    return np.sum(A * A)
    
def grad_W_f(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    return (W @ H - S) @ H.T
    
def grad_H_f(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    return W.T @ (W @ H - S) 

def alpha_W(grad_W, H) -> float:
    
    return 0.25 * (f_norm_sq(grad_W) / f_norm_sq(grad_W @ H))

def alpha_H(grad_H, W) -> float:
    return 0.25 * (f_norm_sq(grad_H) / f_norm_sq(W @ grad_H))

def P1(M) -> np.ndarray:
    # Projecting a matrix M on to the space of m by k real non-negative
    return np.maximum(M, 0)
    

def P2(M) -> np.ndarray:
    # Projecting a matrix M on to the space of k by n real non-negative
    return np.maximum(M, 0)


def nmf_APGD(S, W0, H0, max_iter=1000, tol=1e-6):
    W, H = W0, H0
    print(f"m = {W.shape[0]}, k = {W.shape[1]}, n = {H.shape[1]}")
    
    i = 0
    while f_norm_sq(grad_W_f(W, H)) > tol and f_norm_sq(grad_H_f(W, H)) > tol and i <= max_iter: 
        # There is an error in the algorithm given in the pdf
        
        i = i + 1
        grad_W = grad_W_f(W, H)
        grad_H = grad_H_f(W, H)
        
        a_W = alpha_W(grad_W, H)
        W = P1(W - a_W * grad_W)
        
        a_H = alpha_H(grad_H, W)
        H = P2(H - a_H * grad_H)
        
    return W, H, i

def norm_ratio_W(W, W_p):
    return f_norm_sq(W - W_p) / f_norm_sq(W_p)

def norm_ratio_H(H, H_p):
    return f_norm_sq(H - H_p) / f_norm_sq(H_p)

def L_W(H):
    return 4 * f_norm_sq(H @ H.T) 

def L_H(W):
    return 4  * f_norm_sq(W.T @ W) 

def PALM(S, W0, H0, y_W, y_H, max_iter=1000, tol=1e-5):
    W, H = W0, H0
    print(f"m = {W.shape[0]}, k = {W.shape[1]}, n = {H.shape[1]}")
    
    i = 0
    W_p = (1 / (tol + 1)) * W0
    H_p = (1 / (tol + 1)) * H0
    
    while norm_ratio_W(W, W_p) + norm_ratio_H(H, H_p) > tol and i < max_iter:
        i = i + 1
        W_p = W
        c = y_W * L_W(H)

        
        H_p = H
        d = y_H * L_H(W)
    
    return W, H
        

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
    
    W0 = np.random.random((m, k))
    H0 = np.random.random((k, n))

    
    fig, ax = plt.subplots(2, 4, figsize=(10, 10))
    ax[0, 0].imshow(S,cmap = "hsv")
    ax[0, 0].set_title('Original')
    plt.axis('off')
    ax[1, 0].imshow(np.matmul(W0,H0), cmap="hsv")
    ax[1, 0].set_title('Initial Guess')
    plt.axis('off')
        
    start_time = time.time()
    W, H, nb_iter = nmf_APGD(S, W0, H0) # A CODER
    print("Factorization with Alternating Projected Gradient complete.")
    print("Durée : ", time.time()-start_time)
    print("Norm of the error:", norm(S - np.matmul(W, H), 'fro'))
    print("L1-norm of (W,H):", norm(W.ravel(),1)+norm(H.ravel(),1)) 
    print()
    ax[0, 1].imshow(np.matmul(W,H), cmap="hsv")
    ax[0, 1].set_title('Projected Gradient')





