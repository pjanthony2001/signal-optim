import numpy as np
from scipy.linalg import norm
from sklearn.decomposition import NMF
from scipy.optimize import minimize, least_squares
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
    
def grad_W_f(S, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    return 2 * (W @ H - S) @ H.T
    
def grad_H_f(S, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    return 2 * W.T @ (W @ H - S) 

def alpha_W(grad_W, H) -> float:
    return 0.5 * f_norm_sq(grad_W) / f_norm_sq(grad_W @ H)

def alpha_H(grad_H, W) -> float:
    return 0.5 * f_norm_sq(grad_H) / f_norm_sq(W @ grad_H)

def P1(M, tol) -> np.ndarray:
    # Projecting a matrix M on to the space of m by k real non-negative
    return np.maximum(M, tol)
    

def P2(M, tol) -> np.ndarray:
    # Projecting a matrix M on to the space of k by n real non-negative
    return np.maximum(M, tol)


def nmf_APGD(S, W0, H0, max_iter=50000, tol=1e-8):
    W, H = W0, H0
    epsilon = np.finfo(np.float64).eps
    
    i = 0
    while (f_norm_sq(grad_W_f(S, W, H)) >= tol or f_norm_sq(grad_H_f(S, W, H)) >= tol) and i < max_iter: 
        
        i = i + 1
        grad_W = grad_W_f(S, W, H)
        grad_H = grad_H_f(S, W, H)
        
        a_W = alpha_W(grad_W, H)
        
        W = P1(W - a_W * grad_W, epsilon)
        
        a_H = alpha_H(grad_H, W)
        H = P2(H - a_H * grad_H, epsilon)
        
    return W, H, i

def norm_ratio_W(W, W_p):
    return np.sqrt(f_norm_sq(W - W_p) / f_norm_sq(W_p))

def norm_ratio_H(H, H_p):
    return np.sqrt(f_norm_sq(H - H_p) / f_norm_sq(H_p))

def L_W(H):
    return 4 * f_norm_sq(H @ H.T) 

def L_H(W):
    return 4  * f_norm_sq(W.T @ W)

def Prox_mu_W(X, mu, lambda_W):
    return np.maximum(X - mu * lambda_W, 0)

def Prox_mu_H(X, mu, lambda_H):
    return np.maximum(X - mu * lambda_H, 0)

def PALM(S, W0, H0, y_W, y_H, lambda_W, lambda_H, max_iter=70000, tol=1e-8):
    W, H = W0, H0
    print(f"m = {W.shape[0]}, k = {W.shape[1]}, n = {H.shape[1]}")

    i = 0
    W_p = 0.5 * W0
    H_p = 0.5 * H0
    
    while norm_ratio_W(W, W_p) + norm_ratio_H(H, H_p) >= tol and i < max_iter:
        i = i + 1
        W_p = W
        c = y_W * L_W(H)
        mu_W = 1 / c
        W = Prox_mu_W(W - mu_W * grad_W_f(S, W, H), mu_W, lambda_W) 
        
        H_p = H
        d = y_H * L_H(W)
        mu_H = 1 / d
        H = Prox_mu_W(H - mu_H * grad_H_f(S, W, H), mu_H, lambda_H)
    
    return W, H, i


def keep_strict_top_p(arr, p):
    flat = arr.flatten()
    if p >= flat.size:
        return arr.copy()  # Nothing to mask

    top_p_indices = np.argpartition(flat, -p)[-p:]
    
    # Create a new zero array
    result_flat = np.zeros_like(flat)
    result_flat[top_p_indices] = flat[top_p_indices]

    return result_flat.reshape(arr.shape)


def Prox_mu_1_W(X, p_W):
    return keep_strict_top_p(np.maximum(X, 0), p_W)

def Prox_mu_1_H(X, p_H):
    return keep_strict_top_p(np.maximum(X, 0), p_H)


def PALM_1(S, W0, H0, y_W, y_H, p_W, p_H, max_iter=50000, tol=1e-8):
    W, H = W0, H0
    print(f"m = {W.shape[0]}, k = {W.shape[1]}, n = {H.shape[1]}")

    i = 0
    W_p = 0.5 * W0
    H_p = 0.5 * H0
    
    while norm_ratio_W(W, W_p) + norm_ratio_H(H, H_p) >= tol and i < max_iter:
        i = i + 1
        W_p = W
        c = y_W * L_W(H)
        mu_W = 1 / c
        W = Prox_mu_1_W(W - mu_W * grad_W_f(S, W, H), p_W) 
        
        H_p = H
        d = y_H * L_H(W)
        mu_H = 1 / d
        H = Prox_mu_1_H(H - mu_H * grad_H_f(S, W, H), p_H)
    
    return W, H, i


def scipy_optimize(S, W0, H0):
    m, n = S.shape
    k = W0.shape[1]
    x0 = np.concatenate([W0.flatten(), H0.flatten()])
    x_len = m*k + n*k
    
    def jac(x):
        W = x[:m * k].reshape((m, k))
        H = x[m * k:].reshape((k, n))
        grad_W = grad_W_f(S, W, H)
        grad_H = grad_H_f(S, W, H)
        
        return np.concatenate([grad_W.flatten(), grad_H.flatten()])
        
    def objective(x):
        W = x[:m*k].reshape((m, k))
        H = x[m*k:].reshape((k, n))
        return np.linalg.norm(S - W @ H, 'fro')

    res = minimize(objective, x0, method='trust-constr', bounds=[(0, None)] * x_len, jac=jac, hess=lambda _ : np.zeros((x_len, x_len)))
    W = res.x[:m*k].reshape((m, k))
    H = res.x[m*k:].reshape((k, n))

    return W, H

def sklearn_NMF(S, W0, H0):
    k = W0.shape[1]
    model = NMF(n_components=k, init='custom', solver='mu', max_iter=1000, random_state=42)

    W = model.fit_transform(S, W=W0, H=H0)
    H = model.components_
    
    return W, H


# Examples:
if __name__ == "__main__":
    
    m, n, k = 100, 80, 10  # Example dimensions
    
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
    ax[0, 0].imshow(S, cmap = "hsv")
    ax[0, 0].set_title('Original')
    plt.axis('off')
    ax[1, 0].imshow(np.matmul(W0,H0), cmap="hsv")
    ax[1, 0].set_title('Initial Guess')
    plt.axis('off')
        
    start_time = time.time()
    W, H, nb_iter = nmf_APGD(S, W0, H0)
    print("Factorization with Alternating Projected Gradient complete.")
    print("Durée : ", time.time()-start_time)
    print("Norm of the error:", norm(S - np.matmul(W, H), 'fro'))
    print("L1-norm of (W,H):", norm(W.ravel(),1)+norm(H.ravel(),1))
    print(f"Number of Iterations {nb_iter}")
    print()
    ax[0, 1].imshow(np.matmul(W,H), cmap="hsv")
    ax[0, 1].set_title('Projected Gradient')

    
    lambda_W, lambda_H = 1, 1
    y_W, y_H = 2, 2
    start_time = time.perf_counter()
    W, H, nb_iter = PALM(S, W0, H0, y_W, y_H, lambda_W, lambda_H)
    print("Factorization with PALM complete.")
    print("Durée : ", time.perf_counter() - start_time)
    print("Norm of the error:", norm(S - np.matmul(W, H), 'fro'))
    print("L1-norm of (W,H):", norm(W.ravel(),1)+norm(H.ravel(),1)) 
    print(f"Number of Iterations {nb_iter}")
    print()
    ax[0, 2].imshow(np.matmul(W,H), cmap="hsv")
    ax[0, 2].set_title('PALM')


    p_W, p_H = int(0.5 * (m * k)) , int(0.5 * (k * n))
    y_W, y_H = 1, 1
    start_time = time.perf_counter()
    W, H, nb_iter = PALM_1(S, W0, H0, y_W, y_H, p_W, p_H)
    print("Factorization with PALM 1 complete.")
    print("Durée : ", time.perf_counter() - start_time)
    print("Norm of the error:", norm(S - np.matmul(W, H), 'fro'))
    print("L1-norm of (W,H):", norm(W.ravel(),1)+norm(H.ravel(),1))
    print(f"Number of Iterations {nb_iter}")
    print()
    ax[0, 3].imshow(np.matmul(W,H), cmap="hsv")
    ax[0, 3].set_title('PALM 1')



    start_time = time.perf_counter()
    W, H = scipy_optimize(S, W0, H0)
    print("Factorization with scipy optimize complete.")
    print("Durée : ", time.perf_counter() - start_time)
    print("Norm of the error:", norm(S - np.matmul(W, H), 'fro'))
    print("L1-norm of (W,H):", norm(W.ravel(),1)+norm(H.ravel(),1))
    print(f"Number of Iterations {nb_iter}")
    print()
    ax[1, 1].imshow(np.matmul(W,H), cmap="hsv")
    ax[1, 1].set_title('scipy optimize')
    
    
    start_time = time.perf_counter()
    W, H = sklearn_NMF(S, W0, H0)
    print("Factorization with sklearn NMF complete.")
    print("Durée : ", time.perf_counter() - start_time)
    print("Norm of the error:", norm(S - np.matmul(W, H), 'fro'))
    print("L1-norm of (W,H):", norm(W.ravel(), 1) + norm(H.ravel(), 1))
    print(f"Number of Iterations {nb_iter}")
    print()
    ax[1, 2].imshow(np.matmul(W,H), cmap="hsv")
    ax[1, 2].set_title('sklearn NMF')
    
    plt.show()