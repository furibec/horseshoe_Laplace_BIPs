# =================================================================
# Created by:
# Felipe Uribe @ DTU
# =================================================================
# Version 2021-01
# =================================================================
import time
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.special as spe
import scipy.io as spio
import pickle

import h5py
import hdf5storage
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
# matplotlib.rcParams.update({'font.size': 25})
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# matplotlib.rcParams['text.usetex'] = True

# from projection_functions import A, Amat
from samplers import linear_RTO

# =================================================================
# N = 32           # object size N-by-N pixels
# p = 32    # number of detector pixels
# q = 16            # number of projection angles
N = 64           # object size N-by-N pixels
q = 32            # number of projection angles
p = 64    # number of detector pixels
theta = np.linspace(0, 2*np.pi, q, endpoint=False)

# =================================================================
# load data and set-up likelihood
# =================================================================
# 'b_meas','b_meas_i','std_e','x_true_fine','t_true'
data = spio.loadmat('data/data_true_grains_N64_noi0p01.mat')

# underlying true
A = data['A']
x_truef = data['x_true'].flatten()
X_true = data['X_true']
xt_norm = np.linalg.norm(x_truef)

# noise and data
m = q*p 
y_data = data['b_data'].flatten()
B_data = data['B_data']
sigma_obs = data['std_e'].flatten()
lambd_obs = 1/(sigma_obs**2)

# =================================================================
# prior for the attenuation coefficient (x)
# =================================================================
d = int(N*N)            # dimension
mu_pr_x = np.zeros(d)   # prior mean
I = sp.sparse.identity(N, format='csc', dtype=int)

# 1D finite difference matrix with Neumann BCs
# D = sp.sparse.diags([-np.ones(N), np.ones(N)], offsets=[0, 1], shape=(N, N), format='csc', dtype=int)
# D[-1, -1] = 0
# 1D finite difference matrix with zero BCs
D = sp.sparse.diags([-np.ones(N), np.ones(N)], offsets=[-1, 0], shape=(N+1, N), format='csc', dtype=int)
D = D[:-1, :]

# 2D finite differences in each direction
D1 = sp.sparse.kron(I, D, format='csc')
D2 = sp.sparse.kron(D, I, format='csc')

# we approximate it with a Gaussian with structure
vareps = 1e-6
varepsvec = vareps*np.ones(d)
def Lk_fun(x_k): # returns L_1 and L_2 structure matrices
    diag1, diag2 = 1/np.sqrt((D1 @ x_k)**2 + varepsvec), 1/np.sqrt((D2 @ x_k)**2 + varepsvec)
    W1, W2 = sp.sparse.diags(diag1), sp.sparse.diags(diag2)
    w12 = np.hstack([diag1, diag2]) # weights on each direction
    #
    L_pr = (D1.T @ (W1 @ D1)) + (D2.T @ (W2 @ D2)) # prior structure matrix
    return L_pr, (W1.sqrt() @ D1), (W2.sqrt() @ D2), w12

# =================================================================
# =================================================================
# CONDITIONALS
# =================================================================
# =================================================================
# least-squares form
def proj_forward_reg(x, flag, W1sq_D1, W2sq_D2, lambd, delta):
    # regularized ASTRA projector [A; Lsq] w.r.t. x.
    if flag == 1:
        out1 = np.sqrt(lambd) * (A @ x)#A(x, 1) # A @ x
        out2 = np.sqrt(delta) * (W1sq_D1 @ x)
        out3 = np.sqrt(delta) * (W2sq_D2 @ x)
        out = np.hstack([out1, out2, out3])
    else:
        idx = int(len(x[m:])/2)
        out1 = np.sqrt(lambd) * (A.T @ x[:m]) # A(x[:m], 2) # A.T @ b
        out2 = np.sqrt(delta) * (W1sq_D1.T @ x[m:m+idx])
        out3 = np.sqrt(delta) * (W2sq_D2.T @ x[m+idx:])
        out = out1 + out2 + out3
    return out

# conditional precision of noise: lambd (analytical)
alpha_1 = (m/2) + 1
def pi_lambd_rnd(misfit):
    beta_1 = 0.5*(misfit.T @ misfit) + 1e-4 # rate
    return np.random.gamma(shape=alpha_1, scale=1/beta_1)

# conditional precision of X: delta (sample)
dbar = d-1
alpha_2 = (dbar) + 1 #(dbar/2) + 1
def pi_delta_rnd(x, L):
    beta_2 = (x.T @ (L @ x)) + 1e-4 # rate 0.5*(x.T @ (L @ x)) + 1e-4 
    return np.random.gamma(shape=alpha_2, scale=1/beta_2)

# =================================================================
# =================================================================
# Hybrid MCMC
# =================================================================
# =================================================================
# for CGLS
x_tol, n_cgls = 1e-4, 500

# samples
n_s = int(2e4)         # number of samples in Gibbs sampler
n_b = 2000       # burn-in
n_t = 10
nn_s = (n_s*n_t)+n_b           # total number of samples

# allocation
x_s = np.empty((d, n_s))
weights = np.zeros((2*d, n_s))
lambd_s = np.empty(n_s)
delta_s = np.empty(n_s)
cgls_it = np.empty(nn_s+1)

# initial state related params
Ltp1, W1sq_D1, W2sq_D2, weights_tp1 = Lk_fun(mu_pr_x) #, weights_tp1

# initial states
x_tp1 = mu_pr_x
lambd_tp1 = 5 #1e2
delta_tp1 = 1
              
# =================================================================
np.random.seed(10)
print('\n***Gibbs MCMC***\n')
# plt.figure(1)
st = time.time()
i = 0
for s in range(nn_s+1):     
    # ===X=========================================================
    G_fun = lambda x, flag: proj_forward_reg(x, flag, W1sq_D1, W2sq_D2, lambd_tp1, delta_tp1)
    x_tp1, cgls_it[s] = linear_RTO(x_tp1, G_fun, y_data, lambd_tp1, n_cgls, x_tol)
    # print(it)
    # plt.plot(s, it, 'b.')
    # plt.pause(0.01)

    # ===update Laplace approx=====================================
    Ltp1, W1sq_D1, W2sq_D2, weights_tp1 = Lk_fun(x_tp1) #, weights_tp1

    # ===hyperparams==============================================
    # noise precision
    misfit = y_data - (A @ x_tp1) # A(x_tp1, 1)
    lambd_tp1 = pi_lambd_rnd(misfit)

    # inverse scale
    delta_tp1 = pi_delta_rnd(x_tp1, Ltp1)
    
    # plt.figure(1)
    # plt.imshow(x_tp1.reshape((64,64)).T, extent=[0, 1, 0, 1], aspect='equal', cmap='YlGnBu_r')
    # plt.pause(1) 
    # msg
    if (s > n_b):
        # thinning
        if (np.mod(s, n_t) == 0):
            x_s[:, i] = x_tp1
            lambd_s[i] = lambd_tp1
            delta_s[i] = delta_tp1
            weights[:, i] = weights_tp1
            i += 1
            if (np.mod(i, 50) == 0):
                print("\nSample {:d}/{:d}".format(i, n_s))
    else:
        if (s == 0):
            print("\nBurn-in... {:d} samples\n".format(n_b))
                # print('\t relerr so far', e_x[k+1])
print('\nElapsed time:', time.time()-st, '\n')   
# ite = np.asarray(cgls_it)
#
# mdict = {'X':x_s, 'err_x':e_x, 'prec_x':delta_s, 'prec_noi':lambd_s, 'ite':ite}
# spio.savemat('deconv1D_high_Laplace_Gibbs_Nc2e4_Nb2e3.mat', mdict)
# with open('CT2D_N64_low_Laplace_Gibbs_Nc1e4_Nb1e3_Nt40.pkl', 'wb') as f:
#     pickle.dump([x_s, lambd_s, delta_s, cgls_it], f)
mdict = {'x_s':x_s, 'lambd_s':lambd_s, 'delta_s':delta_s, 'cgls_it':cgls_it, 'weights':weights}
hdf5storage.write(mdict, '.', 'CT2D_N64_low_Laplace_Gibbs_Nc2e4_Nb2e3_Nt10.mat', matlab_compatible=True)