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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 25})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

from image_deblurring_1D import A, Gauss, Defocus
from samplers import linear_RTOw

# ===================================================================
# domain discretization
# ===================================================================
d = 128
bnds = np.array([0, 1])
grid = np.linspace(bnds[0], bnds[1], d)
h = grid[1]-grid[0]

# =========================================================
# forward operator
# =========================================================
# set the PSF array
opt = 1
s_PSF = 4
nn = int(np.ceil((d/20)/2)*2 + 1) # odd number
if (opt == 1):
    P, center = Gauss(nn, s_PSF) 
elif (opt == 2):
    P, center = Defocus(nn, s_PSF)
#
Aop = lambda x, flag: A(x, flag, P, 'constant')

# =================================================================
# load data and set-up likelihood
# =================================================================
# load truth and noisy convolved data
m = np.copy(d)
with open('data/data_m128_med.pkl', 'rb') as f:
    xx, x_truef, g_true, e, y_data, sigma_obs = pickle.load(f)
# f_true += 1e-7
xt_norm = np.linalg.norm(x_truef)
lambd_obs = 1/(sigma_obs**2)
# Lambda_obs = sp.sparse.diags(lambd_obs*np.ones(d))

# =================================================================
# =================================================================
# prior for the attenuation coefficient (x)
# =================================================================
mu_pr_x = np.zeros(d)   # prior mean

# 1D finite difference matrix with Neumann BCs
# D = sp.sparse.spdiags([-np.ones(d), np.ones(d)], [0, 1], d, d).tocsr()
# D[-1, -1] = 0
D = sp.sparse.diags([-np.ones(d), np.ones(d)], offsets=[-1, 0], shape=(d+1, d), format='csc', dtype=int)
D = D[:-1, :]

# we approximate it with a Gaussian with structure
vareps = 1e-6
varepsvec = vareps*np.ones(d)
def Lk_fun(x_k): # returns L_1 and L_2 structure matrices
    diag = 1/np.sqrt((D @ x_k)**2 + varepsvec)
    W = sp.sparse.diags(diag)
    return (D.T @ (W @ D)), (W.sqrt() @ D), diag

# =================================================================
# =================================================================
# CONDITIONALS
# =================================================================
# =================================================================
# least-squares form
def proj_forward_reg(x, flag, Wsq_D, lambd, delta):
    # regularized ASTRA projector [A; Lsq] w.r.t. x.
    if flag == 1:
        out1 = np.sqrt(lambd) * Aop(x, 1) # A @ x
        out2 = np.sqrt(delta) * (Wsq_D @ x)
        out = np.hstack([out1, out2])
    else:
        out1 = np.sqrt(lambd) * Aop(x[:m], 2) # A.T @ b
        out2 = np.sqrt(delta) * (Wsq_D.T @ x[m:])
        out = out1 + out2
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
np.random.seed(5)
# for CGLS
x_tol, n_cgls = 1e-4, 500

# samples
n_s = int(2e4)         # number of samples in Gibbs sampler
n_b = int(0.1*n_s)       # burn-in
n_t = 20
nn_s = (n_s*n_t)+n_b           # total number of samples

# allocation
x_s = np.zeros((d, n_s))
weights = np.zeros((d, n_s))
lambd_s = np.zeros(n_s)
delta_s = np.zeros(n_s)
cgls_it = list()

# initial state related params
L, Wsq_D, weights_tp1 = Lk_fun(mu_pr_x)

# initial states
x_tp1 = mu_pr_x
lambd_tp1 = lambd_obs #1e2
delta_tp1 = 1

# =================================================================
np.random.seed(10)
print('\n***Gibbs MCMC***\n')
# plt.figure(1)
st = time.time()
i = 0
for s in range(nn_s+1):    
    # ===X=========================================================
    G_fun = lambda x, flag: proj_forward_reg(x, flag, Wsq_D, lambd_tp1, delta_tp1)
    x_tp1, it = linear_RTOw(x_tp1, G_fun, y_data, lambd_tp1, n_cgls, x_tol)
    cgls_it.append(it)
    # plt.plot(s, it, 'b.')
    # plt.pause(0.01)
    
    # ===update Laplace approx=====================================
    L, Wsq_D, weights_tp1 = Lk_fun(x_tp1)

    # ===hyperparams==============================================
    # noise precision
    misfit = y_data - Aop(x_tp1, 1)
    lambd_tp1 = pi_lambd_rnd(misfit)

    # inverse scale
    delta_tp1 = pi_delta_rnd(x_tp1, L)

    # msg
    if (s > n_b):
        # thinning
        if (np.mod(s, n_t) == 0):
            x_s[:, i] = x_tp1
            lambd_s[i] = lambd_tp1
            delta_s[i] = delta_tp1
            weights[:, i] = weights_tp1
            i += 1
            if (np.mod(i, 100) == 0):
                print("\nSample {:d}/{:d}".format(i, n_s))
    else:
        if (s == 0):
            print("\nBurn-in... {:d} samples\n".format(n_b))
                # print('\t relerr so far', e_x[k+1])
        
print('\nElapsed time:', time.time()-st, '\n')   
ite = np.asarray(cgls_it)
#
# mdict = {'X':x_s, 'err_x':e_x, 'prec_x':delta_s, 'prec_noi':lambd_s, 'ite':ite}
# spio.savemat('deconv1D_high_Laplace_Gibbs_Nc2e4_Nb2e3.mat', mdict)
# with open('deconv1D_med_Laplace_Gibbs_Nc2e4_Nb2e3_Nt20.pkl', 'wb') as f:
#     pickle.dump([x_s, lambd_s, delta_s, ite, weights], f)
