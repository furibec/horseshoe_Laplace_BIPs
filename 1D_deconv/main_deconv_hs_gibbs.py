# =================================================================
# Created by:
# Felipe Uribe @ DTU
# =================================================================
# Version 2022
# =================================================================
import time
import numpy as np
import scipy as sp
import scipy.stats as sps
import pickle
# import sksparse
# from sksparse.cholmod import cholesky

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

from image_deblurring_1D import A, Gauss, Defocus
from samplers import Gibbs_Horseshoe_all

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
Aop = lambda x, flag: A(x, flag, P, 'reflect') # wrap

# ee = np.eye(d)
# A_mat = np.empty((d,d))
# for i in range(d):
#     A_mat[:,i] = Aop(ee[:,i], 1)

# B = A_mat.T @ A_mat
# plt.figure()
# plt.plot(np.diag(B))
# plt.show()

# =================================================================
# load data and set-up likelihood
# =================================================================
# load truth and noisy convolved data
m = np.copy(d)
with open('data/data_m128_low.pkl', 'rb') as f:
    xx, f_true, g_true, e, y_data, sigma_obs = pickle.load(f)
# f_true += 1e-7
norm_f = np.linalg.norm(f_true)
lambd_obs = 1/(sigma_obs**2)
# Lambda_obs = sp.sparse.diags(lambd_obs*np.ones(d))

# ===================================================================
# MRF prior
# ===================================================================
mu_pr_x = np.zeros(d)   # prior mean

# 1D finite difference matrix with Neumann BCs
D = sp.sparse.spdiags([-np.ones(d), np.ones(d)], [0, 1], d, d).tocsr()
D[-1, -1] = 0

# for x: least-squares form
def proj_forward_reg(x, flag, tau, sigma, lambd):
    # regularized ASTRA projector [A; Lsq @ W] w.r.t. x.
    Wsq = sp.sparse.diags(1/(tau*sigma))
    if flag == 1:
        out1 = np.sqrt(lambd) * Aop(x, 1) # A @ x
        out2 = (Wsq @ D) @ x
        out = np.hstack([out1, out2])
    else:
        out1 = np.sqrt(lambd) * Aop(x[:m], 2) # A.T @ b
        out2 = (Wsq @ D).T @ x[m:]
        out = out1 + out2
    return out

# ===================================================================
# Gibbs
# ===================================================================
np.random.seed(8)
Nc = int(1e4)
Nb = int(0.2*Nc)
Nt = 20
#
tic = time.time()
x_s, lambd_s, sigma2_s, tau2_s, xi_s, gamma_s, ite = Gibbs_Horseshoe_all(Nc, Nb, Nt, d, D, Aop,\
    proj_forward_reg, y_data, 1, 1000)
toc = time.time()-tic
print('\nElapsed time\n:', toc)   # Nc2e4, Nt10 - 6271.46
#
tau_s = np.sqrt(tau2_s)
sigma_s = np.sqrt(sigma2_s)
ite = np.asarray(ite)

with open('deconv1D_low_HSnu1_Nc1e4_Nb2e3_Nt2e1_cglsfixafterNb_tau01_precond.pkl', 'wb') as f:
    pickle.dump([x_s, lambd_s, sigma_s, tau_s, xi_s, gamma_s, ite], f)
# HS 16117.02
# HS2 10106.33