# =================================================================
# Created by:
# Felipe Uribe @ DTU
# =================================================================
# Version 2021-01
# =================================================================
import time
import numpy as np
import scipy as sp
import scipy.io as spio
import pickle

import h5py
import hdf5storage
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 25})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

from projection_functions import A, Amat
from samplers import Gibbs_Horseshoe_all

# =================================================================
# problem parameters
# =================================================================
# N = 32           # object size N-by-N pixels
# p = 32    # number of detector pixels
# q = 16            # number of projection angles
N = 64           # object size N-by-N pixels
p = 64           # number of detector pixels
q = 32            # number of projection angles
# N = 150           # object size N-by-N pixels
# q = 90            # number of projection angles
# p = int(1.5*N)    # number of detector pixels
theta = np.linspace(0, 2*np.pi, q, endpoint=False)

# =================================================================
# load data and set-up likelihood
# =================================================================
# 'b_meas','b_meas_i','std_e','x_true_fine','t_true'
data = spio.loadmat('data/data_true_grains_x1_N64_noi0p02.mat')

# underlying true
x_true = data['x_true_fine']
x_truef = x_true.flatten(order='F')
xt_norm = np.linalg.norm(x_truef)

# noise and data
m = q*p                          # number of data points
y_data = data['y_data'].flatten()
B_data = data['B_data']
sigma_obs = data['std_e'].flatten()
lambd_obs = 1/(sigma_obs**2)

# ===================================================================
# MRF prior: difference priors
# ===================================================================
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

# structure matrix
# L = D1.T@D1 + D2.T@D2 
L = sp.sparse.vstack([D1, D2], format='csc', dtype=int) # (2d x d)

# =================================================================
# CONDITIONALS
# =================================================================
# for x: least-squares form
def proj_forward_reg(x, flag, tau, sigma, lambd):
    # regularized ASTRA projector [A; Lsq @ W] w.r.t. x.
    Wsq = sp.sparse.diags(1/(tau*sigma))
    if flag == 1:
        out1 = np.sqrt(lambd) * A(x, 1)
        out2 = (Wsq @ L) @ x
        out = np.hstack([out1, out2])
    else:
        out1 = np.sqrt(lambd) * A(x[:m], 2)
        out2 = (Wsq @ L).T @ x[m:]
        out = out1 + out2
    return out

# ===================================================================
# Gibbs
# ===================================================================
np.random.seed(1)
Nc = int(1e4)
Nb = int(0.1*Nc)
Nt = 20
#
tic = time.time()
x_s, lambd_s, w2_s, tau2_s, xi_s, gamma_s, ite = Gibbs_Horseshoe_all(Nc, Nb, Nt, L, Amat(), y_data, nu=1)
toc = time.time()-tic
print('\nElapsed time\n:', toc) #344028.7 s
#
tau_s = np.sqrt(tau2_s)
w_s = np.sqrt(w2_s)
ite = np.asarray(ite)
#
# mdict = {'x_s':x_s, 'lambd_s':lambd_s, 'w_s':w_s, 'tau_s':tau_s, 'xi_s':xi_s, 'gamma_s':gamma_s, 'ite':ite}
# hdf5storage.write(mdict, '.', 'CT2D_N64_low_HSnu1_Nc1e4_Nb1e3_Nt2e1_analytic.mat', matlab_compatible=True)
with open('CT2D_N64_low_HSnu1_Nc1e4_Nb1e3_Nt2e1_analytic.pkl', 'wb') as f:
    pickle.dump([x_s, lambd_s, w_s, tau_s, xi_s, gamma_s, ite], f)
# mdict = {'X':x_s, 'err_x':e_x, 'prec_obs':lambd_s, 'prec_x':delta_s}
# hdf5storage.write(mdict, '.', 'grainsx2_FIX_ns10_new.mat', matlab_compatible=True)
# spio.savemat('grainsx1_UQ_ns10.mat', mdict)