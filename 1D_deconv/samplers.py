# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2021
# ========================================================================
import sys
import numpy as np
import scipy as sp
import scipy.stats as sps
from numpy import linalg as LA
eps = 1e-16
epsm = 1e16

import sksparse
from sksparse.cholmod import cholesky

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams.update({'font.size': 20})
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# matplotlib.rcParams['text.usetex'] = True

# ===================================================================
def Gibbs_Horseshoe_all(Nc, Nb, Nt, d, D, A, LS_fun, y_data, nu=1, \
    n_cgls=50, x_tol=1e-3, priorcond=True, analytic=False, nfix=True):
    K = int((Nt*Nc) + Nb)   # total number of Gibbs iterations
    m = len(y_data)

    # ===== variance hyperparams ==========
    # for sigma2_obs:
    alpha_2 = (m/2) + 1
    def pi_sigma2obs_rnd(misfit):        
        beta_2 = 0.5*(misfit.T @ misfit) + 1e-4 # rate, 0.5*np.linalg.norm(misfit)**2 + 1e-4
        # sps.invgamma.rvs(a=alpha_2, loc=0, scale=beta_2)
        # 1/np.random.gamma(shape=alpha_2, scale=1/beta_2)
        return sps.invgamma.rvs(a=alpha_2, loc=0, scale=beta_2)

    # for tau2:
    alpha_3 = (d+nu)/2
    def pi_tau2_rnd(x, sigma2, gamma):
        beta_3 = (nu/gamma) + 0.5*np.sum(((D @ x)**2)/sigma2)
        # if (beta_3<eps):
        #     beta_3 = eps
        # sps.invgamma.rvs(a=alpha_3, loc=0, scale=beta_3) #
        # 1/np.random.gamma(shape=alpha_3, scale=1/beta_3)
        return sps.invgamma.rvs(a=alpha_3, loc=0, scale=beta_3)

    # for sigma2:
    alpha_4 = ((nu + 1)/2) * np.ones(d)
    def pi_sigma2_rnd(x, tau2, xi):
        beta_4 = (nu/xi) + (((D @ x)**2)/(2*tau2))
        # if any(beta_4<eps):
        #     idx = beta_4<eps
        #     beta_4[idx] = eps
        # if any(beta_4>epsm):
        #     idx = beta_4>eps
        #     beta_4[idx] = epsm
        # 1/np.random.gamma(shape=alpha_4, scale=1/beta_4)
        return sps.invgamma.rvs(a=alpha_4, loc=np.zeros(d), scale=beta_4)
    
    # ===== auxiliary hyperparams ==========
    # for gamma:
    alpha_5 = (nu + 1)/2
    def pi_gamma_rnd(tau2, tau_02=1):
        beta_5 = 1/tau_02 + nu/tau2
        if (beta_5>epsm):# or (beta_5==np.nan):
            beta_5 = epsm
        # sps.invgamma.rvs(a=alpha_5, loc=0, scale=beta_5)#
        # 1/np.random.gamma(shape=alpha_5, scale=1/beta_5)
        return sps.invgamma.rvs(a=alpha_5, loc=0, scale=beta_5)

    # for xi:
    alpha_6 = ((nu + 1)/2) * np.ones(d)
    def pi_xi_rnd(sigma2):
        beta_6 = 1 + nu/sigma2
        if any(beta_6>epsm):
            idx = beta_6>epsm
            beta_6[idx] = epsm
        # np.array([sps.invgamma.rvs(a=alpha_6, loc=0, scale=beta_6[i]) for i in range(d)])
        # 1/np.random.gamma(shape=alpha_6, scale=1/beta_6)
        return sps.invgamma.rvs(a=alpha_6, loc=np.zeros(d), scale=beta_6)

    # ===== allocation ==========
    x_s = np.empty((d, Nc))
    lambd_s = np.zeros(Nc)
    sigma2_s = np.empty((d, Nc))
    tau2_s = np.zeros(Nc)
    xi_s = np.empty((d, Nc))
    gamma_s = np.zeros(Nc)
    cgls_it = list()

    # initial state
    x_kp1 = 0.5*np.ones(d)  # np.random.normal(loc=0.5, scale=0.1, size=d)
    lambd_kp1 = 100
    sigma2_kp1 = abs(sps.t.rvs(2, scale=1, size=d))
    tau2_kp1 = 1
    xi_kp1 = np.ones(d)
    gamma_kp1 = 1
    
    # init priorconditioner
    if priorcond:
        # W = sp.sparse.diags(1/(tau2_kp1*sigma2_kp1))
        # L = D.T @ (W @ D)
        L = sp.sparse.diags(1/np.sqrt(tau2_kp1*sigma2_kp1), format='csc') @ D # cholesky factor of the precision matrix
        C = sp.sparse.linalg.inv(L)
    if analytic:
        ee = np.eye(d)
        A_mat = np.empty((d, d))
        for i in range(d):
            A_mat[:, i] = A(ee[:, i], 1)
        B = sp.sparse.csc_matrix(A_mat.T @ A_mat)

    # MCMC
    i = 0
    for k in range(K+1):
        # ===X========================================================
        # sample x
        if analytic:
            Q_pr = D.T @ (sp.sparse.diags(1/(tau2_kp1*sigma2_kp1)) @ D)
            Q_pos = lambd_kp1*B + Q_pr
            L_chol = cholesky(Q_pos, ordering_method='natural')
            mu_pos = L_chol(lambd_kp1*A(y_data, 2))
            #
            u = np.random.randn(d)
            x_kp1 = mu_pos + L_chol.solve_Lt(u, use_LDLt_decomposition=False) 
            cgls_it.append(0)
        else:
            G_fun = lambda x, flag: LS_fun(x, flag, np.sqrt(tau2_kp1), np.sqrt(sigma2_kp1), lambd_kp1)
            if priorcond:
                x_kp1, it = plinear_RTO(x_kp1, G_fun, y_data, lambd_kp1, C, n_cgls, x_tol)
            else:
                x_kp1, it = linear_RTO(x_kp1, G_fun, y_data, lambd_kp1, n_cgls, x_tol)          
            cgls_it.append(it)
        # e_x = np.linalg.norm(x_kp1-x_truef)/xt_norm
        #
        # plt.plot(k, it, 'b.')
        # plt.pause(0.01)

        # ===lambda_obs====================================================
        misfit = y_data - A(x_kp1, 1)
        sigma2obs_kp1 = pi_sigma2obs_rnd(misfit)
        lambd_kp1 = 1/sigma2obs_kp1  # pi_lambd_rnd(misfit)

        # ===tau2===============================================
        tau2_kp1 = pi_tau2_rnd(x_kp1, sigma2_kp1, gamma_kp1)

        # ===sigma2====================================================
        sigma2_kp1 = pi_sigma2_rnd(x_kp1, tau2_kp1, xi_kp1)

        # === update priorconditioner
        if priorcond:
            # W = sp.sparse.diags(1/(tau2_kp1*sigma2_kp1))
            # L = (D.T @ (W @ D))
            L = sp.sparse.diags(1/np.sqrt(tau2_kp1*sigma2_kp1), format='csc') @ D
            C = sp.sparse.linalg.inv(L)

        # ===gamma===============================================
        gamma_kp1 = pi_gamma_rnd(tau2_kp1)

        # ===xi====================================================
        xi_kp1 = pi_xi_rnd(sigma2_kp1)

        # msg
        if (k > Nb):
            if (i == 0) and (k==(Nb+1)) and (nfix==True):  # fix cgls
                n_cgls = int(np.mean(np.asarray(cgls_it)[-20:]))
                # withcond - (1e-3: ) (1e-4: 110) (1e-5: ) (1e-6: )
                # nocond   - (1e-3: 133) (1e-4: 212) (1e-5: 238) (1e-6: 351)
                # (1e-3: 1) (1e-4: 1) (1e-5: 10) (1e-6: 84)
            # thinning
            if (np.mod(k, Nt) == 0):
                x_s[:, i] = x_kp1
                lambd_s[i] = lambd_kp1
                sigma2_s[:, i] = sigma2_kp1
                tau2_s[i] = tau2_kp1
                xi_s[:, i] = xi_kp1
                gamma_s[i] = gamma_kp1
                i += 1
                if (np.mod(i, 100) == 0):
                    print("\nSample {:d}/{:d}".format(i, Nc))
        else:
            if (k == 0):
                print("\nBurn-in... {:d} samples\n".format(Nb))
                # print('\t relerr so far', e_x[k+1])

    return x_s, lambd_s, sigma2_s, tau2_s, xi_s, gamma_s, cgls_it


# =========================================================================
# =========================================================================
# =========================================================================
def plinear_RTO(x_old, G_fun, b_meas, lambd, C, x_maxit, x_tol):
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(nbar)]) + \
        np.random.randn(m+nbar)
    
    # define priorconditioning operation
    #L_chol = cholesky(L, ordering_method='natural')
    def apply_Pinv(x, flag):
        if flag == 1:
            precond = C @ x #sp.sparse.linalg.spsolve(L, x)
            # precond = L_chol.solve_A(x, use_LDLt_decomposition=False) 
        elif flag == 2:
            precond = C.T @ x #sp.sparse.linalg.spsolve(L.T, x)
            # precond = L_chol.solve_At(x, use_LDLt_decomposition=False) 
        return precond  
    x_next, it = pcgls(G_fun, g, x_old, apply_Pinv, x_maxit, x_tol)
    
    return x_next, it
# =========================================================================
def linear_RTO(x_old, G_fun, b_meas, lambd, x_maxit, x_tol):
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(nbar)]) + \
        np.random.randn(m+nbar)
    x_next, it = cgls(G_fun, g, x_old, x_maxit, x_tol)
    #sp.optimize.least_squares(lambda x: G_fun(x,1)-g, x_old)[0]

    return x_next, it
# =========================================================================
def linear_RTOw(x_old, G_fun, b_meas, lambd, x_maxit, x_tol):
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(nbar)]) + np.random.randn(m+nbar)
    x_next, it = cgls(G_fun, g, x_old, x_maxit, x_tol)
    # res = sp.optimize.lsq_linear(G_fun, g, tol=x_tol, max_iter=x_maxit)
    # x_next, it = res.x, res.nit
    
    return x_next, it
# =========================================================================
# @njit
def pcgls(A, b, x0, apply_Pinv, maxit, tol):
    # http://web.stanford.edu/group/SOL/software/cgls/

    # initial state
    x = x0.copy()
    r = b - A(x, 1)
    s = apply_Pinv(A(r, 2), 2)  # P^{-T} * A^{T} * r
    p = s.copy()
     
    # initialization
    norms0 = LA.norm(s) # LA.norm(A(b, 2))#
    gamma = norms0**2
    normx = LA.norm(x)
    xmax = normx.copy()

    # main loop
    k, flag, indefinite = 0, 0, 0
    while (k < maxit) and (flag == 0):
        k += 1
        #
        t = apply_Pinv(p, 1) # P^{-1} p
        q = A(t, 1)
        #
        delta = LA.norm(q)**2  # + shift*LA.norm(p)**2
        if (delta <= 0):
            indefinite = 1
        if (delta == 0):
            delta = 1e-16
        alpha = gamma / delta
        #
        x += alpha*t # maybe this is alpha*q
        r -= alpha*q
        s = apply_Pinv(A(r, 2), 2)
        #
        norms = LA.norm(s)
        gamma1 = gamma.copy()
        gamma = norms**2
        beta = gamma / gamma1
        p = s + beta*p

        # convergence
        # flag = 1: CGLS converged to the desired tolerance TOL within MAXIT
        normx = LA.norm(x)
        xmax = max(xmax, normx)
        # relerr = LA.norm(x - xold) / normx
        flag = (norms <= norms0*tol) or (normx*tol >= 1) #or (relerr <= tol)
        # resNE = norms / norms0
    #
    shrink = normx/xmax
    if (k == maxit):
        flag = 2   # CGLS iterated MAXIT times but did not converge
    if indefinite:
        flag = 3   # Matrix (A'*A + delta*L) seems to be singular or indefinite
        sys.exit('\n Negative curvature detected !')
    if (shrink <= np.sqrt(tol)):
        # Instability likely: (A'*A + delta*L) indefinite and NORM(X) decreased
        flag = 4
        sys.exit('\n Instability likely !')

    return x, k
# =========================================================================
# @njit
def cgls(A, b, x0, maxit, tol):
    # http://web.stanford.edu/group/SOL/software/cgls/

    # initial state
    x = x0.copy()
    r = b - A(x, 1)
    s = A(r, 2)  # - shift*x

    # initialization
    p = s.copy()
    norms0 = LA.norm(s)
    gamma = norms0**2
    normx = LA.norm(x)
    xmax = normx.copy()

    # main loop
    k, flag, indefinite = 0, 0, 0
    while (k < maxit) and (flag == 0):
        k += 1
        # xold = np.copy(x)
        #
        q = A(p, 1)
        delta = LA.norm(q)**2  # + shift*LA.norm(p)**2
        #
        if (delta <= 0):
            indefinite = 1
        if (delta == 0):
            delta = 1e-16
        alpha = gamma / delta
        #
        x += alpha*p
        r -= alpha*q
        s = A(r, 2)  # - shift*x
        #
        norms = LA.norm(s)
        gamma1 = gamma.copy()
        gamma = norms**2
        beta = gamma / gamma1
        p = s + beta*p

        # convergence
        # flag = 1: CGLS converged to the desired tolerance TOL within MAXIT
        normx = LA.norm(x)
        xmax = max(xmax, normx)
        # relerr = LA.norm(x - xold) / normx
        flag = (norms <= norms0*tol) or (normx*tol >= 1) #or (relerr <= tol)
        # resNE = norms / norms0
    #
    shrink = normx/xmax
    if (k == maxit):
        flag = 2   # CGLS iterated MAXIT times but did not converge
    if indefinite:
        flag = 3   # Matrix (A'*A + delta*L) seems to be singular or indefinite
        sys.exit('\n Negative curvature detected !')
    if (shrink <= np.sqrt(tol)):
        # Instability likely: (A'*A + delta*L) indefinite and NORM(X) decreased
        flag = 4
        sys.exit('\n Instability likely !')

    return x, k
