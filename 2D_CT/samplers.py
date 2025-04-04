# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2021
# ========================================================================
import sys
import numpy as np
import numpy.linalg as LA
import scipy as sp
import scipy.stats as sps
# from scipy.sparse.linalg import lsqr
eps = 1e-7
epsm = 1e12

# import sparseqr
import sksparse
from sksparse.cholmod import cholesky
# from direct_sampling import sampler_CG, sampler_squareRootApprox
import matplotlib.pyplot as plt
# matplotlib.rcParams.update({'font.size': 20})
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# matplotlib.rcParams['text.usetex'] = True

# =========================================================================
# =========================================================================
# ===================================================================
def Gibbs_Horseshoe_all(Nc, Nb, Nt, L, Amat, y_data, nu=1, analytic=True):
    K = int((Nt*Nc) + Nb)   # total number of Gibbs iterations
    kd, d = L.shape
    m = len(y_data)

    # ===== variance hyperparams ==========
    # for sigma2_obs:
    alpha_2 = (m/2) + 1
    def pi_sigma2obs_rnd(misfit):        
        beta_2 = 0.5*(misfit.T @ misfit) + 1e-4 # rate, 0.5*np.linalg.norm(misfit)**2 + 1e-4
        # 1/np.random.gamma(shape=alpha_2, scale=1/beta_2)
        return sps.invgamma.rvs(a=alpha_2, loc=0, scale=beta_2)

    # for tau2:
    alpha_3 = (kd+nu)/2
    def pi_tau2_rnd(x, w2, gamma):
        beta_3 = (nu/gamma) + 0.5*np.sum(((L @ x)**2)/w2)
        if (beta_3<eps):
            beta_3 = eps
        tau2 = sps.invgamma.rvs(a=alpha_3, loc=0, scale=beta_3)
        # 1/np.random.gamma(shape=alpha_3, scale=1/beta_3)
        return tau2

    # for w2:
    alpha_4 = ((nu+1)/2) * np.ones(kd)
    def pi_w2_rnd(x, tau2, xi):
        beta_4 = (nu/xi) + (((L @ x)**2)/(2*tau2))
        if any(beta_4<eps):
            idx = np.argwhere(beta_4<eps)
            beta_4[idx] = eps
        if any(beta_4>epsm):
            idx = np.argwhere(beta_4>epsm)
            beta_4[idx] = epsm
        w2 = sps.invgamma.rvs(a=alpha_4, loc=np.zeros(kd), scale=beta_4)
        # 1/np.random.gamma(shape=alpha_4, scale=1/beta_4)
        return w2

    # ===== auxiliary hyperparams ==========
    # for gamma:
    alpha_5 = (nu+1)/2
    def pi_gamma_rnd(tau2, tau_02):
        beta_5 = 1/tau_02 + nu/tau2
        if (beta_5>epsm):# or (beta_5==np.nan):
            beta_5 = epsm
        gamma = sps.invgamma.rvs(a=alpha_5, loc=0, scale=beta_5) 
        # 1/np.random.gamma(shape=alpha_5, scale=1/beta_5)
        return gamma

    # for xi:
    alpha_6 = ((nu+1)/2) * np.ones(kd)
    def pi_xi_rnd(w2):
        beta_6 = 1 + nu/w2
        if any(beta_6>epsm):
            idx = np.argwhere(beta_6>epsm)
            beta_6[idx] = epsm
        xi = sps.invgamma.rvs(a=alpha_6, loc=np.zeros(kd), scale=beta_6)
        # 1/np.random.gamma(shape=alpha_6, scale=1/beta_6)
        return xi

    # ===== allocation ==========
    x_s = np.empty((d, Nc))
    lambd_s = np.zeros(Nc)
    w2_s = np.empty((kd, Nc))
    tau2_s = np.zeros(Nc)
    xi_s = np.empty((kd, Nc))
    gamma_s = np.zeros(Nc)
    cgls_it = list()

    # initial state
    x_kp1 = 0.5*np.ones(d)  # np.random.normal(loc=0.5, scale=0.1, size=d)
    lambd_kp1 = 10
    w2_kp1 = abs(sps.t.rvs(2, scale=1, size=kd))
    tau2_kp1 = 0.1
    xi_kp1 = np.ones(kd)
    gamma_kp1 = 1

    # operator preprocessing
    if analytic:
        ATA = sp.sparse.csc_matrix(Amat.T @ Amat)
        const = Amat.T @ y_data # A(y_data, 2) 

    # MCMC
    i = 0
    for k in range(K+1):
        # ===X========================================================
        # sample x
        if analytic:
            w12 = 1/(tau2_kp1*w2_kp1) 
            # if any(w12>epsm):
            #     idx = np.argwhere(w12>epsm)
            #     w12[idx] = epsm
            Q_pr = L.T @ (sp.sparse.diags(w12) @ L)
            Q_pos = ATA.multiply(lambd_kp1) + Q_pr
            L_chol = cholesky(Q_pos, ordering_method='natural')
            #
            v = L_chol.solve_L(lambd_kp1*const, use_LDLt_decomposition=False)
            m_pos = L_chol.solve_Lt(v, use_LDLt_decomposition=False)
            w = L_chol.solve_Lt(np.random.randn(d), use_LDLt_decomposition=False)
            x_kp1 = m_pos + w            
            # mu_pos = L_chol(lambd_kp1*const)
            # x_kp1 = mu_pos + L_chol.solve_Lt(np.random.randn(d), use_LDLt_decomposition=False) 
            it = 0
        cgls_it.append(it)
        # plt.figure(1)
        # plt.imshow(x_kp1.reshape(32,32), extent=[0, 1, 0, 1], aspect='equal', cmap='YlGnBu_r')
        # plt.pause(0.1)
        
        # ===lambda_obs====================================================
        misfit = y_data - (Amat@x_kp1) # A(x_kp1, 1)
        sigma2obs_kp1 = pi_sigma2obs_rnd(misfit)
        lambd_kp1 = 1/sigma2obs_kp1# 8.6030641
        # print(lambd_kp1)

        # ===tau2===============================================
        tau2_kp1 = pi_tau2_rnd(x_kp1, w2_kp1, gamma_kp1)
        # print(np.sqrt(tau2_kp1))

        # ===w2====================================================
        w2_kp1 = pi_w2_rnd(x_kp1, tau2_kp1, xi_kp1)

        # ===gamma===============================================
        gamma_kp1 = pi_gamma_rnd(tau2_kp1, sigma2obs_kp1)#1/lambd_kp1

        # ===xi====================================================
        xi_kp1 = pi_xi_rnd(w2_kp1)

        # msg
        if (k > Nb):
            # thinning
            if (np.mod(k, Nt) == 0):
                x_s[:, i] = x_kp1
                lambd_s[i] = lambd_kp1
                w2_s[:, i] = w2_kp1
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

    return x_s, lambd_s, w2_s, tau2_s, xi_s, gamma_s, cgls_it


# =========================================================================
# =========================================================================
# =========================================================================
def plinear_RTO(x_old, G_fun, b_meas, lambd, L, x_maxit, x_tol):
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(2*nbar)]) + \
        np.random.randn(m+2*nbar)
    
    # define priorconditioning operation
    #L_chol = cholesky(L, ordering_method='natural')
    def apply_Pinv(x, flag):
        if flag == 1:
            precond = sp.sparse.linalg.spsolve(L, x)
            # precond = L.solve_A(x, use_LDLt_decomposition=False) 
        elif flag == 2:
            precond = sp.sparse.linalg.spsolve(L.T, x)
            # precond = L.solve_At(x, use_LDLt_decomposition=False) 
        return precond  
    x_next, it = pcgls(G_fun, g, x_old, apply_Pinv, x_maxit, x_tol)
    
    return x_next, it
# =========================================================================
def linear_RTO(x_old, G_fun, b_meas, lambd, x_maxit, x_tol):
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(2*nbar)]) + \
                    np.random.randn(m+2*nbar)
    x_next, it = cgls(G_fun, g, x_old, x_maxit, x_tol)
    #
    # def Lq_fun(x):
    #     r = G_fun(x, 1) - g
    #     return r
    # def Jac(x): 
    #     J = sp.sparse.vstack([np.sqrt(lambd_obs)*A_mat, C])
    #     return J   
    # opt = sp.optimize.least_squares(Lq_fun, x_old, Jac, max_nfev=x_maxit)
    # x_next, it = opt.x, opt.nfev
    #
    # x, istop, itn, normr = lsqr(G_fun, g)[:4]

    return x_next, it
# =========================================================================
def linear_RTOw(x_old, G_fun, b_meas, lambd, x_maxit, x_tol):
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(2*nbar)]) + np.random.randn(m+2*nbar)
    x_next, it = cgls(G_fun, g, x_old, x_maxit, x_tol)

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
