# ===================================================================
# Created by:
# Felipe Uribe @ DTU
# ===================================================================
# Version 2021-04
# ===================================================================
import numpy as np
import scipy as sp
from scipy.ndimage import convolve1d



# ===================================================================
# ===================================================================
# ===================================================================
# forward and adjoint functions
# ===================================================================
# ===================================================================
# ===================================================================
def A(X, flag, P, BC):
    if flag == 1:
        # forward projection
        return proj_forward(X, P, BC)
    elif flag == 2:
        # backward projection  
         return proj_backward(X, P, BC)

#=========================================================================
def proj_forward(X, P, BC):
    Ax = convolve1d(X, P, mode=BC) 
    # np.convolve(P, X, mode='same') or sp.signal.convolve1d(X, P)
    return Ax

#=========================================================================
def proj_backward(B, P, BC):
    P = P[::-1]#np.fliplr(P)
    ATy = convolve1d(B, P, mode=BC) # sp.signal.convolve1d(B_ext, P)
    return ATy



# ===================================================================
# ===================================================================
# ===================================================================
# Point spread functions
# ===================================================================
# ===================================================================
# ===================================================================
# Array with PSF for Moffat blur (astronomical telescope)
# ===================================================================
def Moffat(dim, s, beta):
    # Set up grid points to evaluate the Gaussian function
    x = np.arange(-np.fix(dim/2), np.ceil(dim/2))

    # Compute the Gaussian, and normalize the PSF.
    PSF = ( 1 + (x**2)/(s**2) )**(-beta)
    PSF = PSF / PSF.sum()

    # find the center
    center = np.where(PSF == PSF.max())[0][0]

    return PSF, center.astype(int)

# ===================================================================
# Array with PSF for Gaussian blur (astronomic turbulence)
# ===================================================================
def Gauss(dim, s):    
    # Set up grid points to evaluate the Gaussian function
    x = np.arange(-np.fix(dim/2), np.ceil(dim/2))

    # Compute the Gaussian, and normalize the PSF.
    PSF = np.exp( -0.5*((x**2)/(s**2)) )
    PSF /= PSF.sum()

    # find the center
    center = np.where(PSF == PSF.max())[0][0]

    return PSF, center.astype(int)

# ===================================================================
# Array with PSF for out-of-focus blur
# ===================================================================
def Defocus(dim, R):    
    center = np.fix(int(dim/2))
    if (R == 0):    
        # the PSF is a delta function and so the blurring matrix is I
        PSF = np.zeros(dim)
        PSF[center] = 1
    else:
        PSF = np.ones(dim) / (np.pi * R**2)
        k = np.arange(1, dim+1)
        aa = (k-center)**2
        idx = np.array((aa > (R**2)))
        PSF[idx] = 0
    PSF = PSF / PSF.sum()

    return PSF, center.astype(int)