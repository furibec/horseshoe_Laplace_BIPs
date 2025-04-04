# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2022
# ========================================================================
import numpy as np
import astra

#=========================================================================
# global: fixed parameters
#=========================================================================
# N = 150           # object size N-by-N pixels
# q = 90            # number of projection angles
# p = int(1.5*N)    # number of detector pixels
# N = 64           # object size N-by-N pixels
# p = 64    # number of detector pixels
# q = 32            # number of projection angles
N = 32           # object size N-by-N pixels
p = 32    # number of detector pixels
q = 16            # number of projection angles
theta = np.linspace(0, 2*np.pi, q, endpoint=False)

# problem setting
source_origin = 3*N                       # source origin distance [cm]
detector_origin = N                       # origin detector distance [cm]
detector_pixel_size = (source_origin+detector_origin)/source_origin
detector_length = detector_pixel_size*p   # detector length

# object dimensions
vol_geom = astra.create_vol_geom(N, N)

# =================================================================
# forward model
# ================================================================= 
def A(x, flag):
    if flag == 1:
        # forward projection
        return proj_forward_sino(x)
    elif flag == 2:
        # backward projection  
         return proj_backward_sino(x)

#=========================================================================
def proj_forward_sino(x):     
    # object 
    proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta, source_origin, detector_origin)
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom) # strip_fanflat
    
    # forward projection
    x = x.reshape((N, N))
    _, Ax = astra.create_sino(x, proj_id)   
    return Ax.flatten()

#=========================================================================
def proj_backward_sino(b):          
    # object 
    proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta, source_origin, detector_origin)
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom) 
    
    # backward projection   
    b = b.reshape((q, p))
    _, ATb = astra.create_backprojection(b, proj_id)
    return ATb.flatten()


#=========================================================================
def Amat():     
    # object 
    proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta, source_origin, detector_origin)
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom) # strip_fanflat
    
    # matrix
    mat_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(mat_id)
    
    #empty memory
    astra.projector.delete(proj_id)
    astra.matrix.delete(mat_id)
    return A