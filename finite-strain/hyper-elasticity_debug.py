import numpy as np
import scipy.sparse.linalg as sp
import itertools

# ----------------------------------- GRID ------------------------------------

ndim   = 3   # number of dimensions
# N      = 31  # number of voxels (assumed equal for all directions)
#N      = 3  # Y: mini grid
Nx = 3
Ny = 3
Nz = 1
shape  = [Nx,Ny,Nz]  # number of voxels as list: [Nx,Ny,Nz]

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ijxyz          ->jixyz  ',A2   )
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz',A4,B4)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)
dot11  = lambda A1,B1: np.einsum('ixyz   ,ixyz   ->xyz    ',A1,B1)

# identity tensor                                               [single tensor]
i      = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones(shape))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones(shape))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones(shape))
I4s    = (I4+I4rt)/2.
II     = dyad22(I,I)

# ------------------- use FFT define of elast-plast example -------------------
# projection operator (only for non-zero frequency, associated with the mean)
# NB: vectorized version of "hyper-elasticity.py"
# - allocate / support function
Ghat4  = np.zeros([3,3,3,3,Nx,Ny,Nz])                # projection operator
x      = np.zeros([3      ,Nx,Ny,Nz],dtype='int64')  # position vectors
q      = np.zeros([3      ,Nx,Ny,Nz],dtype='int64')  # frequency vectors
delta  = lambda i,j: np.float_(i==j)                  # Dirac delta function
# - set "x" as position vector of all grid-points   [grid of vector-components]
x[0],x[1],x[2] = np.mgrid[:Nx,:Ny,:Nz]
# - convert positions "x" to frequencies "q"        [grid of vector-components]
for i in range(3):
    freq = np.arange(-(shape[i]-1)/2,+(shape[i]+1)/2,dtype='int64')
    q[i] = freq[x[i]]
# - compute "Q = ||q||", and "norm = 1/Q" being zero for the mean (Q==0)
#   NB: avoid zero division
q       = q.astype(np.float_)
Q       = dot11(q,q)
Z       = Q==0
Q[Z]    = 1.
norm    = 1./Q
norm[Z] = 0.
# - set projection operator                                   [grid of tensors]
for i, j, l, m in itertools.product(range(3), repeat=4):
    Ghat4[i,j,l,m] = norm*delta(i,m)*q[j]*q[l]

# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[Nx,Ny,Nz]))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[Nx,Ny,Nz]))

# functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_dF   = lambda dFm: trans2(ddot42(K4,trans2(dFm.reshape(3,3,Nx,Ny,Nz))))
G_K_dF = lambda dFm: G(K_dF(dFm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)
# phase  = np.zeros([N,N,N]); phase[-9:,:9,-9:] = 1.
phase  = np.zeros(shape); phase[-1:,:1,-1:] = 1.  # Y: single vox incl
print(phase[:,0,:])
print(phase[:,1,:])
print(phase[:,2,:])
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones(shape)*(1.-phase)+M1*np.ones(shape)*phase
K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]

# constitutive model: grid of "F" -> grid of "P", "K4"        [grid of tensors]
def constitutive(F):
    C4 = K*II+2.*mu*(I4s-1./3.*II)
    S  = ddot42(C4,.5*(dot22(trans2(F),F)-I))
    P  = dot22(F,S)
    K4 = dot24(S,I4)+ddot44(ddot44(I4rt,dot42(dot24(F,C4),trans2(F))),I4rt)
    return P,K4

# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize deformation gradient, and stress/stiffness       [grid of tensors]
F     = np.array(I,copy=True)
P,K4  = constitutive(F)

# set macroscopic loading
DbarF = np.zeros([ndim,ndim,Nx,Ny,Nz]); DbarF[0,1] += 1.0

# initial residual: distribute "barF" over grid using "K4"
b     = -G_K_dF(DbarF)
F    +=         DbarF
print(F[:,:,0,0,0])
Fn    = np.linalg.norm(F)
iiter = 0

# iterate as long as the iterative update does not vanish
while True:
    dFm,_ = sp.cg(tol=1.e-8,
      A = sp.LinearOperator(shape=(F.size,F.size),matvec=G_K_dF,dtype='float'),
      b = b,
    )                                        # solve linear system using CG
    F    += dFm.reshape(ndim,ndim,Nx,Ny,Nz)     # update DOFs (array -> tens.grid)
    P,K4  = constitutive(F)                  # new residual stress and tangent
    b     = -G(P)                            # convert res.stress to residual
    print('%10.2e'%(np.linalg.norm(dFm)/Fn)) # print residual to the screen
    if np.linalg.norm(dFm)/Fn<1.e-5 and iiter>0: break # check convergence
    iiter += 1