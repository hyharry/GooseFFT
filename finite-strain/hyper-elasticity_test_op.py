import numpy as np
import scipy.sparse.linalg as sp
import itertools

# ----------------------------------- GRID ------------------------------------

ndim   = 3   # number of dimensions
# N      = 31  # number of voxels (assumed equal for all directions)
#N      = 3  # Y: mini grid
Nx = 3
Ny = 3
Nz = 3

# Nx = 5
# Ny = 5
# Nz = 5
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
phase  = np.zeros(shape); phase[1,1,1] = 1.  # Y: single vox incl at center of 3x3x3
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones(shape)*(1.-phase)+M1*np.ones(shape)*phase
# K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
# mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]
K      = param(70e9,70e8)  # bulk  modulus                   approx. Alu
mu     = param(25e9,25e8)  # shear modulus                   approx. Alu

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
DbarF = np.zeros([ndim,ndim,Nx,Ny,Nz])

ksp_i = [0] # hack for remember it_n during call
def ksp_callback(xk):
    ksp_i[0] += 1

##### bc_strain from DAMASK #####
dot_F = np.array([
    [0,0,0],
    [0,1e-3,0],
    [0,0,0]
])

##### bc_stress from DAMASK #####
barP = np.zeros([ndim,ndim,Nx,Ny,Nz])
dot_F = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0]
])

delta_P_0 = 2.5e6
delta_P_1 = 5.0e6
delta_P_2 = 2.5e6
delta_P = np.zeros((3,3))
delta_P[0,0] = delta_P_0
delta_P[1,1] = delta_P_1
delta_P[2,2] = delta_P_2
delta_P *= 100

##### adjust zeroth frequency -> for all stress component! ij
# for i, j, l, m in itertools.product(range(3), repeat=4):
#     Ghat4[i,j,l,m,Nx//2,Ny//2,Nz//2] = delta(i,m)*delta(j,l)

##### bc_mix from DAMASK #####
barP = np.zeros([ndim,ndim,Nx,Ny,Nz])
dot_F = np.array([
    [0,0,0],
    [0,1e-3,0],
    [0,0,0]
])

delta_P_0 = 0
delta_P_2 = 0
delta_P = np.zeros((3,3))
delta_P[0,0] = delta_P_0
delta_P[2,2] = delta_P_2

##### adjust zeroth frequency -> only change the component with ij defined by stress bc!!!
# for i, j, l, m in itertools.product(range(3), repeat=4):
#     if [i,j] == [0,0] or [i,j] == [2,2]:
#         # print(f'change Ghat(q=0) for Ghat[{i},{j},{l},{m}]')
#         # print(f'vor  Ghat[{i},{j},{l},{m}]={Ghat4[i,j,l,m,Nx//2,Ny//2,Nz//2]}')
#         Ghat4[i,j,l,m,Nx//2,Ny//2,Nz//2] = delta(i,m)*delta(j,l)
#         # print(f'nach Ghat[{i},{j},{l},{m}]={Ghat4[i,j,l,m,Nx//2,Ny//2,Nz//2]}')

##### bc_debug for DAMASK #####
barP = np.zeros([ndim,ndim,Nx,Ny,Nz])
dot_F = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0]
])

delta_P_0 = 2.5e6
delta_P_1 = 5.0e6
delta_P_2 = 2.5e6
delta_P = np.zeros((3,3))
delta_P[0,0] = delta_P_0
delta_P[1,1] = delta_P_1
delta_P[2,2] = delta_P_2
delta_P[0,1] = delta_P_0

##### adjust zeroth frequency -> for all stress component! ij
for i, j, l, m in itertools.product(range(3), repeat=4):
    if [i,j] == [0,0] or [i,j] == [1,1] or [i,j] == [2,2] or [i,j] == [0,1]:
        Ghat4[i,j,l,m,Nx//2,Ny//2,Nz//2] = delta(i,m)*delta(j,l)

##### debug purpose (DAMASK G_conv) #####
def G_dbg(A2, M_bc):
    temp = ddot42(Ghat4,fft(A2))
    temp[:,:,Nx//2,Ny//2,Nz//2] = M_bc
    G_field = np.real( ifft( temp ) ).reshape(-1)
    return G_field

G_K_dF_dbg = lambda dFm, M_bc : G_dbg(K_dF(dFm), M_bc)

#  def formResidual(F, P_aim):
#      """
#      F: field variable 3,3,nx,ny,nz
#      P_aim: 3,3
#      """
#      tmp = K_dF(F)

dummy_zero = np.zeros([ndim,ndim,Nx,Ny,Nz])

from functools import partial

check = lambda dP: np.einsum('ijkl,lk->ij', Ghat4[:,:,:,:,1,1,1].reshape((3,3,3,3)), dP)

# G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
# K_dF   = lambda dFm: trans2(ddot42(K4,trans2(dFm.reshape(3,3,Nx,Ny,Nz))))
# G_K_dF = lambda dFm: G(K_dF(dFm))

def K_G_dF(dF): # != G_K_dF(dF)
    G_dF = np.real( ifft( ddot42(Ghat4,fft(dF.reshape(ndim,ndim,Nx,Ny,Nz))) ) )
    result = K_dF(G_dF).reshape(-1)
    return result

pairs = {0: (0, 0), 1: (1, 1), 2: (2, 2),
         3: (1, 2), 4: (2, 0), 5: (0, 1),
         6: (2, 1), 7: (0, 2), 8: (1, 0)}
inv_pairs = {v: k for k, v in pairs.items()}

def tensor_2_m99(C_3333):
    M = np.zeros((9,9))
    for i,j,k,l in itertools.product(range(3), repeat=4):
        ij = inv_pairs[(i,j)]
        kl = inv_pairs[(k,l)]
        M[ij,kl] = C_3333[i,j,k,l]
    return M

def m99_2_tensor(M):
    C_3333 = np.zeros((3,3,3,3))
    for i,j,k,l in itertools.product(range(3), repeat=4):
        ij = inv_pairs[(i,j)]
        kl = inv_pairs[(k,l)]
        M[ij,kl] = C_3333[i,j,k,l]
    return C_3333

def tensor_flat_inv(vec):
    C_3333 = vec.reshape((3,3,3,3))
    M = tensor_2_m99(C_3333)
    M_inv = np.linalg.inv(M)
    # M_inv = np.linalg.pinv(M)
    result = m99_2_tensor(M_inv)
    return result

def tensor_flat_det(vec):
    C_3333 = vec.reshape((3,3,3,3))
    M = tensor_2_m99(C_3333)
    result = np.linalg.det(M)
    return result

def tensor_flat_2_m99(vec):
    C_3333 = vec.reshape((3,3,3,3))
    result = tensor_2_m99(C_3333)
    return result

def reshape_K(K4):
    K_tmp = K4.reshape(-1,Nx*Ny*Nz)
    result = np.apply_along_axis(tensor_flat_2_m99,0,K_tmp)
    result = result.reshape((9,9,Nx,Ny,Nz))
    return result

def inv_K(K4):
    K_tmp = K4.reshape(-1,Nx*Ny*Nz)
    result = np.apply_along_axis(tensor_flat_inv,0,K_tmp)
    result = result.reshape((ndim,ndim,ndim,ndim,Nx,Ny,Nz))
    return result

def det_K(K4):
    K_tmp = K4.reshape(-1,Nx*Ny*Nz)
    result = np.apply_along_axis(tensor_flat_det,0,K_tmp)
    result = result.reshape((-1,Nx,Ny,Nz))
    return result

#print(K4.shape)

# precond = inv_K(K4)
# apply_precond = lambda dF: np.einsum('ijklxyz,klxyz ->ijxyz',precond,dF)
# 
# G_K_dF_x = lambda dF: G_K_dF(apply_precond(dF))

t = 0.4
N = 8 
dt = t/N

print("##################### sim run start #####################")

for inc in range(N):
    print(f"------------- inc {inc+1} ------------")
    DbarF_curr = dummy_zero + dt * dot_F[:,:,np.newaxis,np.newaxis,np.newaxis]
    barP_curr = barP + (inc+1) * delta_P[:,:,np.newaxis,np.newaxis,np.newaxis]

    # initial residual: distribute "barF" over grid using "K4"
    ### !!! this is bad order, bad initi F !!! ###
    # F    +=         DbarF_curr
    # b     = -G_K_dF(F) + G(barP_curr)
    ##############################################
    ## op 1
    # b     = -G_K_dF(DbarF_curr) + G(barP_curr)
    # F    +=         DbarF_curr
    ## op 2 -> better
    F    +=         DbarF_curr
    P,K4  = constitutive(F)
    b     = -G(P) + G(barP_curr)
    Fn    = np.linalg.norm(F)
    newton_i = 0
    ksp_i = [0]

    # tmp = det_K(K4)
    # print(np.isclose(tmp, 0))
    if inc > 0:
        precond = inv_K(K4)
        apply_precond = lambda dF: np.einsum('ijklxyz,klxyz ->ijxyz',precond,dF.reshape((ndim,ndim,Nx,Ny,Nz)))
        G_K_dF_x = lambda dF: G_K_dF(apply_precond(dF))

    # iterate as long as the iterative update does not vanish
    while True:
        # if inc == 0:
        if True:
            dFm,_ = sp.cg(tol=1.e-8,
            #dFm,_ = sp.gmres(tol=1.e-8,
            A = sp.LinearOperator(shape=(F.size,F.size),matvec=G_K_dF,dtype='float'),
            b = b,
            callback=ksp_callback, #### cg counter
            )                                        # solve linear system using CG
            F    += dFm.reshape(ndim,ndim,Nx,Ny,Nz)  # update DOFs (array -> tens.grid)
        else:
            dFm,_ = sp.cg(tol=1.e-8,
            #dFm,_ = sp.gmres(tol=1.e-8,
            A = sp.LinearOperator(shape=(F.size,F.size),matvec=G_K_dF_x,dtype='float'),
            b = b,
            callback=ksp_callback, #### cg counter
            )                                        # solve linear system using CG
            F    += apply_precond(dFm.reshape(ndim,ndim,Nx,Ny,Nz))
        P,K4  = constitutive(F)                  # new residual stress and tangent
        # b     = -G(P)                            # convert res.stress to residual
        b     = -G(P) + G(barP_curr)

        # tmp = inv_K(K4)
        # print(tmp.shape)

        # tmp = K4.copy()
        # tmp = tmp.reshape(-1,Nx*Ny*Nz)
        # tmp = tmp.reshape(ndim,ndim,ndim,ndim,Nx,Ny,Nz)
        # print(f'after reshape = {np.linalg.norm(K4-tmp)}')
        
        # test = G_K_dF(dFm)
        # test_1 = K_G_dF(dFm)
        # print(f'|K_G - G_K| = {np.linalg.norm(test_1 - test)}')

        # ^^^^^^^^^ dbg purpose ^^^^^^^^^
        # b_mean = b.reshape(ndim,ndim,Nx,Ny,Nz).mean(axis=(2,3,4))
        # Pav_Paim = P.mean(axis=(2,3,4))-barP_curr.mean(axis=(2,3,4))
        # print('P_aim', barP_curr.mean(axis=(2,3,4)))
        # print('P_av', P.mean(axis=(2,3,4)))
        # b_dbg = -G_dbg(P-barP_curr, Pav_Paim)
        # print(np.linalg.norm(b - b_dbg))
        #print(check(P.mean(axis=(2,3,4))-barP_curr.mean(axis=(2,3,4)))/1e6)
        #print((P.mean(axis=(2,3,4))-barP_curr.mean(axis=(2,3,4)))/1e6)

        dF_norm_rel = np.linalg.norm(dFm)/Fn
        rhs_norm = np.linalg.norm(b)
        newton_i += 1
        print(f'newton {newton_i} end: |dF|/|F| = {dF_norm_rel:8.2e}, |rhs| = |G(P)| = {rhs_norm:8.2e}, ksp iter = {ksp_i[0]}')
        if np.linalg.norm(dFm)/Fn<1.e-5 : break # check convergence
    
    # print(f'current F_bar = \n{F.mean(axis=(2,3,4))}')
    print(f'current P_bar = \n{P.mean(axis=(2,3,4))}')
    print(f'=> load inc {inc+1} done with {newton_i} newton iter!')

print("##################### sim run done! #####################")
