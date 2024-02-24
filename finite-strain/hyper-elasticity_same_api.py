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

Ghat4_orig = Ghat4.copy()

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

##### adjust zeroth frequency -> for all stress component! ij
for i, j, l, m in itertools.product(range(3), repeat=4):
    if [i,j] == [0,0] or [i,j] == [1,1] or [i,j] == [2,2]:
        Ghat4[i,j,l,m,Nx//2,Ny//2,Nz//2] = delta(i,m)*delta(j,l)

##### debug purpose (DAMASK G_conv) #####
def G_dbg(A2, M_bc):
    temp = ddot42(Ghat4,fft(A2))
    temp[:,:,Nx//2,Ny//2,Nz//2] = M_bc
    G_field = np.real( ifft( temp ) ).reshape(-1)
    return G_field

G_K_dF_dbg = lambda dFm, M_bc : G_dbg(K_dF(dFm), M_bc)

def G_conv(P, P_aim):
    tmp = ddot42(Ghat4_orig,fft(P))
    tmp[:,:,Nx//2,Ny//2,Nz//2] = P_aim*Nx*Ny*Nz
    return np.real( ifft( tmp ) ).reshape(-1)


# G = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)

def formResidual(F, P_aim):
    """
    F: field variable 3,3,nx,ny,nz
    P_aim: 3,3
    """
    P, K_curr = constitutive(F)
    P_av = P.mean(axis=(2,3,4))
    return G_conv(P, P_av - P_aim), K_curr

def dbg_formResidual(F, P_aim):
    P, K4 = constitutive(F)
    ref = G(P - P_aim[:,:,np.newaxis,np.newaxis,np.newaxis])
    dam,_ = formResidual(F, P_aim)
    diff = np.linalg.norm(dam-ref)
    if diff < 1e-3:
        sig = 'OK'
    else:
        sig = 'not OK'
    print(f'curr diff for formResidual = {diff} \t {sig}')

# K_dF   = lambda dFm: trans2(ddot42(K4,trans2(dFm.reshape(3,3,Nx,Ny,Nz))))
# G_K_dF = lambda dFm: G(K_dF(dFm))

def formJacobian(dF, K_curr):
    dP = trans2(ddot42(K_curr,trans2(dF.reshape(3,3,Nx,Ny,Nz))))
    # dP = ddot42(K_curr,trans2(dF.reshape(3,3,Nx,Ny,Nz)))
    dP_with_bc = dP.mean(axis=(2,3,4)) ## !!!!!!! formJac should not set aim!
    return G_conv(dP, dP_with_bc)

def dbg_formJacobian(dF, K_curr):
    ref = G_K_dF(dF)
    # dP_with_bc = np.zeros((3,3))
    ref_dP_with_bc = ref.reshape(3,3,Nx,Ny,Nz).mean(axis=(2,3,4))
    # dP_with_bc = ref_dP_with_bc
    # __import__('pdb').set_trace()
    dam = formJacobian(dF, K_curr)
    diff = np.linalg.norm(dam-ref)
    if diff < 1e-3:
        sig = 'OK'
    else:
        sig = 'not OK'
    print(f'curr diff for formJacobian = {diff:.3e} \t {sig}')

F_1 = F.copy()
P_1 = P.copy()


from functools import partial

check = lambda dP: np.einsum('ijkl,lk->ij', Ghat4[:,:,:,:,1,1,1].reshape((3,3,3,3)), dP)

t = 0.4
N = 8 
dt = t/N

print("##################### sim run start #####################")

for inc in range(N):
    print(f"------------- inc {inc+1} ------------")
    DbarF_curr = DbarF + dt * dot_F[:,:,np.newaxis,np.newaxis,np.newaxis]
    barP_curr = barP + (inc+1) * delta_P[:,:,np.newaxis,np.newaxis,np.newaxis]
    barP_curr_1 = (inc+1) * delta_P

    # initial residual: distribute "barF" over grid using "K4"
    F    +=         DbarF_curr
    P,K4  = constitutive(F)
    # b     = -G(P) + G(barP_curr)
    b     = -G(P - barP_curr)
    # dbg_formResidual(F,barP_curr_1)

    F_1  += DbarF_curr
    # tmp,K4_1 = formResidual(F_1,barP_curr_1)
    tmp,K4_1 = formResidual(F,barP_curr_1)
    b_1 = -tmp
    print(f'|b-b1| = {np.linalg.norm(b-b_1)}')
    # P_1,_ = constitutive(F_1)
    # dP_with_bc = -(P_1.mean(axis=(2,3,4,)) - barP_curr_1)

    Fn    = np.linalg.norm(F)
    Fn_1  = np.linalg.norm(F_1)

    test = G_K_dF(F)

    newton_i = 0
    ksp_i = [0]

    # iterate as long as the iterative update does not vanish
    while True:
        dFm,_ = sp.cg(tol=1.e-8,
        #dFm,_ = sp.gmres(tol=1.e-8,
        A = sp.LinearOperator(shape=(F.size,F.size),matvec=G_K_dF,dtype='float'),
        b = b,
        callback=ksp_callback, #### cg counter
        )                                        # solve linear system using CG
        F    += dFm.reshape(ndim,ndim,Nx,Ny,Nz)  # update DOFs (array -> tens.grid)
        P,K4  = constitutive(F)                  # new residual stress and tangent
        # b     = -G(P)                            # convert res.stress to residual
        b     = -G(P) + G(barP_curr)

        # dbg_formResidual(F, barP_curr_1)

        # dP_with_bc = P.mean(axis=(2,3,4,)) - barP_curr_1
        # dP_with_bc = P.mean(axis=(2,3,4,)) 
        # dbg_formJacobian(dFm, K4)

        # tmp = G_K_dF(F)
        # tmp_1 = G_K_dF_1(F)
        # print(f'tmp {np.linalg.norm(tmp-tmp_1)}')

        G_K_dF_1 = partial(formJacobian, K_curr=K4_1)
        dFm_1,_ = sp.cg(tol=1.e-8,
        A = sp.LinearOperator(shape=(F_1.size,F_1.size),matvec=G_K_dF_1,dtype='float'),
        b = b_1,
        callback=ksp_callback, #### cg counter
        )                                        # solve linear system using CG
        F_1    += dFm_1.reshape(ndim,ndim,Nx,Ny,Nz)  # update DOFs (array -> tens.grid)
        P_1,K4_1  = constitutive(F_1)                  # new residual stress and tangent
        dP_with_bc = (P_1.mean(axis=(2,3,4,)) - barP_curr_1)
        #tmp,K4_1_tmp = formResidual(F_1,dP_with_bc)
        tmp,K4_1_tmp = formResidual(F_1,barP_curr_1)
        b_1 = -tmp

        print(np.linalg.norm(F-F_1))
        print(np.linalg.norm(b-b_1))

        # ^^^^^^^^^ dbg purpose ^^^^^^^^^
        #b_mean = b.reshape(ndim,ndim,Nx,Ny,Nz).mean(axis=(2,3,4))
        #Pav_Paim = P.mean(axis=(2,3,4))-barP_curr.mean(axis=(2,3,4))
        #print('P_aim', barP_curr.mean(axis=(2,3,4)))
        #print('P_av', P.mean(axis=(2,3,4)))
        #b_dbg = -G_dbg(P-barP_curr, Pav_Paim)
        #print(np.linalg.norm(b - b_dbg))
        #print(check(P.mean(axis=(2,3,4))-barP_curr.mean(axis=(2,3,4)))/1e6)
        #print((P.mean(axis=(2,3,4))-barP_curr.mean(axis=(2,3,4)))/1e6)

        dF_norm_rel = np.linalg.norm(dFm)/Fn
        rhs_norm = np.linalg.norm(b)
        newton_i += 1
        print(f'newton {newton_i} end: |dF|/|F| = {dF_norm_rel:8.2e}, |rhs| = |G(P)| = {rhs_norm:8.2e}, ksp iter = {ksp_i[0]}')
        if np.linalg.norm(dFm)/Fn<1.e-5 : break # check convergence
    
    # print(f'current F_bar = \n{F.mean(axis=(2,3,4))}')
    # print(f'current P_bar = \n{P.mean(axis=(2,3,4))}')
    print(f'=> load inc {inc+1} done with {newton_i} newton iter!')

print("##################### sim run done! #####################")
