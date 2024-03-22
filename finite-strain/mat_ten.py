#!/usr/bin/env python3

import numpy as np
import itertools

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
        C_3333[i,j,k,l] = M[ij,kl]
    return C_3333

# voigt
def tensor_2_m66_crop(C_3333):
    M99 = tensor_2_m99(C_3333)
    return M99[:6,:6]

def m66_2_tensor(M66):
    C_3333 = np.zeros((3,3,3,3))
    for i,j,k,l in itertools.product(range(3), repeat=4):
        ij = inv_pairs[(i,j)]
        if ij > 5: ij = ij - 3
        kl = inv_pairs[(k,l)]
        if kl > 5: kl = kl - 3
        C_3333[i,j,k,l] = M66[ij,kl]
    return C_3333


if __name__ == "__main__":
    print('test 3x3x3x3 <-> 9x9')
    ar = np.random.rand(3,3,3,3)
    print(np.linalg.norm(ar - m99_2_tensor(tensor_2_m99(ar))))
    ar = np.random.rand(6,6)
    print(np.linalg.norm(ar - tensor_2_m66_crop(m66_2_tensor(ar))))
