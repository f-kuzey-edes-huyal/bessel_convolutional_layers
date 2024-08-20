import numpy as np
from scipy import special

def getTransMat(m_max, j_max, k):

    R = k // 2

    k_mn = np.zeros((m_max+1, j_max+1), dtype=np.double)
    for m in range(0, m_max+1, 1):
        k_mn[m,1:] = special.jnp_zeros(m, j_max) / R

    # Build grid
    _x = np.linspace(-R, R, k)
    grid = np.meshgrid(_x, _x)
    theta = np.angle(grid[0][:,:] + 1j * grid[1][:,:])

    # Compute A
    J = np.zeros((R+1, (j_max+1)*(m_max+1)))
    P = np.linspace(0, R, R+1)
    for m in range(0, m_max+1):
        for j in range(0, j_max+1):

            J[:,j+m*(j_max+1)] = special.jv(m, k_mn[m,j] * P[:])

    A = np.sqrt(2. * np.pi * np.matmul(P, np.square(J)))
    A = np.reshape(A, (m_max+1, j_max+1))

    for m in range(1, m_max+1):
        A[m,0] = 1.

    # Compute transMat
    transMat = np.zeros((k, k, m_max+1, j_max+1), dtype=np.complex64)
    for j in range(0, j_max+1):
        for m in range(0, m_max+1):
            for _j in range(k):
                for _i in range(k):
                    if np.sqrt(grid[0][_i,_j]**2 + grid[1][_i,_j]**2) > R:
                        continue
                    transMat[_i, _j, m, j] = special.jv(m, k_mn[m,j] * np.sqrt(grid[0][_i,_j]**2 + grid[1][_i,_j]**2))
                    transMat[_i, _j, m, j] *= np.exp(-1j * m * theta[_i,_j]) * A[m,j]**-1
                    
    return transMat