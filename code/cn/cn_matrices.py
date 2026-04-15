# Build Crank Nicolson matrices for the 1D diffusion equation
### A c^{n+1} = B c^{n} ###
import numpy as np

def build_CN_matrices(Nx, dt, dx, v, D, bc_type='Dirichlet'):
    
    r = D * dt / (2*dx**2)
    s = v * dt / (4*dx) 
    A = np.zeros((Nx, Nx))
    B = np.zeros((Nx, Nx))
    for i in range(1, Nx-1):
        A[i, i] = 1 + 2*r
        A[i, i-1] = -r - s
        A[i, i+1] = -r + s
        B[i, i] = 1 - 2*r
        B[i, i-1] = r + s
        B[i, i+1] = r - s
    # Boundary conditions
    if bc_type == 'Dirichlet':
        A[0, :] = 0
        A[0, 0] = 1

        A[-1, :] = 0
        A[-1, -1] = 1

        B[0, :] = 0
        B[-1, :] = 0
        
    elif bc_type == 'Neumann':
        # batas kiri
        A[0, 0] = 1 + 2*r
        A[0, 1] = -2*r-2*s
        B[0, 0] = 1 - 2*r
        B[0, 1] = 2*r + 2*s
        # batas kanan
        A[-1, -1] = 1 + 2*r
        A[-1, -2] = -2*r+2*s
        B[-1, -1] = 1 - 2*r
        B[-1, -2] = 2*r - 2*s
    return A, B

def solve_tridiagonal(a, b, c, d):
    """
    Menyelesaikan sistem tridiagonal Ax = d menggunakan algoritma Thomas
    a: diagonal bawah (length n-1)
    b: diagonal utama (length n)
    c: diagonal atas (length n-1)
    d: ruas kanan (length n)
    """
    n = len(b)
    cp = np.zeros(n)    #c'
    dp = np.zeros(n)    #d'
    x = np.zeros(n)

    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    denom = b[n-1] - a[n-2] * cp[n-2]
    dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / denom

    # Back substitution
    x[n-1] = dp[n-1]
    for i in np.arange((n-2), -1, -1):
        x[i] = dp[i] - (cp[i]) * (x[i+1])
    return x



