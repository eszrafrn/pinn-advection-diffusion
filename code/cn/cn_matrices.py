# Build Crank Nicolson matrices for the 1D diffusion equation
### A c^{n+1} = B c^{n} ###
import numpy as np

def build_CN_matrices(Nx, dt, dx, v, D, bc_type='Dirichlet'):
    
    r = D * dt / (2*dx**2)
    s = v * dt / (4*dx) 
    A = np.zeros((Nx, Nx))
    B = np.zeros((Nx, Nx))
    for i in range(1, Nx-1):
        A[i, i-1] = -r - s
        A[i, i] = 1 + 2*r
        A[i, i+1] = -r + s

        B[i, i-1] = r + s
        B[i, i] = 1 - 2*r
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
        # batas kiri (x=0)
        A[0, 0] = 1 + 2*r
        A[0, 1] = -2*r

        B[0, 0] = 1 - 2*r
        B[0, 1] = 2*r

        # batas kanan (x=L)
        A[-1, -2] = -2*r
        A[-1, -1] = 1 + 2*r
        
        B[-1, -2] = 2*r
        B[-1, -1] = 1 - 2*r

    return A, B

def solve_tridiagonal(a, b, c, d):
    """
    Menyelesaikan sistem tridiagonal Ax = d menggunakan algoritma Thomas
    a: diagonal bawah 
    b: diagonal utama 
    c: diagonal atas 
    d: ruas kanan 
    """
    n = len(b)
    
    a = np.copy(a)
    b = np.copy(b)
    c = np.copy(c)
    d = np.copy(d)


    # Forward elimination
    for i in range(1, n):
        m = a[i] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x



