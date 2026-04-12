import numpy as np
import cn_matrices, initial_conditions, boundary_conditions

class CNSolver:
    def __init__(self, L, T, Nx, Nt, v, D, IC, BC):
        self.L = L
        self.T = T
        self.Nx = Nx
        self.Nt = Nt
        self.v = v
        self.D = D
        self.IC = IC
        self.BC = BC

        self.dx = L / (Nx - 1)
        self.dt = T / Nt

        self.x = np.linspace(0, L, Nx)
        self.t = np.linspace(0, T, Nt + 1)

        self.Pe = v * L / D if D != 0 else np.inf

        print("Initialized summary:")
        print(f" Spatial grid: {self.Nx} points")
        print(f" Time steps: {self.Nt}")
        print(f" Time step size: {self.dt:.4f}")
        print(f" Grid spacing: {self.dx:.4f}")
        print(f" Péclet number: {self.Pe:.2f}")

    # Time stepping 
    def time_step(self, u, a, b_diag, c_diag, B):
        # RHS
        rhs = B @ u
        
        # Apply BC ke RHS
        self.BC.apply(rhs)

        # Solve tridiagonal system
        u_next = cn_matrices.solve_tridiagonal(a, b_diag, c_diag, rhs)

        # Enforce BC ke solusi akhir
        self.BC.apply(u_next)

        return u_next

    # Main solver
    def solve(self, save_history=False):

        # Build matrices
        A, B = cn_matrices.build_CN_matrices(
            self.Nx, self.dt, self.dx, self.v, self.D,
            bc_type=self.BC.type
        )

        # Extract diagonals 
        a = A.diagonal(-1)
        b_diag = A.diagonal(0)
        c_diag = A.diagonal(1)

        # Initial condition
        u = self.IC(self.x)
        self.BC.apply(u)

        # Save history (optional)
        if save_history:
            self.solution_history = [u.copy()]

        # Time loop
        for _ in range(self.Nt):
            u = self.time_step(u, a, b_diag, c_diag, B)

            if save_history:
                self.solution_history.append(u.copy())

        return u