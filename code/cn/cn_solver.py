import numpy as np
import matplotlib.pyplot as plt
import cn_matrices, initial_conditions, boundary_conditions
import utils
import os
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
        self.mass_history = []

        print("="*50)
        print("Initialized summary:")
        print(f"Grid: Nx={self.Nx}, Nt={self.Nt}")
        print(f"Spacing: dx={self.dx:.6f}, dt={self.dt:.6f}")
        print(f" Péclet number: {self.Pe:.2f}")

    # Time stepping 
    def time_step(self, u, a, b_diag, c_diag, B):
        # RHS
        rhs = B @ u
        
        if self.BC.type == 'Dirichlet':
            rhs[0] = self.BC.left
            rhs[-1] = self.BC.right

        #solve
        u_next = cn_matrices.solve_tridiagonal(a, b_diag, c_diag, rhs)

        # Enforce BC ke solusi akhir hanya untuk dirichlet
        if self.BC.type == 'Dirichlet':
            u_next[0] = self.BC.left
            u_next[-1] = self.BC.right

        return u_next

    # Main solver
    def solve(self, save_history=False):

        # Build matrices
        A, B = cn_matrices.build_CN_matrices(
            self.Nx, self.dt, self.dx, self.v, self.D,
            bc_type=self.BC.type
        )

        # extract tridiagonal system
        a = np.concatenate(([0], A.diagonal(-1))) # add dummy at start
        b_diag = A.diagonal(0)
        c_diag = np.concatenate((A.diagonal(1), [0]))  # add dummy at end
        
        # Initial condition
        u = self.IC(self.x).copy()

        if self.BC.type == 'Dirichlet':
            self.BC.apply(u)

        initial_mass = np.trapezoid(u, dx=self.dx)
        self.mass_history = [initial_mass]

        # Save history 
        if save_history:
            self.solution_history = [u.copy()]

        # Time loop
        for n in range(self.Nt):
            u = self.time_step(u, a, b_diag, c_diag, B)
            
            # Compute and store mass
            mass = utils.compute_mass(u, self.dx)
            self.mass_history.append(mass)

            if save_history:
                self.solution_history.append(u.copy())

        return u
    
    def plot_mass_evolution(self, save_path=None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.t, self.mass_history, marker='o', color='royalblue', label='$M(t)$')
        plt.axhline(y=self.mass_history[0], color='gray', linestyle='--', label='$M_0$')
        plt.xlabel('Waktu (t)')
        plt.ylabel('Total Massa')
        plt.title('Evolusi Mass dalam Solusi CN')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Plot evolusi massa disimpan ke {save_path}")

        plt.show()
        plt.close()
