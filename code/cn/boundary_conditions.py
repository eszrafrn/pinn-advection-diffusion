class Dirichlet_BC:
    def __init__(self, left=0.0, right=0.0):
        self.left = left
        self.right = right
        self.type = 'Dirichlet'
    
    def apply_left(self, c):
        c[0] = self.left

    def apply_right(self, c):
        c[-1] = self.right

    def apply(self, c, dx=None):
        self.apply_left(c)
        self.apply_right(c)
    
    def __repr__(self):
        return f"Dirichlet_BC(left={self.left}, right={self.right})"


class Neumann_BC:
    def __init__(self, flux_left=0.0, flux_right=0.0):
        self.flux_left = flux_left
        self.flux_right = flux_right
        self.type = 'Neumann'

        if flux_left != 0.0 or flux_right != 0.0:
            raise NotImplementedError("Hanya Neumann Homogen yang didukung")


    def apply(self, c, dx=None):
        pass

    def __repr__(self):
        return f"Neumann_BC(flux_left={self.flux_left}, flux_right={self.flux_right})"