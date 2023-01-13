
class CA_BTST_individual:
    def __init__(self, id, B, S, size, density, iterations, file=None):
        self.id = id
        self.B = B
        self.S = S
        self.size = size
        self.density = density
        self.iterations = iterations
        self.file = file
        self.density_evolution = None
