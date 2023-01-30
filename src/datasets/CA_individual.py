
class CA_individual:
    def __init__(self, id, B, S, size, density, iterations, file=None):
        self.id = id
        self.B = B
        self.S = S
        self.size = size
        self.density = density
        self.iterations = iterations
        self.file = file
        self.density_evolution = None


class CA_individual_state:
    def __init__(self, id, B, S, initial_state, iterations, file=None):
        self.id = id
        self.B = B
        self.S = S
        self.initial_state = initial_state
        self.iterations = iterations
        self.file = file
        self.density_evolution = None
