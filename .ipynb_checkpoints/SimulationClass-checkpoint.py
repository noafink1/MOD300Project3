import numpy as np

class Simulation():
    def __init__(self, 
                population_size, 
                no_init_infected=1,
                nx=50,
                ny=50,
                q=0.9):
        self.N_ = population_size
        self.IO_ = no_init_infected
        self.nx_ = nx
        self.ny_ = ny
        self.infection_probability_ = q
        self.HUMAN = 0
        self.ZOMBIE = 1
        self.STATE = np.repeat(self.HUMAN, self.N_)
        self.STATE[0] = self.ZOMBIE
        self.Walkers = np.random.randint(0, [self.nx_, self.ny_], size=(self.N_, 2))
        
    def move_walkers(self):
        u = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        dir = np.random.randint(0,4,size=self.N_)
        self.Walkers = self.Walkers + u[dir]

        