from SimulationClass import *

# Create a simulation object
sim = Simulation(683)

sim.init_zombies(10)

sim.make_gif("10zombies.gif", 300)

sim.init_zombies(1)

sim.make_gif("1zombie.gif", 300)

