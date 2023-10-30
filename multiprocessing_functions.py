import multiprocessing
from SimulationClass import *

def simulation_without_dead_zombies(initial_infected):
    no_humans = []
    no_zombies = []
    sim = Simulation(683)
    sim.init_zombies(initial_infected)
    sim.run_simulation(300)
    
    no_humans.append(sim.no_humans)
    no_zombies.append(sim.no_zombies)

    dictionary = {"no_humans" : no_humans, "no_zombies" : no_zombies}
    return dictionary


def run_multiprocessing_tasks(initial_infected):
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    
    results = pool.map(simulation_without_dead_zombies, initial_infected)

    pool.close()
    pool.join()
    
    return results
