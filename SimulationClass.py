import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class Simulation():
    '''
    Class used to simulate the spread of a zombie virus in a population using random walk.
    The population is represented by a 2D grid, where each cell can contain a human, a zombie or a dead zombie.
    The simulation is run by calling the run_simulation method, which takes the number of iterations as input.
    The simulation can be reset by calling the reset method.
    The simulation can be plotted by calling the plot method.
    The simulation can be saved as a gif by calling the make_gif method.
    '''
    def __init__(self, 
                population_size, 
                nx=50,
                ny=50,
                q=0.9,
                p_death = 0.0):
        '''
        Initialize the Simulation class.

        Parameters:
        population_size (int): The number of individuals in the population.
        nx (int): The number of cells in the x-direction.
        ny (int): The number of cells in the y-direction.
        q (float): The probability of a human becoming a zombie when it encounters a zombie.
        p_death (float): The life expectancy of a zombie.
        '''
        self.N_ = population_size
        self.IO_ = 1
        self.nx_ = nx
        self.ny_ = ny
        self.infection_probability_ = q
        self.HUMAN = 0
        self.ZOMBIE = 1
        self.DEAD_ZOMBIE = 2
        self.p_death = p_death
        self.no_humans = np.empty(0, dtype=int)    
        self.no_zombies = np.empty(0, dtype=int)  
        self.no_dead_zombies = np.empty(0, dtype=int)
        self.beta = np.empty(0, dtype=float)
        self.tau_death = np.empty(0, dtype=float)
        self.STATE = np.repeat(self.HUMAN, self.N_)
        self.Walkers = np.random.randint(0, [self.nx_, self.ny_], size=(self.N_, 2))
        self.Old_Walkers = np.copy(self.Walkers)
        self.immunity_rate = 0.0
        self.immunity_list = np.zeros(self.N_,dtype=bool)
        self.num_of_immune = 0

    def set_immunity(self, immunity_rate):
        '''
        Takes in a immunity rate as input and sets the immunity rate of the population to the given value.

        Parameters:
        immunity_rate (float): The immunity rate of the population.
        '''
        self.immunity_rate = immunity_rate
        num_of_immune = int((self.N_-self.IO_) * immunity_rate)
        self.num_of_immune = num_of_immune
        self.immunity_list[self.IO_:self.IO_+num_of_immune] = True

    def init_zombies(self, n):
        '''
        Initializes the population with n zombies.

        Parameters:
        n (int): The number of zombies to initialize.
        '''
        self.IO_ = n
        self.STATE[0:n] = self.ZOMBIE

    def check_illegal_move(self):
        '''
        Checks whether the walkers are outside the grid and returns a boolean array with True at the index of the walkers that are outside the grid.
        '''
        # Checks whether walkers are outside the of the x axis
        wrong_place_x = np.logical_or(self.Walkers[:,0] < 0, self.Walkers[:,0] > self.nx_-1)

        # Checks whether walkers are outside the of the y axis
        wrong_place_y = np.logical_or(self.Walkers[:,1] < 0, self.Walkers[:,1] > self.ny_-1)

        # Returns a boolean array with True at the index of the walkers that are outside the grid
        return np.logical_or(wrong_place_x, wrong_place_y)
    
    def move_walkers(self):
        '''
        Moves the walkers one step in a random direction.
        '''
        u = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

        dir = np.random.randint(0, 4, size=self.N_)

        # List of the movement of the walkers
        movement = u[dir]

        dead_zombie_positions = self.Walkers[self.STATE == self.DEAD_ZOMBIE]
        
        # The walkers are moved one step in a random direction
        self.Walkers += movement
        self.Walkers[self.STATE == self.DEAD_ZOMBIE] = dead_zombie_positions

        # The walkers that are outside the grid are set to their old position
        index = self.check_illegal_move()
        self.Walkers[index] = self.Old_Walkers[index]

        self.Old_Walkers = np.copy(self.Walkers)

    def check_collision(self):
        '''
        Checks whether a zombie and a human are at the same position and returns a boolean array with True at the index where a human is at the same place as a zombie.
        '''
        zombie_coordinates = self.Walkers[self.STATE == self.ZOMBIE]
        zombie_coordinates_reshaped = zombie_coordinates[:, None, :]
        walkers_reshaped = self.Walkers[None, :, :]
        
        # This yields an matrix with True or False if the coordinate of the walker matches the coordinate of a zombie
        matches = np.all(zombie_coordinates_reshaped == walkers_reshaped , axis=-1)

        # Since every zombie also is a walker, they will always yield True at the point it meets itself on the matrix.
        # Therefore the diagonal is set to False.
        np.fill_diagonal(matches, False)

        # List that is of the same length as walkers and contains True if the walker is at the same position as a zombie and is not a zombie.
        collision_mask = np.any(matches, axis=0)
        return collision_mask
    
    def set_zombie(self):
        '''
        Changes the state of a human to a zombie if it is at the same position as a zombie and the infection probability is met.
        '''
        collision = self.check_collision()
        if np.any(collision):
            random = np.around(np.random.uniform(0.0, 1.0, self.N_), 2)
            condition1 = (random <= self.infection_probability_)
            condition2 = (self.STATE == self.HUMAN)
            condition3 = (self.immunity_list == False)

            # If the walker is a human, is not immune and is at the same position as a zombie, it becomes a zombie.
            self.STATE = np.where(condition1 & condition2 & condition3 & collision, self.ZOMBIE, self.STATE)

    def check_if_zombies_die(self):
        '''
        Checks whether a zombie dies and changes the state of the zombie to a dead zombie if the probability of death is met.
        '''
        if self.p_death != 0:
            random = np.around(np.random.uniform(0.0, 1.0, self.N_), 2)
            condition1 = random <= self.p_death
            condition2 = self.STATE == self.ZOMBIE

            # If the walker is a zombie and the probability of death is met, it becomes a dead zombie.
            self.STATE = np.where(condition1 & condition2, self.DEAD_ZOMBIE, self.STATE)

    def plot(self):
        '''
        Plots the population.
        Green represents humans, red represents zombies and black represents dead zombies.
        '''
        H = self.Walkers[self.STATE == self.HUMAN]
        Z = self.Walkers[self.STATE == self.ZOMBIE]
        D = self.Walkers[self.STATE == self.DEAD_ZOMBIE]
        plt.figure(figsize=(6,6))
        plt.xlim(-1, self.nx_+1)
        plt.ylim(-1, self.ny_+1)
        xticks = np.arange(0, self.nx_+2, 1)
        yticks = np.arange(0, self.ny_+2, 1)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.scatter(H[:,0], H[:,1], color='green', s=60)
        plt.scatter(Z[:,0], Z[:,1], color='red', s=60)
        plt.scatter(D[:,0], D[:,1], color='black', s=60)
        plt.plot([0, self.ny_], [0,0], linestyle='dashed', color='black')
        plt.plot([0, self.ny_], [self.nx_, self.nx_], linestyle='dashed', color='black')
        plt.plot([0,0], [0,self.ny_], linestyle='dashed', color='black')
        plt.plot([self.nx_, self.ny_], [0, self.ny_], linestyle='dashed', color='black')
        plt.title('Zombie simulation')
        plt.grid()
        plt.show()

    def make_gif(self, name, n):
        '''
        Saves the simulation as a gif.

        Parameters:
        name (str): The name of the gif, does not need .gif ending.
        n (int): The number of iterations to run the simulation.
        '''
        frames = []
        for i in range(n):
            if np.sum(self.STATE == self.HUMAN) == 0:
                    break
            if np.sum(self.STATE == self.ZOMBIE) == 0:
                    break
            self.check_if_zombies_die()
            self.move_walkers()
            self.set_zombie()
            if i % 5 == 0:             
                H = self.Walkers[self.STATE == self.HUMAN]
                Z = self.Walkers[self.STATE == self.ZOMBIE]
                D = self.Walkers[self.STATE == self.DEAD_ZOMBIE]
                I = self.Walkers[self.immunity_list == True]
                plt.figure(figsize=(10,10))
                plt.xlim(-1, self.nx_+1)
                plt.ylim(-1, self.ny_+1)
                plt.grid()
                plt.xticks([])
                plt.yticks([])
                scatter_H = plt.scatter(H[:,0], H[:,1], color='green', s=60, label='Humans')
                scatter_Z = plt.scatter(Z[:,0], Z[:,1], color='red', s=60, label='Zombies')
                scatter_DZ = plt.scatter(D[:,0], D[:,1], marker="X",color='black', s=70, label='Dead Zombies')
                scatter_I = plt.scatter(I[:,0], I[:,1], color='blue', s=60, label='Immune')
                plt.plot([0, self.ny_], [0,0], linestyle='dashed', color='black')
                plt.plot([0, self.ny_], [self.nx_, self.nx_], linestyle='dashed', color='black')
                plt.plot([0,0], [0,self.ny_], linestyle='dashed', color='black')
                plt.plot([self.nx_, self.ny_], [0, self.ny_], linestyle='dashed', color='black')
                plt.title('Zombie simulation')
                plt.legend(handles=[scatter_H, scatter_Z, scatter_DZ, scatter_I], loc='upper left')
                fig = plt.gcf()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                img = Image.open(buf)
                frames.append(img)
                plt.close()

        frames[0].save(f"{name}.gif", save_all=True, append_images=frames[1:], optimize=True, duration=500, loop=0)
        plt.close()
        self.reset()


    def calculate_no_humans_and_zombies(self):
        '''
        Calculates the number of humans and zombies in the population.
        Returns a tuple with the number of humans and zombies.
        '''
        return np.sum(self.STATE == self.HUMAN), np.sum(self.STATE == self.ZOMBIE)
    
    def calculate_beta(self):
        '''
        Calculates the beta value of the population.
        '''
        curr_humans, curr_zombies = self.calculate_no_humans_and_zombies()

        # Returns 0 if there are no humans or zombies left in the population.
        if curr_humans == 0 or curr_zombies == 0:
            return 0

        # Returns the beta value of the population.
        return -np.round(((self.no_humans[-1] - self.no_humans[-2])*self.N_)/(self.no_humans[-1]*self.no_zombies[-1]), 3)

    def reset(self):
        '''
        Resets the simulation. Sets all the necessary variables to their initial values.
        '''
        self.STATE = np.repeat(self.HUMAN, self.N_)
        self.init_zombies(self.IO_)
        self.Walkers = np.random.randint(0, [self.nx_, self.ny_], size=(self.N_, 2))
        self.Old_Walkers = np.copy(self.Walkers)
        self.no_humans = np.empty(0, dtype=int)
        self.no_zombies = np.empty(0, dtype=int)
        self.no_dead_zombies = np.empty(0, dtype=int)
        self.beta = np.empty(0, dtype=float)
        self.tau_death = np.empty(0, dtype=float)
        self.immunity_list = np.zeros(self.N_,dtype=bool)
        self.set_immunity(self.immunity_rate)
        

    def run_simulation(self, n, calculate_beta=False, calculate_no_dead_zombies=False):
        '''
        Runs the simulation for n iterations.

        Parameters:
        n (int): The number of iterations to run the simulation.
        calculate_beta (bool): Whether to calculate the beta value of the population.
        calculate_no_dead_zombies (bool): Whether to calculate the number of dead zombies in the population.
        '''
        for i in range(n):
            if calculate_no_dead_zombies:
                self.no_dead_zombies = np.append(self.no_dead_zombies, np.sum(self.STATE == self.DEAD_ZOMBIE))
            
            self.no_humans=np.append(self.no_humans,self.calculate_no_humans_and_zombies()[0])
            self.no_zombies = np.append(self.no_zombies,self.calculate_no_humans_and_zombies()[1])
                        
            self.check_if_zombies_die()
            self.move_walkers()
            self.set_zombie()

            if i > 0 and calculate_beta:
                self.beta = np.append(self.beta, self.calculate_beta())
