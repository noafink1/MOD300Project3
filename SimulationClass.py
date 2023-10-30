import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class Simulation():
    def __init__(self, 
                population_size, 
                nx=50,
                ny=50,
                q=0.9,
                p_death = 0.0):
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
        
    def init_zombies(self, n):
        self.IO_ = n
        self.STATE[0:n] = self.ZOMBIE

    def check_illegal_move(self):
        wrong_place_x = np.logical_or(self.Walkers[:,0] < 0, self.Walkers[:,0] > self.nx_-1)
        wrong_place_y = np.logical_or(self.Walkers[:,1] < 0, self.Walkers[:,1] > self.ny_-1)
        return np.logical_or(wrong_place_x, wrong_place_y)
    
    def move_walkers(self):
        u = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

        dir = np.random.randint(0, 4, size=self.N_)
        movement = u[dir]

        dead_zombie_positions = self.Walkers[self.STATE == self.DEAD_ZOMBIE]
        
        self.Walkers += movement
        self.Walkers[self.STATE == self.DEAD_ZOMBIE] = dead_zombie_positions

        index = self.check_illegal_move()
        self.Walkers[index] = self.Old_Walkers[index]

        self.Old_Walkers = np.copy(self.Walkers)

    def check_collision(self):
        zombie_coordinates = self.Walkers[self.STATE == self.ZOMBIE]
        zombie_coordinates_reshaped = zombie_coordinates[:, None, :]
        walkers_reshaped = self.Walkers[None, :, :]
        
        #this yields an matrix with true or false if the coordinate of the walker
        #matches the coordinate of a zombie
        matches = np.all(zombie_coordinates_reshaped == walkers_reshaped , axis=-1)

        #because every zombie also is a walker, they will always yield true at the point it meets iteself on the matrix
        #therefor we set the diagonal to false
        np.fill_diagonal(matches, False)

        collision_mask = np.any(matches, axis=0)
        
        return collision_mask
        
        #_______OLD CODE_______
        # living_zombie_array = self.Walkers[self.STATE == self.ZOMBIE]
        # match_mask = np.all(living_zombie_array[:, None, :] == self.Walkers, axis=-1)
        # zombie_collision_with_zombies = np.any(match_mask, axis=0)
        # return np.logical_xor(zombie_collision_with_zombies, self.STATE)
    
    def set_zombie(self):
        collision = self.check_collision()
        if np.any(collision):
            random = np.around(np.random.uniform(0.0, 1.0, self.N_), 2)
            condition1 = (random <= self.infection_probability_)
            condition2 = (self.STATE == 0)

            self.STATE = np.where(condition2 & condition1 & collision, self.ZOMBIE, self.STATE)

        #___ OLD CODE ___
        # random = np.around(np.random.uniform(0.0, 1.0, self.N_), 2)
        # condition1 = random <= self.infection_probability_
        # condition2 = self.check_collision() == 1
        # self.STATE = np.where(condition1 & condition2, self.ZOMBIE, self.STATE)

    def check_if_zombies_die(self):
        if self.p_death != 0:
            random = np.around(np.random.uniform(0.0, 1.0, self.N_), 2)
            condition1 = random <= self.p_death
            condition2 = self.STATE == self.ZOMBIE
            self.STATE = np.where(condition1 & condition2, self.DEAD_ZOMBIE, self.STATE)

    def plot(self):
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
        """
        makes a gif
        n: simulation time
        name: name of the gif, does NOT need the .gif ending
        """
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
                plt.figure(figsize=(10,10))
                plt.xlim(-1, self.nx_+1)
                plt.ylim(-1, self.ny_+1)
                plt.grid()
                plt.xticks([])
                plt.yticks([])
                scatter_H = plt.scatter(H[:,0], H[:,1], color='green', s=60, label='Humans')
                scatter_Z = plt.scatter(Z[:,0], Z[:,1], color='red', s=60, label='Zombies')
                scatter_DZ = plt.scatter(D[:,0], D[:,1], color='black', s=60, label='Dead Zombies')
                plt.plot([0, self.ny_], [0,0], linestyle='dashed', color='black')
                plt.plot([0, self.ny_], [self.nx_, self.nx_], linestyle='dashed', color='black')
                plt.plot([0,0], [0,self.ny_], linestyle='dashed', color='black')
                plt.plot([self.nx_, self.ny_], [0, self.ny_], linestyle='dashed', color='black')
                plt.title('Zombie simulation')
                plt.legend(handles=[scatter_H, scatter_Z, scatter_DZ], loc='upper left')
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
        return np.sum(self.STATE == self.HUMAN), np.sum(self.STATE == self.ZOMBIE)
    
    def calculate_beta(self):
        curr_humans, curr_zombies = self.calculate_no_humans_and_zombies()

        if curr_humans == 0 or curr_zombies == 0:
            return 0

        return -np.round(((curr_humans - self.no_humans[-1])*self.N_)/(curr_humans*curr_zombies), 3)
    
    def calculate_tau_death(self):
        if len(self.no_dead_zombies) < 2:
            return 0

        return 1/((self.no_dead_zombies[-1]-self.no_dead_zombies[-2])/self.no_zombies[-1])
    
    def reset(self):
        self.STATE = np.repeat(self.HUMAN, self.N_)
        self.init_zombies(self.IO_)
        self.Walkers = np.random.randint(0, [self.nx_, self.ny_], size=(self.N_, 2))
        self.Old_Walkers = np.copy(self.Walkers)
        self.no_humans = np.empty(0, dtype=int)
        self.no_zombies = np.empty(0, dtype=int)
        self.no_dead_zombies = np.empty(0, dtype=int)
        self.beta = np.empty(0, dtype=float)
        self.tau_death = np.empty(0, dtype=float)

    def run_simulation(self, n, calculate_beta=False, calculate_no_dead_zombies=False, calculate_tau_death=False):
        for i in range(n):
            
            if calculate_no_dead_zombies:
                self.no_dead_zombies = np.append(self.no_dead_zombies, np.sum(self.STATE == self.DEAD_ZOMBIE))
            
            self.no_humans=np.append(self.no_humans,self.calculate_no_humans_and_zombies()[0])
            self.no_zombies = np.append(self.no_zombies,self.calculate_no_humans_and_zombies()[1])
            
            
            self.check_if_zombies_die()
            self.move_walkers()
            self.set_zombie()


            if i >= 1 and calculate_beta:
                self.beta = np.append(self.beta, self.calculate_beta())

            if i >= 1 and calculate_tau_death:
                self.tau_death = np.append(self.tau_death, self.calculate_tau_death())

            # self.plot()

        