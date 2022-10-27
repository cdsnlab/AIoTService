import pygad
import numpy as np

class Genetic_algorithm:
    last_fitness = 0
    function_inputs = np.array([4, -2, 3.5])
    desired_output = 3
    
    def fitness_func(solution, solution_idx):
        output = np.sum(solution*Genetic_algorithm.function_inputs)
        fitness = 1.0 / np.abs(output - Genetic_algorithm.desired_output)
        return fitness    
            
    def crossover_func(parents, offspring_size, ga_instance):
        offspring = []
        while len(offspring) != offspring_size[0]:
            idx_1, idx_2 = np.random.choice(range(parents.shape[0]), size=2, replace=False)
            chlid = np.mean((parents[idx_1, :], parents[idx_2, :]), axis=0)
            offspring.append(chlid)
        return np.array(offspring)

    def callback_generation(ga_instance):
        ga_instance.generation += 1
        print("Generation = {generation}".format(generation=ga_instance.generation))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        print("Change     = {change}".format(change=ga_instance.best_solution()[1] - Genetic_algorithm.last_fitness))
        Genetic_algorithm.last_fitness = ga_instance.best_solution()[1]
        
    def __init__(self):
        self.generation = 0
        self.num_genes = 3
        self.sol_per_pop = 50 # Number of solutions in the population.
        self.num_generations = 1 # Number of generations.
        self.num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
        self.ga_instance = pygad.GA(num_generations=self.num_generations,
                                    num_parents_mating=self.num_parents_mating, 
                                    fitness_func=Genetic_algorithm.fitness_func,
                                    sol_per_pop=self.sol_per_pop, 
                                    num_genes=self.num_genes,
                                    init_range_low=-1,
                                    init_range_high=1,
                                    parent_selection_type="rws",
                                    #    keep_parents=,
                                    crossover_type=Genetic_algorithm.crossover_func,
                                    crossover_probability=0.8,
                                    mutation_type="random",
                                    mutation_probability=0.1,
                                    mutation_by_replacement=True,
                                    random_mutation_min_val=-1.0,
                                    random_mutation_max_val=1.0,
                                    on_generation=Genetic_algorithm.callback_generation)
    def show_results(self):
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        prediction = np.sum(np.array(Genetic_algorithm.function_inputs)*solution)
        print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
        print("Desired output : {desired_output}".format(desired_output=Genetic_algorithm.desired_output))
    
    def save_model(self, filename):
        # Saving the GA instance.
        self.ga_instance.save(filename=filename)
    
    def run(self):
        self.ga_instance.run()

ga = Genetic_algorithm()
ga.run()
ga.show_results()


Genetic_algorithm.desired_output = 9
Genetic_algorithm.last_fitness




# parents = np.array([[0.19439869, 0.12931624, 0.3089004 ],
#                     [0.29439869, 0.22931624, 0.3089004 ],
#                     [0.39439869, 0.32931624, 0.3089004 ],
#                     [0.49439869, 0.42931624, 0.3089004 ],
#                     [0.59439869, 0.52931624, 0.3089004 ],
#                     [0.69439869, 0.62931624, 0.3089004 ],
#                     [0.79439869, 0.72931624, 0.3089004 ]])

# idx_1, idx_2 = np.random.choice(range(parents.shape[0]), size=2, replace=False)
# chlid = np.mean((parents[idx_1, :], parents[idx_2, :]), axis=0)
# offspring.append(chlid)

# np.array(offspring)