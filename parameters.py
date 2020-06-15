import numpy as np
import array
import random
from deap import base
from deap import creator
from deap import algorithms
from deap import tools
import matplotlib.pyplot as plt
import pandas as pd

# Basic variables
N: int = 943
M: int = 1682

# Load preprocessed data
user_ratings = np.load('user_ratings.npy')

# Pick a user (randomly)
user_id = random.randint(0, N-1)  # array index [0, N-1] -> user id [1, N]
target_ratings = user_ratings[user_id, :]
print(f'Selected User {user_id + 1}.')

for i in range(0, N):
    user_mean = int(np.around(np.mean(user_ratings[i, :]), decimals=0))
    for j in range(0, M):
        if user_ratings[i, j] == 0:
            user_ratings[i, j] = user_mean
corr = np.corrcoef(user_ratings)

# Find 10 most similar users (neighbours)
indices = np.argpartition(corr[user_id, :], -11)[-11:]
indices = np.delete(indices, np.where(indices == user_id), axis=0)  # remove self
print(f'Neighbours: {indices + 1}')
neighbourhood = np.empty([11, M], dtype=np.int32)
for k in range(1, 11):
    neighbourhood[k, :] = user_ratings[indices[k-1], :]
del user_ratings
del corr

# Set up genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # goal is max correlation
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)


def create_individual(ind_ratings):
    individual = array.array('i', ind_ratings)
    for i in range(M):
        if ind_ratings[i] == 0:
            individual[i] = random.randint(1, 5)
    return individual


def evaluate_individual(individual, neighbours):
    for i in range(M):
        neighbours[0, i] = individual[i]
    corr_matrix = np.corrcoef(neighbours)
    corr_mean = np.mean(corr_matrix[0, 1:]) + 1
    return corr_mean,


def check_ratings(ind_ratings):
    def decorator(func):
        def wrappper(*args, **kargs):
            args_tuple = func(*args, **kargs)
            individual = args_tuple[0]
            for i in range(0, M):
                if ind_ratings[i] != 0 and ind_ratings[i] != individual[i]:
                    individual[i] = ind_ratings[i]
            return individual,
        return wrappper
    return decorator


toolbox = base.Toolbox()
toolbox.register("ratings_attr", create_individual, target_ratings)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ratings_attr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual, neighbours=neighbourhood)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=1/M)
toolbox.decorate("mutate", check_ratings(target_ratings))

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)

population_sizes = [20, 20, 20, 20, 20, 200, 200, 200, 200]
crossover_prob = [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.1, 0.9]
mutation_prob = [0.0, 0.01, 0.1, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01]
max_gen = 200

sum_gens = np.zeros(9, dtype=np.int32)
sum_best_fit = np.zeros(9)
results = np.empty([9, 5])
set_count = 0

for pop_size, cxpb, mutpb in zip(population_sizes, crossover_prob, mutation_prob):

    # Parameter-specific initialisations
    toolbox.register("select", tools.selTournament, k=pop_size, tournsize=10)
    pop = toolbox.population(n=pop_size)

    print(f'\nUsing values pop_size={pop_size}, cxpb={cxpb}, mutpb={mutpb} ...')
    for n in range(10):

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        log = tools.Logbook()
        log.record(gen=0, **record)

        term = False  # exit condition
        g = 1

        gen_term_cond = int(0.1 * max_gen)
        gen_best_unchanged = 0

        # Genetic Algorithm
        while not term:
            # Select and clone the next generation individuals
            offspring = map(toolbox.clone, toolbox.select(pop))

            # Apply crossover and mutation on the offspring
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            record = stats.compile(pop)
            log.record(gen=g, **record)

            # Termination conditions
            best = log.select("max")
            # reached max number of generations
            if g >= max_gen:
                term = True
            # best individual of gen is <1% better than best individual of previous gen
            elif (g > 20) and (best[-1][0] - best[-20][0]) < (0.00001 * best[-20][0]):
                term = True
            # best individual of gen is same as best individual of previous gen
            elif (g > 1) and (best[-1][0] == best[-2][0]):
                gen_best_unchanged += 1
                if gen_best_unchanged > gen_term_cond:
                    term = True
                else:
                    g += 1
            # else continue
            else:
                gen_best_unchanged = 0
                g += 1

        gen, best = log.select("gen", "max")

        # arrays to store graph data
        best_fitnesses = np.zeros(max_gen + 1)
        num_values = np.zeros(max_gen + 1)

        sum_gens[set_count] += g
        sum_best_fit[set_count] += best[-1][0]
        for i in range(len(gen)):
            best_fitnesses[i] = best[i][0]
            num_values[i] += 1
        del gen
        del best
        del record

    toolbox.unregister("select")

    # Prepare data for pyplot
    num_values = np.trim_zeros(num_values)
    best_fitnesses.resize(len(num_values))
    for i in range(len(num_values)):
        best_fitnesses[i] /= num_values[i]

    # Plot data
    fig = plt.figure(dpi=300, edgecolor='black')
    plt.plot(best_fitnesses)
    plt.title('Evolution of Best Individual | pop_size={0} | cxpb={1} | mutpb={2}'.format(pop_size, cxpb, mutpb))
    plt.ylabel('Fitness of Best Individual')
    plt.xlabel('Generation')
    plt.legend('Fitness', loc='upper left')
    plt.show()
    # Save to file in png format (report)
    fig.savefig(fname=f'./param_plots/params_set_{set_count + 1}')

    del best_fitnesses
    del num_values
    results[set_count, 0] = pop_size
    results[set_count, 1] = cxpb
    results[set_count, 2] = mutpb
    set_count += 1

for i in range(9):
    results[i, 3] = int(sum_gens[i] / 10)
    results[i, 4] = sum_best_fit[i] / 10

del sum_gens
del sum_best_fit

print('\nDone.\n')
table_cols = ['Population Size', 'Crossover Probability', 'Mutation Probability',
              'Generations (avg)', 'Best Fitness (avg)']
df = pd.DataFrame(data=results, columns=table_cols)
df = df.astype({'Population Size': np.int32, 'Generations (avg)': np.int32})
print(df.to_string(index=False, formatters=({'Best Fitness (avg)': '{:,.5f}'.format})))
