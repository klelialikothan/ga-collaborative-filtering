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
user_ratings = np.load('train_data.npy')
test_data = np.load('test_data.npy')

# Pick a user (randomly)
user_id = random.randint(0, N-1)  # array index [0, N-1] -> user id [1, N]
target_ratings = user_ratings[user_id, :]
print(f'Selected User {user_id + 1}.')
test_ratings_user = np.empty([10, 2], dtype=np.int32)
j = 0
for i in range(np.shape(test_data)[0]):
    if test_data[i, 0] == user_id:
        test_ratings_user[j, 0] = test_data[i, 1] - 1
        test_ratings_user[j, 1] = test_data[i, 2]
        j += 1

# Fill in missing ratings with mean rating
for i in range(0, N):
    user_mean = int(np.around(np.mean(user_ratings[i, :]), decimals=0))
    for j in range(0, M):
        if user_ratings[i, j] == 0:
            user_ratings[i, j] = user_mean

# Find 10 most similar users (neighbours)
corr = np.corrcoef(user_ratings)
indices = np.argpartition(corr[user_id, :], -11)[-11:]
indices = np.delete(indices, np.where(indices == user_id), axis=0)  # remove self
print(f'Neighbours: {indices + 1}')
neighbourhood = np.empty([11, M], dtype=np.int32)
for k in range(1, 11):
    neighbourhood[k, :] = user_ratings[indices[k-1], :]
del user_ratings
del corr

# Set up genetic algorithm
pop_size = 200
cxpb = 0.9
mutpb = 0.01
max_gen = 200
gen_term_cond = int(0.1 * max_gen)

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
toolbox.register("select", tools.selTournament, k=pop_size, tournsize=10)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=1/M)
toolbox.decorate("mutate", check_ratings(target_ratings))

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)


def rmse(ind_ratings, test_ratings):
    sq_err = 0
    for j, k in test_ratings:
        sq_err += (ind_ratings[j] - k)**2
    return (sq_err/10)**(1/2)/4


def mae(ind_ratings, test_ratings):
    abs_err = 0
    for j, k in test_ratings:
        abs_err += abs(ind_ratings[j] - k)
    return abs_err/40


pop = toolbox.population(n=pop_size)

mean_rmse = 0
mean_mae = 0

for n in range(10):

    print(f'\nIteration {n + 1} ...')

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    log = tools.Logbook()
    log.record(gen=0, **record)
    rmse_values = np.zeros(max_gen+1)
    mae_values = np.zeros(max_gen+1)

    term = False  # exit condition
    gen_best_unchanged = 0
    g = 1

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

        best = log.select("max")
        best_indiv = None
        for ind in offspring:
            if ind.fitness.values == best[-1][0]:
                best_indiv = ind
        rmse_values[g] = rmse(best_indiv, test_ratings_user)
        mae_values[g] = mae(best_indiv, test_ratings_user)

        # reached max number of generations
        if g >= max_gen:
            term = True
        # best individual of gen is <1% better than best individual of previous gen
        elif (g > 20) and (best[-1][0] - best[-20][0]) < (0.0001 * best[-20][0]):
            term = True
        # best individual of gen is same as best individual of previous gen
        elif (g > 20) and (best[-1][0] == best[-20][0]):
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

    for i in range(len(gen)):
        best_fitnesses[i] = best[i][0]
        num_values[i] += 1
    del gen
    del best
    del record

    # Prepare data for pyplot
    rmse_values = np.trim_zeros(rmse_values)
    mae_values = np.trim_zeros(mae_values)
    num_values = np.trim_zeros(num_values)
    best_fitnesses.resize(len(num_values))
    for i in range(len(num_values)):
        best_fitnesses[i] /= num_values[i]
    mean_rmse += rmse_values[-1]
    mean_mae += mae_values[-1]

    # Plot data
    fig = plt.figure(dpi=300, edgecolor='black')
    plt.plot(best_fitnesses)
    plt.title(f'Evolution of Best Individual | ID={user_id} | Iteration #{n + 1}')
    plt.ylabel('Fitness of Best Individual')
    plt.xlabel('Generation')
    plt.legend('Fitness', loc='upper right')
    plt.show()
    # Save to file in png format (report)
    fig.savefig(fname=f'./eval_1_plots/fit_{n + 1}')

    fig2 = plt.figure(dpi=300, edgecolor='black')
    plt.plot(rmse_values)
    plt.plot(mae_values)
    plt.title(f'Prediction Error for Best Individual | ID={user_id} | Iteration #{n + 1}')
    plt.ylabel('Statistics')
    plt.xlabel('Generation')
    plt.legend(['RMSE', 'MAE'], loc='upper right')
    plt.show()
    # Save to file in png format (report)
    fig2.savefig(fname=f'./eval_1_plots/metrics_{n + 1}')


print('\nDone.\n')
mean_rmse /= 10
mean_mae /= 10
table_cols = ['User ID', 'RMSE (avg)', 'MAE (avg)']
df = pd.DataFrame(data=[[user_id, mean_rmse, mean_mae]], columns=table_cols)
df = df.astype({'User ID': np.int32})
print(df.to_string(index=False, formatters=({'RMSE (avg)': '{:,.5f}'.format, 'MAE (avg)': '{:,.5f}'.format})))
