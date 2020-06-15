import numpy as np
import array
import random
from deap import base
from deap import creator
from deap import algorithms
from deap import tools
import pandas as pd

# Basic variables
N: int = 943
M: int = 1682

# Load preprocessed data
user_ratings_zero_fill = np.load('train_data.npy')  # zeros indicate missing ratings
test_data = np.load('test_data.npy')

# Fill in missing ratings with mean rating, store in separate array
user_ratings = user_ratings_zero_fill
for i in range(0, N):
    user_mean = int(np.around(np.mean(user_ratings[i, :]), decimals=0))
    for j in range(0, M):
        if user_ratings[i, j] == 0:
            user_ratings[i, j] = user_mean

# Calculate Pearson Correlation Matrix for all users
corr = np.corrcoef(user_ratings)

# Metrics
def rmse(ind_ratings, test_ratings):
    sq_err = 0
    for j, k in test_ratings:
        sq_err += (ind_ratings[j] - k)**2
    # note: divided by 4 as 4 is max value of sqrt(squared error) => returns a number in [0, 1]
    return (sq_err/10)**(1/2)/4


def mae(ind_ratings, test_ratings):
    abs_err = 0
    for j, k in test_ratings:
        abs_err += abs(ind_ratings[j] - k)
    # note: divided by 4 as 4 is max value of absolute error => returns a number in [0, 1]
    return abs_err/40


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

pop_size = 200
cxpb = 0.9
mutpb = 0.01
max_gen = 200
gen_term_cond = int(0.1 * max_gen)

toolbox = base.Toolbox()

toolbox.register("select", tools.selTournament, k=pop_size, tournsize=10)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=1/M)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)

results = np.zeros([50, 3])

for i in range(50):

    # Pick a user (randomly)
    user_id = random.randint(0, N-1)  # array index [0, N-1] -> user id [1, N]
    target_ratings = user_ratings_zero_fill[user_id, :]
    results[i, 0] = user_id
    # print(f'Selected User {user_id + 1}.')

    # Find 10 most similar users (neighbours)
    indices = np.argpartition(corr[user_id, :], -11)[-11:]
    indices = np.delete(indices, np.where(indices == user_id), axis=0)  # remove self
    # print(f'Neighbours: {indices + 1}')
    neighbourhood = np.empty([11, M], dtype=np.int32)
    for k in range(1, 11):
        neighbourhood[k, :] = user_ratings[indices[k-1], :]

    # Prepare test data -> format={movie_id, rating}
    test_ratings_user = np.empty([10, 2], dtype=np.int32)
    j = 0
    for k in range(np.shape(test_data)[0]):
        if test_data[k, 0] == user_id:
            test_ratings_user[j, 0] = test_data[k, 1] - 1
            test_ratings_user[j, 1] = test_data[k, 2]
            j += 1

    # Parameter-specific initialisations
    toolbox.register("ratings_attr", create_individual, target_ratings)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ratings_attr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, neighbours=neighbourhood)
    toolbox.decorate("mutate", check_ratings(target_ratings))
    pop = toolbox.population(n=pop_size)

    for n in range(10):

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        log = tools.Logbook()
        log.record(gen=0, **record)

        term = False  # exit condition
        gen_best_unchanged = 0
        gen = 1

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
            log.record(gen=gen, **record)

            # Termination conditions
            best = log.select("max")
            # reached max number of generations
            if gen >= max_gen:
                term = True
            # best individual of gen is <1% better than best individual of previous gen
            elif (gen > 20) and (best[-1][0] - best[-20][0]) < (0.0001 * best[-20][0]):
                term = True
            # best individual of gen is same as best individual of previous gen
            elif (gen > 20) and (best[-1][0] == best[-20][0]):
                gen_best_unchanged += 1
                if gen_best_unchanged > gen_term_cond:
                    term = True
                else:
                    gen += 1
            # else continue
            else:
                gen_best_unchanged = 0
                gen += 1
            if term:
                best_indiv = None
                for ind in offspring:
                    if ind.fitness.values == best[-1][0]:
                        best_indiv = ind
                results[i, 1] += rmse(best_indiv, test_ratings_user)
                results[i, 2] += mae(best_indiv, test_ratings_user)

        del record

    # Unregister user-specific tools
    toolbox.unregister("ratings_attr")
    toolbox.unregister("individual")
    toolbox.unregister("population")
    toolbox.unregister("evaluate")

    # Calculate mean values of metrics over 10 iterations for current user
    results[i, 1] /= 10
    results[i, 2] /= 10


print('\nDone.\n')
table_cols = ['User ID', 'RMSE (avg)', 'MAE (avg)']
df = pd.DataFrame(data=results, columns=table_cols)
df = df.astype({'User ID': np.int32})
df = df.sort_values('User ID')  # Sort by user id, ascending
# df.to_csv(path_or_buf='./eval_50.csv', index=False)  # Save to csv (report)
print(df.to_string(index=False, formatters=({'RMSE (avg)': '{:,.5f}'.format, 'MAE (avg)': '{:,.5f}'.format})))
print(f'Mean RMSE for all 50 Users: {np.mean(df["RMSE (avg)"]):.5f}')
print(f'Mean MAE for all 50 Users: {np.mean(df["MAE (avg)"]):.5f}')
