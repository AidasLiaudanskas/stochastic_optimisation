import scipy as sp
import pandas as pd
import seaborn as sns
from plotting import *
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
# from sim_annealing import simulated_annealing
# from evolution_strategies import evolution_strategies

# import random as rnd
from time import time
import signal
# rnd.seed(5)
# rnd.seed(time())
# print("hi")


def generate_x(n_dims):
    x = [(rnd.random() - 0.5) * 1024 for i in range(n_dims)]
    return x


def mix_and_match_n(parents):
    no_arguments = len(parents)
    n_dim = len(parents[0])
    for parent in parents:
        assert len(
            parent) == n_dim, "Make sure all parents have the same dimension"
    child = np.empty((n_dim))
    # Reorder into arrays with elements from the same dimension
    reordered_parents = list(zip(*parents))
    for i in range(n_dim):
        child[i] = np.random.choice(reordered_parents[i])
    return child


def generate_offspring(n_dims, mu, lambd, parents=0, global_recomb=False):
    # Limits Specific to task at hand
    upper_lim = 512.0
    lower_lim = -512.0
    # For the first run:
    offspring = np.empty((lambd, n_dims))
    # print("Parent shape is", np.array(parents).shape)
    if np.array(parents).shape == ():
        for i in range(lambd):
            offspring[i] = generate_x(n_dims)
    else:
        assert np.array(parents).shape == (
            mu, n_dims), "Check parents shape in generate_offspring()"
        for i in range(lambd):
            # Mix and match mu number of parents to make lambda offspring:
            # Not using the same parent twice.
            # Choise between global Discrete recombination and choosing from 2 parents
            if global_recomb:
                temp_parents = parents
            else:
                temp_parents = rnd.sample(list(parents), 2)
            child = mix_and_match_n(temp_parents)
            cnt = 0
            while any(child[i] > upper_lim for i in range(n_dims)) or any(child[i] < lower_lim for i in range(n_dims)):
                # Got into endless loop sometimes, need to fix this:
                cnt += 1
                child = mix_and_match_n(temp_parents)
                if cnt > 5:
                    child = generate_x(n_dims)
            offspring[i] = child
    return offspring  # lambda by n_dim matrix


def evaluate_population(function, population):
    lambd = population.shape[0]
    f_values = np.empty(lambd)
    for i in range(lambd):
        f_values[i] = function(population[i])
    return f_values


def select_parents(f_values, population, mu):
    # Sort both lists according to f_values

    population = population[f_values.argsort()]
    f_values = f_values[f_values.argsort()]
    # Take mu elements with the smallest values as parents:
    # print("f_values", f_values)
    parents = population[:mu]
    # print("parents", parents)
    return parents


def alphas_to_A(params, n_dim):
    for i in range(n_dim):
        for j in range(n_dim):
            # if i!=j: # Might need this to avoid tampering with covariances
            params['A_inv'][i, j] = 0.5 * np.tan(2 * params['alpha_ij'][i, j]) * (
                params['sigmas'][i] - params['sigmas'][j])
        return params


def A_to_alphas(params, n_dim):
    eps = np.finfo(np.float32).eps  # Avoid division by zero
    for i in range(n_dim):
        for j in range(n_dim):
            # if i!=j: # Might need this to avoid tampering with covariances
            params['alpha_ij'][i, j] = 0.5 * np.arctan(
                2 * params['A_inv'][i, j] / (params['sigmas'][i] - params['sigmas'][j] + eps))
    # return alphas:
    return params


def mutate(parents, params, n_dim):
    # Tried to work with covariances, couldn't make it work. Left only variances
    if not params:
        # Initialize parameters:
        params['tau'] = 1 / np.sqrt(2 * np.sqrt(n_dim))
        params['tau_dash'] = 1 / np.sqrt(2 * n_dim)
        params['beta'] = 0.0873
        # Need a positive definite matrix:
        rand_mat = np.random.normal(size=(n_dim, n_dim))
        params['sigmas'] = np.ones(n_dim)
        params['A_inv'] = 0.5 * \
            (np.dot(rand_mat, rand_mat.T) + params['sigmas'])
        # print("Fresh A_inv \n", params['A_inv'])
        params['alpha_ij'] = np.zeros((n_dim, n_dim))
        params = A_to_alphas(params, n_dim)
        # print("Alphas: \n", params['alpha_ij'])

    # do mutation parameter updates:
    N0 = np.random.normal()
    Ni = np.random.normal(size=(n_dim))
    Nij = np.random.normal(size=(n_dim, n_dim))
    # Mutate standard deviations:
    params['sigmas'] = params['sigmas'] * \
        np.exp(params['tau_dash'] * N0 + params['tau'] * Ni)
    # print("Sigmas: ", params['sigmas'])
    # for i in range(n_dim):
    #     params['A_inv'][i, i]=params['A_inv'][i, i] * \
    #         np.exp(params['tau_dash'] * N0 + params['tau'] * Ni[i])
    # Mutate rotation angles:
    # TODO: Ensure this part is correct. Might not need to mutate diagonal elements with these angles!
    params['alpha_ij'] += params['beta'] * Nij
    params = alphas_to_A(params, n_dim)
    # reset the diagonal of A_inv:
    # for i in range(n_dim):
    #     params['A_inv'][i,i] = 0
    params['A_inv'] = params['A_inv'] + np.diag(params['sigmas'])
    # print("Mutated A_inv \n", params['A_inv'])
    # print("sigmas: \n", params["sigmas"])
    # Getting too complicated let's try with just the covariances:
    A = np.diag(params['sigmas'])
    # Check if a is positive definite:
    # print(params['A_inv'])
    # print("A is positive definite! \n", np.linalg.cholesky(params['A_inv']))
    # try:
    #     np.linalg.cholesky(params['A_inv'])
    # except Exception as e:
    #     raise
    # print(params['A_inv'])
    # A=params['A_inv']
    for individual in parents:
        means = [0 for _ in range(n_dim)]
        randomness = np.random.multivariate_normal(means, A)
        # print("Mutating by: \n", randomness)
        individual += randomness
    return parents, params


def evolution_strategies(function, n_dims, mu=5, lambd=35, global_recomb=False, plus_recomb=True):
    np.random.seed(5)
    n_iters = 10000
    start_time = time()
    counter = 0
    parents = np.zeros((mu, n_dims))
    params = {}
    parent_list = np.empty((n_iters // 70 - 1, mu, n_dims))
    offspring_list = np.empty((n_iters // 70 - 1, lambd, n_dims))
    f_mean_record = []
    f_record = np.empty((n_iters // 70 - 1, lambd))
    # lambda is a keyword in Python, use lambd instead
    # mu:lambda recommended as 1:7
    # generate random population:
    population = generate_offspring(n_dims=n_dims, mu=mu, lambd=lambd)
    population, params = mutate(population, params, n_dims)
    f_values = evaluate_population(function, population)
    counter += lambd
    # Limited to 10000 function evaluations, so have to track that.
    while counter < n_iters:
        # asses population:
        # select mu parents:
        if plus_recomb and parents[0, 0]:
            parents = select_parents(f_values, np.concatenate(
                (population, parents), axis=0), mu)
        else:
            parents = select_parents(f_values, population, mu)
        parent_list[counter // 70 - 2, ] = parents
        # Create lambda new offspring by mutation and recombination:
        # parents, params = mutate(parents, params, n_dims)
        population = generate_offspring(
            n_dims, mu, lambd, parents=parents, global_recomb=global_recomb)
        population, params = mutate(population, params, n_dims)
        offspring_list[counter // 70 - 2] = population

        # assess new population:
        f_values = evaluate_population(function, population)
        # in sync with population record
        f_record[counter // 70 - 2] = f_values
        counter += lambd
        # update archives:
        f_mean_record.append(np.mean(f_values))
        if len(f_mean_record) > 3 and (np.abs(f_mean_record[-3] - f_mean_record[-1]) < 10):
            population = generate_offspring(n_dims=n_dims, mu=mu, lambd=lambd)
        # Make a record of the parents over iteration.
    time_elapsed = time() - start_time
    minimum = f_record.min()
    min_row, min_column = np.where(f_record == minimum)
    min_x = offspring_list[min_row[0]][min_column[0]]
    print("Time taken is {} seconds and the minimum found is {} at {} ".format(
        time_elapsed, minimum, min_x))
    return minimum, population, min_x, f_mean_record, parent_list, offspring_list
