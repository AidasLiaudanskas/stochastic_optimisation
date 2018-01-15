import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import random as rnd
from time import time
# rnd.seed(5)
# rnd.seed(time())


def generate_x(n_dims):
    x = [(rnd.random() - 0.5) * 1024 for i in range(n_dims)]
    return x


def findT0(function, n_dims, n_iters=50):
    # Function to estimate the T0 as suggested by White[1984]
    f_values = []
    for i in range(n_iters):
        x = generate_x(n_dims)
        f_values.append(function(x))
    return np.std(f_values)


def T_decrease_coeff(current_T, sigma_f):
    # temperature decrease coefficient as suggested by Huang et al.[1986]
    # Sometimes division by 0 is encountered, so small value is added to avoid this
    alpha = max(0.5, np.exp(- 0.7 * current_T / (sigma_f + 0.00000000005)))
    return alpha


def generate_u(n_dims):
    u = [(rnd.random() - 0.5) * 2 for i in range(n_dims)]
    return u


def update_D(D, u):
    # From the lecture notes, Dr. Parks version:
    alpha, omega = 0.1, 2.1
    R = np.diag(np.abs(np.dot(D, u)))
    D_new = (1 - alpha) * D + alpha * omega * R
    return D_new


def simulated_annealing(function, upper_lim=512, lower_lim=-512, n_dims=2, restart_criterion=70, temp_interval=150):
    start_time = time()
    # rnd.seed(5)
    # Will append all the temperatures to this array
    T = [findT0(function, n_dims, n_iters=50)]
    counter = 0  # No more than 10k evals allowed.
    # Choose initial solution:
    x = [generate_x(n_dims)]  # Keep track of all the solutions in this array
    # Stop conditions will be counter exceeding 10k or no
    # improvement(less than 0.5 abs diff in f_value) was made in the last 500 iters
    # Decrease in f with new solution:
    f_values = [function(x[-1])]
    counter += 1
    D = np.eye(n_dims)
    # Need to add restarts to improve performance. Logic:
    # If global optimum is not improved within 500 iterations, restart from global optimum
    restart_counter = 0
    restart_limit = 500
    while counter < 10000:
        # Decrease temperature by alpha every 150 iterations.
        # print("Counter= {}".format(counter))
        for i in range(temp_interval):
            # constraint handling:
            while True:
                u = generate_u(n_dims)
                x_new = x[-1] + np.dot(D, u)
                if all(x_i < upper_lim for x_i in x_new) and all(x_i > lower_lim for x_i in x_new):
                    break
                # if all(x_i > lower_lim for x_i in x_new):
                #     break
            actual_step = np.sqrt(np.sum((np.dot(D, u))**2))
            # print("actual_step", actual_step)
            if len(f_values) > 50:
                # if np.linalg.norm(np.subtract(x[-50], x[-1])) < 2:
                if np.abs(f_values[-50] - f_values[-1]) < restart_criterion:
                    x_new = generate_x(n_dims)
            func_new = function(x_new)
            counter += 1
            func_diff = func_new - f_values[-1]
            if func_new - min(f_values) > 0:
                restart_counter += 1
            else:
                restart_counter = 0
            if restart_counter > restart_limit:
                x_new = x[f_values.index(min(f_values))]
                # A bit inefficient, but the only way with restarts. Should be worth it.
                # My bit below:
                # If norm of current x minus x 50 iterations ago is
                # Less than n, restart (avoids getting stuck in local optima for too long)

            if func_diff < 0.0:
                x.append(x_new)
                f_values.append(func_new)
                D = update_D(D, u)  # Only if the solution is accepted

            else:
                accept_prob = np.exp(- func_diff / (T[-1] * actual_step))
                # accept_prob = np.exp(- func_diff / (T[-1] ))
                if rnd.random() < accept_prob:
                    x.append(x_new)
                    f_values.append(func_new)
                    D = update_D(D, u)  # Only if the solution is accepted

        alpha_T = T_decrease_coeff(T[-1], np.std(f_values[-150:-1]))
        T.append((T[-1]) * alpha_T)
    time_elapsed = time() - start_time
    minimum = min(f_values)
    min_x = x[f_values.index(min(f_values))]
    print("Time taken is {} seconds and the minimum found is {} at {} ".format(
        time_elapsed, minimum, min_x))
    # print("Final T = {}; Last f_value = {};".format(
    #     T[-1], f_values[-1]))

    return minimum, x, min_x, f_values
