from plotting import *
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from sim_annealing import simulated_annealing
from evolution_strategies import evolution_strategies

# import random as rnd
from time import time
import signal


def egg_holder(x, counter=None):
    fun = 0.0
    # Be aware of python indexing, it starts at 0 for the first element
    # Also, range loops until the specified value, i.e. the specified number is not included
    for i in range(len(x) - 1):
        fun += -(x[i + 1] + 47) * np.sin(np.sqrt(np.abs(x[i + 1] + 0.5 * x[i] + 47))
                                         ) - x[i] * np.sin(np.sqrt(np.abs(x[i] - x[i + 1] - 47)))

    return fun


if __name__ == '__main__':
    # route = zip(x,y)
    result_table = []
    f_mins = []
    x_mins = []

    class TimeoutException(Exception):   # Custom exception class
        pass

    def timeout_handler(signum, frame):   # Custom signal handler
        raise TimeoutException

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    minimum_global = []
    minimum_pair = []
    # mu_lambda
    results_table = []
    # mu_lambda = [ (mu, 7*mu) for mu in range(5,21)  ]
    # for (mu,lambd) in mu_lambda:
    # print("mu, lambda = {} {}".format(mu, lambd))
    results = []
    # while len(results) < 50:
    #     done=False
    #     while not done:
    #         try:
    #             # Code hangs sometimes, seems to hand at np.random.choice line most of the time.
    #             signal.alarm(5)
    # minimum, population, min_x, f_mean_record, parents_list, offspring_list = evolution_strategies(
    # egg_holder, n_dims=5, mu=19, lambd=133, global_recomb=True, plus_recomb=False)
    #             # print("Minimum:", minimum)
    #             if minimum > -3630:
    #                 results.append(minimum)
    #                 print(len(results))
    #             # signal.alarm(0)
    #         except Exception as e:
    #             print(e)
    #         else:
    #         # This block is only executed if the try-block executes without
    #         # raising an exception
    #             done=True
    #             signal.alarm(0)
    # # results_table.append(results)
    # print("Minimum found: ", min(results))
    #
    # print("plus recombination list: \n", minimum_global)
    # print("comma recombination list: \n", minimum_pair)
    # print("global recombination mean {}, sdev {}".format(np.mean(minimum_global), np.std(minimum_global)) )
    # print("pair recombination mean {}, sdev {}".format(np.mean(minimum_pair), np.std(minimum_pair)))
    #
    # means, sdevs = [], []
    # len(results_table)
    # for i in range(len(results_table)):
    #     means.append( np.mean(results_table[i]))
    #     sdevs.append( np.std(results_table[i]))
    # for i in range(len(results_table)):
    #     print("{}".format(int(means[i])))
    # means
    # min(means)

    # sdevs
    # minimum, population, min_x, f_mean_record, parents_list, offspring_list = evolution_strategies(
    # egg_holder, n_dims=5, mu=19, lambd=133, global_recomb=True, plus_recomb=False)

    # print("f_mean_record \n", f_mean_record)
    # result_table.append(results)
    # f_mins.append(f_min)
    # x_mins.append(min_x)
    # print("Population shape: ", parents_list.shape)
    # contour_plot_function(egg_holder, route=population,
    #                       min_x=min_x, populations=offspring_list[::5])
    # for i in range(1,7):
    #     title = "Generation number {} ".format(i*10)
    #     plot_generations(egg_holder, title=title, population=parents_list[10*i])

    # plt.plot(f_values)
    # plt.plot(f_mean_record)
    # plt.title('Mean f_value of population vs generation')
    # plt.ylabel('Mean f_value of population')
    # plt.xlabel('generation')
    # plt.show()
    # print(results)
    # minimums = []
    temp_intervals = [10 * x for x in range(3, 30)]
    minimum_list = []
    for temp_interval in temp_intervals:
        print(temp_interval)
        minimums = []
        while len(minimums) < 10:
            minimum, x_s, min_x, f_values = simulated_annealing(
            egg_holder, n_dims=5, temp_interval=temp_interval)
            if minimum > -3620:
                minimums.append(minimum)
        minimum_list.append(minimums)


    # need to filter for x < -3630
    minimum_list
    min(minimum_list)
    np.min(minimum_list)
    minimum_list[3][4]

    for i in range(len(minimum_list)):
        for j in range(10):
            if minimum_list[i][j] < -3630:
                minimum_list[i][j] = -3000


    means, sdevs = [], []
    len(minimum_list)
    for i in range(len(minimum_list)):
        means.append( np.mean(minimum_list[i]))
        sdevs.append( np.std(minimum_list[i]))
    print(temp_intervals[10])
    for i in temp_intervals:
        print("{}".format(int(i)))
    for i in range(len(minimum_list)):
        print("{}".format(int(means[i])))
        if int(means[i]) == -3129:
            print(i)

    min(means)

    for i in range(len(minimum_list)):
        print("{}".format(int(sdevs[i])))






    # Plot simulated_annealing:
    # f_min, x_s, min_x, results, f_values = simulated_annealing(
    # egg_holder, n_dims=5)
    # contour_plot_function(egg_holder, route=x_s[::5], min_x = min_x)

    # minimum = min(f_mins)
    # min_x = x_mins[f_mins.index(min(f_mins))]
    # print("The minimum found is {} at {} with criterion {} ".format(
    #     minimum, min_x, f_mins.index(min(f_mins)) - 1))

    # print(min(results[0]))3
