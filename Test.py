import os
import numpy as np
from Factory import set_problem
import BasicCuckooSearch as BCS
import ParameterEvolutionCuckooSearch as PECS
import RandomParameterCuckooSearch as RPCS

repeat = 15

problems = ["rastrigin", "griewank", "rosenbrock", "sphere"]
n_var = 50

n_pop = 20
max_eval = 4000000
max_gen = int (max_eval / n_pop)

alpha_1 = 1e-09
beta_1 = 0.3
levy_alg = "default"
pa_1 = 0.1

betal, betau = 0.1, 1.9
step_gen = 200
indicator = "df/f"
alpha_2 = 1.0
beta_2 = 1.0

# create ./tmp to save all records
os.makedirs("./tmp", exist_ok=True)

for problem_name in problems:

    # create folder to save record for the problem
    os.makedirs(f"./tmp/{problem_name}", exist_ok=True)

    print("\n****************************************")

    # set problem
    problem = set_problem(problem_name)

    # print table header
    print("", "{:>5}".format("Seed"),
          "{:>10}".format("PECS"), "",
          sep="|")
    print("", "{:->5}".format(""),
          "{:->10}".format(""), "",
          sep="|")

    for seed in range(1000, 1000 + repeat):

        # set file name of record
        fname = f"./tmp/{problem_name}/{seed}.csv"

        # Parameter Evolution Cuckoo Search
        X, F =\
        PECS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, levy_alg, pa_1,
                      betal, betau, step_gen, indicator,
                      alpha_2, beta_2,
                      seed, fname)

        # print results of three algorithms with the seed
        print("", "{:>5}".format(seed),
              "{:>10.3e}".format(min(F)), "",
              sep="|")