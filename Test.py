import os
import numpy as np
from Factory import set_problem
import BasicCuckooSearch as BACS
import RandomParameterCuckooSearch as RPCS
import JADEForCuckooSearch as JACS
import ParameterEvolutionCuckooSearch as PECS

repeat = 31
epsilon = 1e-05

problems = ["ackley", "rastrigin", "griewank", "levy", "schwefel", "xin-she", "schaffer"]

n_var = 50

n_pop = 20
max_eval = 4000000
max_gen = int (max_eval / n_pop)

alpha_1 = 1e-09
betas = [0.6, 0.5, 0.3, 0.3, 0.5, 0.4, 0.2, 0.3] # tuned by Ikeda
levy_alg = "default"
pa_1 = 0.1

betal, betau = 0.1, 1.9 # boundary of stability parameters
step_gen = 100
indicator = "df/f"
alpha_2 = 0.1 # same as JADE
beta_2 = 1.0 # same as JADE

c = 0.1 # learning rate in JACS

# create ./tmp to save all records
os.makedirs("./tmp", exist_ok=True)

for problem_name, beta in zip(problems, betas):

    # create folder to save record for the problem
    os.makedirs(f"./tmp/{problem_name}", exist_ok=True)

    print("\n****************************************")

    # set problem
    problem = set_problem(problem_name)

    # print table header
    print("", "{:>5}".format("Seed"),
          "{:>10}".format("PECS"),
          "{:>10}".format("JACS"),
          "{:>10}".format("RPCS"),
          "{:>10}".format("BACS"),
          "", sep="|")
    print("", "{:->5}".format(""),
          "{:->10}".format(""),
          "{:->10}".format(""),
          "{:->10}".format(""),
          "{:->10}".format(""),
          "", sep="|")

    for seed in range(1000, 1000 + repeat):

        # set file name of record for PECS
        fname = f"./tmp/{problem_name}/PECS_{seed}.csv"

        # Parameter Evolution Cuckoo Search
        X1, F1 =\
        PECS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, levy_alg, pa_1,
                      betal, betau, step_gen, indicator,
                      alpha_2, beta_2,
                      epsilon, seed, fname)
        
        # set file name of record for JACS
        fname = f"./tmp/{problem_name}/JACS_{seed}.csv"

        # JADE for Cuckoo Search
        X2, F2 =\
        JACS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, levy_alg, pa_1,
                      (betal + betau) / 2, betal, betau, c,
                      epsilon, seed, fname)

        # Random Parameter Cuckoo Search
        X3, F3 =\
        RPCS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, betal, betau, levy_alg, pa_1,
                      epsilon, seed)

        # Basic Cuckoo Search
        X4, F4 =\
        BACS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, beta, levy_alg, pa_1,
                      epsilon, seed)

        # print results of three algorithms with the seed
        print("", "{:>5}".format(seed),
              "{:>10.3e}".format(min(F1)),
              "{:>10.3e}".format(min(F2)),
              "{:>10.3e}".format(min(F3)),
              "{:>10.3e}".format(min(F4)),
              "", sep="|")