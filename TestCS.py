import os
import numpy as np
from Factory import set_problem
import BasicCuckooSearch as BACS
import RandomParameterCuckooSearch as RPCS
import JADEForCuckooSearch as JACS
import ParameterEvolutionCuckooSearch as PECS

repeat = 31

"""
problems = [
            "sphere", "rotated_hyper_ellipsoid", "different_power", "weighted_sphere",          # bowl shape
            "dixon_price", "rosenbrock_chain", "rosenbrock_star", "k_tablet",                   # valley shape
            "zakharov",                                                                         # plate shape
            "ackley", "rastrigin", "griewank", "levy", "schwefel", "xin_she", "schaffer"        # many local optima
           ]
"""

problems = ["sphere", "rotated_hyper_ellipsoid", "different_power", "weighted_sphere"]
problems = ["dixon_price", "rosenbrock_chain", "rosenbrock_star", "k_tablet"]
problems = ["zakharov", "ackley", "rastrigin", "griewank"]
problems = ["levy", "schwefel", "xin_she", "schaffer"]

n_var = 30

n_pop = 20
max_eval = 300000
max_gen = int (max_eval / n_pop)
epsilon = 1e-10

alpha_1 = 1e-07
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

    # set problem
    problem = set_problem(problem_name)

    for seed in range(1000, 1000 + repeat):

        # set file name of record for PECS
        fname = f"./tmp/{problem_name}/PECS_{seed}.csv"

        # Parameter Evolution Cuckoo Search
        X1, F1, n_eval1 =\
        PECS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, levy_alg, pa_1,
                      betal, betau, step_gen, indicator,
                      alpha_2, beta_2,
                      epsilon, seed, fname)
        
        # set file name of record for JACS
        fname = f"./tmp/{problem_name}/JACS_{seed}.csv"

        # JADE for Cuckoo Search
        X2, F2, n_eval2 =\
        JACS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, levy_alg, pa_1,
                      (betal + betau) / 2, betal, betau, c,
                      epsilon, seed, fname)

        # Random Parameter Cuckoo Search
        X3, F3, n_eval3 =\
        RPCS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, betal, betau, levy_alg, pa_1,
                      epsilon, seed)

        # Basic Cuckoo Search
        X4, F4, n_eval4 =\
        BACS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, beta, levy_alg, pa_1,
                      epsilon, seed)
