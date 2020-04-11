import os
import numpy as np
from Factory import set_problem
import JADE
import ParameterEvolutionDifferentialEvolution as PEDE

repeat = 31

problems = [
            "sphere", "rotated_hyper_ellipsoid", "different_power", "weighted_sphere",          # bowl shape
            "dixon_price", "rosenbrock_chain", "rosenbrock_star", "k_tablet",                   # valley shape
            "zakharov",                                                                         # plate shape
            "ackley", "rastrigin", "griewank", "levy", "schwefel", "xin_she", "schaffer"        # many local optima
           ]

problems = ["griewank"]

n_var = 30

n_pop = 100
max_eval = 300000
max_gen = int (max_eval / n_pop)
epsilon = 0

Fl, Fu = 0, 1
CRl, CRu = 0, 1
mu_F, mu_CR = 0.5, 0.5
p = 0.05

n_step = 20
alpha = 0.1
beta = 1.0
c = 0.1

# create ./tmp to save all records
os.makedirs("./tmp", exist_ok=True)

for problem_name in problems:

    # create folder to save record for the problem
    os.makedirs(f"./tmp/{problem_name}", exist_ok=True)

    # set problem
    problem = set_problem(problem_name)

    for seed in range(1000, 1000 + repeat):

        # set file name of record for PEDE
        fname = f"./tmp/{problem_name}/PEDE_{seed}.csv"

        # Parameter Evolution Cuckoo Search
        X1, F1, n_eval1 =\
            PEDE.optimize(problem, n_var, n_pop, max_gen, max_eval,
                          p, Fl, Fu, CRl, CRu,
                          n_step, alpha, beta,
                          epsilon, seed, fname)

        X2, F2, n_eval2 =\
            JADE.optimize(problem, n_var, n_pop, max_gen, max_eval,
                          mu_CR, mu_F, p,
                          c,
                          epsilon, seed)

        print(f"F: PEDE={min(F1)}, JADE={min(F2)}; n_eval: PEDE={n_eval1}, JADE={n_eval2}")