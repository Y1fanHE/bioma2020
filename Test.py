import numpy as np
from Factory import set_problem
import BasicCuckooSearch as BCS
import ParameterEvolutionCuckooSearch as PECS
import RandomParameterCuckooSearch as RPCS

repeat = 15

problems = ["rastrigin"]
n_var = 50

n_pop = 20
max_eval = 4000000
max_gen = int (max_eval / n_pop)

alpha_1 = 1e-09
beta_1 = 0.3
levy_alg = "mantegna"
pa_1 = 0.1

betal, betau = 0.3, 1.9
step_gen = 200
indicator = "df/f"
alpha_2 = 1.0
beta_2 = 1.0
pa_2 = 0.2

for problem_name in problems:

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

        # Parameter Evolution Cuckoo Search
        X, F =\
        PECS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, levy_alg, pa_1,
                      betal, betau, step_gen, indicator,
                      alpha_2, beta_2,
                      seed, f"./tmp/{problem_name}_{seed}.csv")

        # print results of three algorithms with the seed
        print("", "{:>5}".format(seed),
            "{:>10.3e}".format(min(F)), "",
            sep="|")