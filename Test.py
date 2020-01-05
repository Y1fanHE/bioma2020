import numpy as np
from Factory import set_problem
import BasicCuckooSearch as BCS
import ParameterEvolutionCuckooSearch as PECS
import RandomParameterCuckooSearch as RPCS

repeat = 15

problems = ["k_tablet", "different_power", "griewank", "xin_she"]
n_var = 50

n_pop = 20
max_eval = 4000000
max_gen = int (max_eval / n_pop)

alpha_1 = 1e-09
beta_1 = 0.3
levy_alg = "mantegna"
pa = 0.1

betal, betau = 0.3, 1.9
step_gen = 150
alpha_2 = 0.1
beta_2 = 1.0

for problem_name in problems:

    print("\n****************************************")

    # set problem
    problem = set_problem(problem_name)

    # print table header
    print("",
          "{:>5}".format("Seed"),
          "{:>10}".format("BCS"),
          "{:>10}".format("RPCS"),
          "{:>10}".format("PECS"),
          "",
          sep="|")
    print("",
          "{:->5}".format(""),
          "{:->10}".format(""),
          "{:->10}".format(""),
          "{:->10}".format(""),
          "",
          sep="|")

    # open record file
    f = open(f"./tmp/{problem_name}.csv", "a")

    # write table header
    f.write("Seed,BCS,RPCS,PECS\n")

    for seed in range(1000, 1000 + repeat):

        # Basic Cuckoo Search
        X1, F1 =\
        BCS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                     alpha_1, beta_1, levy_alg, pa,
                     seed)

        # Random Parameter Cuckoo Search
        X2, F2 =\
        RPCS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, betal, betau, levy_alg, pa,
                      seed)

        # Parameter Evolution Cuckoo Search
        X3, F3 =\
        PECS.optimize(problem, n_var, n_pop, max_gen, max_eval,
                      alpha_1, levy_alg, pa,
                      betal, betau, step_gen,
                      alpha_2, beta_2,
                      seed)

        # print results of three algorithms with the seed
        print("",
            "{:>5}".format(seed),
            "{:>10.3e}".format(min(F1)),
            "{:>10.3e}".format(min(F2)),
            "{:>10.3e}".format(min(F3)),
            "",
            sep="|")
        
        # write results of three algorithms with the seed
        s = str(seed) + ","\
            "{:.5e}".format(min(F1)) + "," +\
            "{:.5e}".format(min(F2)) + "," +\
            "{:.5e}".format(min(F3)) + "\n"
        f.write(s)

    # close record file
    f.close()