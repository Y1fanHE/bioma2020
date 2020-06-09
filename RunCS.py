#################################################################
#                        ABSTRACT OF CODE                       #
#                -------------------------------                #
# Author: Yifan He (he.yifan.xs@alumni.tsukuba.ac.jp)           #
#                -------------------------------                #
# This file runs the program for Cuckoo Search (CS) Tuned on    #
# 30-D problems.                                                #
#                -------------------------------                #
# The parameters are set as follows.                            #
#  A) repetition: 31, max eval: 300,000 tolerance: 1e-10        #
#  B) N: 20, alpha: 1e-07, beta: [0.1, 1.9], pa: 0.1            #
#  C) mode of levy flight: x = x + L(alpha, beta)               #
#  D) mode of replacement: x = xbest + 0.5 * (xrand - xbest)    #
#  E) adaptive settings: None                                   #
#                -------------------------------                #
# {seed, best fitness, evaluation cost} are recorded.           #
#                -------------------------------                #
# Testing problems are                                          #
#  A) Unimodal:   Sphere, Sum squares, Zakharov, Rosenbrock     #
#  B) Multimodal: Ackley, Alpine N.1, Periodic, Rastrigin,      #
#                 Schwefel, Styblinki-Tang, Griewank, Salomon,  #
#                 Xin-She Yang's N.2, N.3 and N.4               #
#                -------------------------------                #
# Please follow the paper for the further information.          #
#################################################################

import argparse, yaml, os
from PyBenchFCN import Factory
from CuckooSearch import evolve

betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

# betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# betas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

os.makedirs("./tmp", exist_ok=True)

# problem_names = ["sphere", "sumsquares", "zakharov", "rosenbrock",
#                  "ackley", "alpinen1", "periodic", "rastrigin",
#                  "schwefel", "styblinskitank", "griewank", "salomon",
#                  "xinsheyangn2", "xinsheyangn3", "xinsheyangn4"]

problem_names = ["sphere", "sumsquares"]
problem_names = ["zakharov", "rosenbrock"]
problem_names = ["ackley", "alpinen1"]
problem_names = ["periodic", "rastrigin"]
problem_names = ["schwefel", "styblinskitank"]
problem_names = ["griewank", "salomon"]
problem_names = ["xinsheyangn2", "xinsheyangn3"]
problem_names = ["xinsheyangn4"]


# for problem_name in problem_names:

#     os.makedirs(f"./tmp/{problem_name}", exist_ok=True)
#     problem = Factory.set_sop(problem_name, n_var=30)

#     for beta in betas:
#         for seed in range(1000, 1031):

#             file = open(f"./tmp/{problem_name}/{beta}.csv", "a")

#             X, F, c_eval = \
#                 evolve(problem, n_eval=300000, n_pop=20,
#                        levy_mode="lf", replace_mode="best-to-rand",
#                        alpha=1e-07, beta=beta, pa=0.1, adapt_strategy="none",
#                        epsilon=0, seed=seed, is_print=False, file=f"./tmp/{problem_name}/{beta}_{seed}.csv")

#             file.write(f"{seed},{min(F)},{c_eval}\n")
#             file.close()

epsilons = yaml.safe_load(open("epsilons.yml", "r"))
for problem_name in problem_names:

    if problem_name == "sphere": beta = 0.6
    if problem_name == "sumsquares": beta = 0.5
    if problem_name == "zakharov": beta = 0.3
    if problem_name == "rosenbrock": beta = 0.4
    if problem_name == "ackley": beta = 0.5
    if problem_name == "alpinen1": beta = 0.3
    if problem_name == "periodic": beta = 0.6
    if problem_name == "rastrigin": beta = 0.4
    if problem_name == "schwefel": beta = 0.2
    if problem_name == "styblinskitank": beta = 0.4
    if problem_name == "griewank": beta = 0.2
    if problem_name == "salomon": beta = 0.2
    if problem_name == "xinsheyangn2": beta = 0.3
    if problem_name == "xinsheyangn3": beta = 0.5
    if problem_name == "xinsheyangn4": beta = 0.6

    os.makedirs(f"./tmp/{problem_name}", exist_ok=True)
    problem = Factory.set_sop(problem_name, n_var=30)

    for seed in range(1000, 1031):

        X, F, c_eval = \
            evolve(problem, n_eval=300000, n_pop=20,
                   levy_mode="lf", replace_mode="best-to-rand",
                   alpha=1e-07, beta=beta, pa=0.1, adapt_strategy="none",
                   epsilon=epsilons["CS"][problem_name], seed=seed, is_print=False, file=f"./tmp/{problem_name}/CS_{seed}.csv")
