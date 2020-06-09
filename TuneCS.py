import argparse, yaml, os
from PyBenchFCN import Factory
from CuckooSearch import evolve

n_steps = [1, 2, 5, 10, 20, 50]

os.makedirs("./tmp", exist_ok=True)

problem_names = ["rastrigin"]
for problem_name in problem_names:

    os.makedirs(f"./tmp/{problem_name}", exist_ok=True)
    problem = Factory.set_sop(problem_name, n_var=30)

    for n_step in n_steps:
        for seed in range(1000, 1021):

            file = open(f"./tmp/{problem_name}/{n_step}.csv", "a")

            X, F, c_eval = \
                evolve(problem, n_eval=150000, n_pop=20,
                       levy_mode="lf", replace_mode="best-to-rand",
                       alpha=1e-07, beta=[0.1, 1.0, 1.9], pa=0.1,
                       adapt_params=["beta"], adapt_strategy="ga",
                       params_of_adapt_strategy={"pc_adapt" : 0.7,
                                                 "pm_adapt" : 0.3,
                                                 "crx_mode_adapt" : "uniform_sum",
                                                 "mut_mode_adapt" : "levy",
                                                 "alpha_adapt" : 0.1,
                                                 "beta_adapt" : 1.0},
                       indicator="dfii", n_step=n_step,
                       epsilon=1e-05, seed=seed, is_print=False, file="none")

            file.write(f"{seed},{min(F)},{c_eval}\n")
            file.close()
