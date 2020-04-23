import argparse, yaml, os
from Factory import set_problem
from DifferentialEvolution import evolve

os.makedirs("./tmp", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('prob', type=argparse.FileType('r'))
parser.add_argument('prof', type=argparse.FileType('r'))
args = parser.parse_args()

problems = yaml.safe_load(args.prob)
prof = yaml.safe_load(args.prof)

for problem_name in problems["names"]:
    os.makedirs(f"./tmp/{problem_name}", exist_ok=True)
    problem = set_problem(problem_name)
    for seed in range(prof["seed"][0], prof["seed"][1]):
        X, F, c_eval = \
        evolve(problem, n_var=prof["n_var"], n_eval=prof["n_eval"], n_pop=prof["n_pop"],
               diff_mode=prof["diff_mode"],
               F=prof["F"], CR=prof["CR"], p=prof["p"],
               adapt_params=prof["adapt_params"], adapt_strategy=prof["adapt_strategy"], indicator=prof["indicator"], n_step=prof["n_step"],
               epsilon=prof["epsilon"], seed=seed, is_print=prof["is_print"], file=f"./tmp/{problem_name}/{prof['file_prefix']}{seed}.csv")
