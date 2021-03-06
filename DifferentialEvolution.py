import numpy as np
from LevyVector import levy
import AdaptiveStrategy as Adaption

def evolve(problem, n_eval=300000, n_pop=100,
           diff_mode="de/curr-to-pbest/1",
           F=0.5, CR=0.9, p=0.05,
           adapt_params=[], adapt_strategy="none", params_of_adapt_strategy={},
           indicator="none", n_step=1,
           epsilon=1e-14, seed=1000, is_print=True, file="none"):

    '''
    Differential Evolution (DE) with Self-adaptive strategy

    Problem Parameters
    ----------
    problem: object
      - optimization problem to be solved

    Running Parameters
    ----------
    n_eval: int
      - maximum number of evaluations
    n_pop: int
      - population size
    epsilon: float
      - tolerance of algorithm running
    seed: int
      - seed of random number
    is_print: bool
      - whether print results on screen or not
    file: string
      - file to record history
      - valid input:
        + "none": do not record
        + otherwise, write to the file name

    Algorithm Parameters
    ----------
    diff_mode: string
      - mode of differential mutation
      - valid input: { "de", "de/curr-to-pbest", "de/curr-to-pbest/1" }

    Variation Parameters
    ----------
    F, CR, p: float OR tuple
      - scaling factor, crossover rate, elite rate
      - valid input:
        + for strategy type 1, input (lower, upper)
        + for strategy type 2, input (lower, mean, upper)
        + for non-adaptive, input fixed float
      * change { adapt_params, adapt_strategy } correspondingly

    Strategy Parameters
    ----------
    adapt_params: list of string
      - parameters which are set to be adaptive
      - valid input: { ["F"], ["CR"], ["F, CR"] }
      * change { F, CR, adapt_strategy } correspondingly
    adapt_strategy: string
      - parameter variation method in adaptive strategy
      - valid input: { "none", "random", "best_levy", "ga", "cauchy", "jade" }
      * change { F, CR, adapt_params } correspondingly
    indicator: string
      - indicator used to assess quality of parameters
        automatically set to "none" if adapt_strategy == "none" or "random"
      - valid input: { "dfii/dfi" } !TODO - increase more indicators
    n_step: int
      - frequency to update parameters
        e.g. n_step=20 means to parameters every 20 generations

    Return
    ----------
    X: 2D-Array
      - decision variables of individuals (final evaluation)
    F: 1D-Array
      - fitness variables of individuals (final evaluation)
    c_eval: int
      - number of evaluations
    '''

    if type(seed) == int: np.random.seed(seed)                          # Set seed for random number generator

    evaluate_fitness = problem.f                                        # Get problem information
    xl, xu = problem.boundaries
    n_var = problem.n_var

    #####################
    # PARAMETER SETTING #
    #####################

    strategy_type1 = ["best_levy", "ga", "random"]                      # Strategy with random parameter initialization
    strategy_type2 = ["cauchy"]                                         # Strategy with cauchy inintialization
    strategy_type3 = ["jade"]                                           # Strategy with initialization in JADE

    if adapt_strategy == "none":
        Fs = np.ones(n_pop) * F
        CRs = np.ones(n_pop) * CR

    if adapt_strategy in strategy_type1:                                # Initialize parameters with uniform distribution
        if "F" in adapt_params:
            Fs = np.random.uniform(F[0], F[1], n_pop)
            if type(CR) == float: CRs = np.ones(n_pop) * CR
        if "CR" in adapt_params:
            if type(F) == float: Fs = np.ones(n_pop) * F
            CRs = np.random.uniform(CR[0], CR[1], n_pop)

    if adapt_strategy in strategy_type2:                                # Initialize parameters with Cauchy distribution
        if "F" in adapt_params:
            F_mu = F[1]
            Fs = np.random.standard_cauchy(n_pop) * 0.1 + F_mu
            Fs = np.clip(Fs, F[0], F[2])
            if type(CR) == float: CRs = np.ones(n_pop) * CR
        if "CR" in adapt_params:
            CR_mu = CR[1]
            if type(F) == float: Fs = np.ones(n_pop) * F
            CRs = np.random.standard_cauchy(n_pop) * 0.1 + CR_mu
            CRs = np.clip(CRs, CR[0], CR[2])

    if adapt_strategy in strategy_type3:                                # Initialize F with Cauchy distribution, CR with Gaussian distribution
        if "F" in adapt_params:
            F_mu = F[1]
            Fs = np.random.standard_cauchy(n_pop) * 0.1 + F_mu
            Fs = np.clip(Fs, F[0], F[2])
            if type(CR) == float: CRs = np.ones(n_pop) * CR
        if "CR" in adapt_params:
            CR_mu = CR[1]
            if type(F) == float: Fs = np.ones(n_pop) * F
            CRs = np.random.normal(CR_mu, 0.1, n_pop)
            CRs = np.clip(CRs, CR[0], CR[2])

    #################
    # START PROGRAM #
    #################

    X = np.random.uniform(xl, xu, (n_pop, n_var))                       # Initialize population
    Y = np.array([evaluate_fitness(x) for x in X])                      # Evaluate fitness
    c_eval, c_gen = n_pop, 0

    if adapt_strategy in ["none", "random"]:
        indicator = "none"
    else:
        I = np.zeros(n_pop)

    #############
    # MAIN LOOP #
    #############

    while True:                                                         # Enter generation loop

        c_gen += 1

        for i in np.random.permutation(n_pop):                          # Traverse population

            xi, yi = X[i,:], Y[i]                                       # Get current individual
            Fi, CRi = Fs[i], CRs[i]                                     # Get current parameters

            maskbit = np.random.choice([0, 1], n_var, p=[1 - CRi, CRi]) # Get maskbit for crossover

            if diff_mode == "de":                                       # Reproduce by DE
                xr1 = X[np.random.choice(n_pop),:]
                xr2 = X[np.random.choice(n_pop),:]
                xi_ = xi + Fi * (xr1 - xr2) * maskbit
            if diff_mode == "de/curr-to-pbest":                         # Reproduce by DE/current to pbest
                pbest = sorted(np.arange(n_pop), key=lambda k: Y[k])[:int(p * n_pop)]
                xpbest = X[np.random.choice(pbest),:]
                xi_ = xi + Fi * (xpbest - xi) * maskbit
            if diff_mode == "de/curr-to-pbest/1":                       # Reproduce by DE/current to pbest/1
                pbest = sorted(np.arange(n_pop), key=lambda k: Y[k])[:int(p * n_pop)]
                xpbest = X[np.random.choice(pbest),:]
                xr1 = X[np.random.choice(n_pop),:]
                xr2 = X[np.random.choice(n_pop),:]
                xi_ = xi + Fi * (xpbest - xi + xr1 - xr2) * maskbit
            xi_ = np.clip(xi_, xl, xu)                                  # Repair to satisfy constraint

            yi_ = evaluate_fitness(xi_)                                 # Evaluate offspring
            c_eval += 1

            if yi_ < yi:                                                # Select better individual
                X[i], Y[i] = xi_, yi_

            if c_eval >= n_eval or min(Y) - problem.optimalF <= epsilon:# Termination criteria
                if is_print == True:
                    print("{}, {:.2e}, {:.1f}, {:.1f}".                 # Print results on screen
                          format(c_eval, min(Y), np.mean(Fs), np.mean(CRs)))
                if file != "none":                                      # Write results to file
                    history = open(file, "a")
                    history.write(f"{c_eval},{min(Y)},{np.mean(Fs)},{np.mean(CRs)}\n")
                    history.close()
                return X, Y, c_eval

            if indicator == "dfii/fi":                                  # Cumulate indicator
                I[i] += max([yi - yi_, 0]) / (yi - min(Y) + 1e-14)
            if indicator == "dfii":
                I[i] += max([yi - yi_, 0])

        if is_print == True:
            print("{}, {:.2e}, {:.1f}, {:.1f}".                         # Print results on screen
                  format(c_eval, min(Y), np.mean(Fs), np.mean(CRs)))
        if file != "none":                                              # Write results to file
            history = open(file, "a")
            history.write(f"{c_eval},{min(Y)},{np.mean(Fs)},{np.mean(CRs)}\n")
            history.close()

        ############
        # ADAPTION #
        ############

        if c_gen % n_step == 0:

            if adapt_strategy == "none":                                # No adaption
                Fs, CRs = Fs, CRs

            if adapt_strategy == "random":                              # Random adaption
                Fs, CRs = Fs, CRs
                if "F" in adapt_params:
                    Fs = Adaption.rand_adapt(Fs, F[0], F[1])
                if "CR" in adapt_params:
                    CRs = Adaption.rand_adapt(CRs, CR[0], CR[1])

            """
            !TODO - design a better variation for best_levy
            """
            if adapt_strategy == "best_levy":                           # Best levy adaption
                Fs, CRs = Fs, CRs
                if "CR" in adapt_params:
                    CRs = Adaption.best_levy_adapt(CRs, I[:], CR[0], CR[1], params_of_adapt_strategy)
                if "F" in adapt_params:
                    Fs = Adaption.best_levy_adapt(Fs, I[:], F[0], F[1], params_of_adapt_strategy)

            if adapt_strategy == "ga" and np.sum(I) != 0:               # GA adaption, no operation if all indicator is 0
                Fs, CRs = Fs, CRs
                if "CR" in adapt_params:
                    CRs = Adaption.ga_adapt(CRs, I[:], CR[0], CR[1], n_step, params_of_adapt_strategy)
                if "F" in adapt_params:
                    Fs = Adaption.ga_adapt(Fs, I[:], F[0], F[1], n_step, params_of_adapt_strategy)

            if adapt_strategy == "cauchy":                              # Cauchy adaption
                Fs, CRs = Fs, CRs
                if "CR" in adapt_params:
                    CRs, CR_mu = Adaption.cauchy_adapt(CRs, I[:], CR[0], CR[2], CR_mu, n_step, params_of_adapt_strategy)
                if "F" in adapt_params:
                    Fs, F_mu = Adaption.cauchy_adapt(Fs, I[:], F[0], F[2], F_mu, n_step, params_of_adapt_strategy)

            if adapt_strategy == "jade":                                # JADE adaption
                Fs, CRs = Fs, CRs
                if "F" in adapt_params:
                    params_of_adapt_strategy["avg_mode_adapt"] = "lehmer"
                    Fs, F_mu = Adaption.cauchy_adapt(Fs, I[:], F[0], F[2], F_mu, n_step, params_of_adapt_strategy)
                if "CR" in adapt_params:
                    params_of_adapt_strategy["avg_mode_adapt"] = "arithmetic"
                    CRs, CR_mu = Adaption.gauss_adapt(CRs, I[:], CR[0], CR[2], CR_mu, n_step, params_of_adapt_strategy)

            if indicator != "none": I = I * 0                           # Clear indicator

    return X, Y, c_eval

"""
##########
# SAMPLE #
##########

from Factory import set_problem

problem = set_problem("rastrigin")

X, F, c_eval = \
evolve(problem, n_var=30, n_eval=300000, n_pop=100,
       diff_mode="de/curr-to-pbest/1",
       F=(0.0, 0.5, 1.0), CR=(0.0, 0.5, 1.0), p=0.05,
       adapt_params=["F", "CR"], adapt_strategy="jade", indicator="dfii/fi", n_step=1,
       epsilon=1e-14, seed=1003, is_print=True, file="history.csv")
"""