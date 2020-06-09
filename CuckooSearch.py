import numpy as np
from LevyVector import levy
import AdaptiveStrategy as Adaption

def evolve(problem, n_eval=300000, n_pop=15,
           levy_mode="lf", replace_mode="lf",
           alpha=0.01, beta=1.5, pa=0.1,
           adapt_params=[], adapt_strategy="none", params_of_adapt_strategy={},
           indicator="none", n_step=20,
           epsilon=1e-14, seed=1000, is_print=True, file="none"):

    '''
    Cuckoo Search (CS) with Self-adaptive strategy

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
    levy_mode: string
      - mode of levy flight mutation
      - valid input: { "lf", "lf/curr-to-best", "lf/curr-to-best/1" }
    replace_mode: string
      - mode of replace operation
      - valid input: { "lf", "random" }

    Variation Parameters
    ----------
    alpha, beta, pa: float OR tuple
      - scaling factor, stability factor, replace rate
      - valid input:
        + for strategy type 1, input (lower, upper)
        + for strategy type 2, input (lower, mean, upper)
        + for non-adaptive, input fixed float
      * change { adapt_params, adapt_strategy } correspondingly

    Strategy Parameters
    ----------
    adapt_params: list of string
      - parameters which are set to be adaptive
      - valid input: { ["alpha"], ["beta"], ["alpha, beta"] } !TODO - alpha-adaption is incomplete
      * change { alpha, beta, adapt_strategy } correspondingly
    adapt_strategy: string
      - parameter variation method in adaptive strategy
      - valid input: { "none", "random", "best_levy", "ga", "cauchy" }
      * change { alpha, beta, adapt_params } correspondingly
    indicator: string
      - indicator used to assess quality of parameters
        automatically set to "none" if adapt_strategy == "none" or "random"
      - valid input: { "dfii/fi", "dfij/fj" } !TODO - increase more indicators
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

    if adapt_strategy == "none":
        ALPHA = np.ones(n_pop) * alpha
        BETA = np.ones(n_pop) * beta

    if adapt_strategy in strategy_type1:                                # Initialize parameters with uniform distribution
        if "alpha" in adapt_params:
            ALPHA = np.random.uniform(alpha[0], alpha[1], n_pop)
            if type(beta) == float: BETA = np.ones(n_pop) * beta
        if "beta" in adapt_params:
            if type(alpha) == float: ALPHA = np.ones(n_pop) * alpha
            BETA = np.random.uniform(beta[0], beta[1], n_pop)

    if adapt_strategy in strategy_type2:                                # Initialize parameters with Cauchy distribution
        if "alpha" in adapt_params:
            alpha_mu = alpha[1]
            ALPHA = np.random.standard_cauchy(n_pop) * 0.1 + alpha_mu
            ALPHA = np.clip(ALPHA, alpha[0], alpha[2])
            if type(beta) == float: BETA = np.ones(n_pop) * beta
        if "beta" in adapt_params:
            beta_mu = beta[1]
            if type(alpha) == float: ALPHA = np.ones(n_pop) * alpha
            BETA = np.random.standard_cauchy(n_pop) * 0.1 + beta_mu
            BETA = np.clip(BETA, beta[0], beta[2])

    #################
    # START PROGRAM #
    #################

    X = np.random.uniform(xl, xu, (n_pop, n_var))                       # Initialize population
    F = np.array([evaluate_fitness(x) for x in X])                      # Evaluate fitness
    c_eval, c_gen = n_pop, 0

    if adapt_strategy in ["none", "random"]:
        indicator = "none"
    else:
        I = np.zeros(n_pop)

    #############
    # MAIN LOOP #
    #############

    while True:                                                         # Enter generation loop

        ###############
        # LEVY FLIGHT #
        ###############

        c_gen += 1

        for i in np.random.permutation(n_pop):                          # Traverse population

            xi, fi = X[i,:], F[i]                                       # Get current individual
            alphai, betai = ALPHA[i], BETA[i]                           # Get current parameters

            if levy_mode == "lf":                                       # Reproduce by Levy Flight
                xi_ = xi + levy(alphai, betai, n_var)
            if levy_mode == "lf/curr-to-best":                          # Reproduce by LF/current to best
                xbest = X[np.argmin(F),:]
                xi_ = xi + levy(alphai, betai, n_var) * (xbest - xi)
            if levy_mode == "lf/curr-to-best/1":                        # Reproduce by LF/current to best/1
                xbest = X[np.argmin(F),:]
                xr1 = X[np.random.choice(n_pop),:]
                xr2 = X[np.random.choice(n_pop),:]
                xi_ = xi + levy(alphai, betai, n_var) * (xbest - xi + xr1 - xr2)
            xi_ = np.clip(xi_, xl, xu)                                  # Repair to satisfy constraint

            fi_ = evaluate_fitness(xi_)                                 # Evaluate offspring
            c_eval += 1

            j = np.random.choice(n_pop)                                 # Randamly select individual
            fj = F[j]
            if fi_ <= fj:                                               # Select better individual
                X[j], F[j] = xi_, fi_

            if c_eval >= n_eval or min(F) - problem.optimalF <= epsilon:# Termination criteria
                if is_print == True:
                    print("{}, {:.2e}, {:.1e}, {:.1f}".                 # Print results on screen
                          format(c_eval, min(F), np.mean(ALPHA), np.mean(BETA)))
                if file != "none":                                      # Write results to file
                    history = open(file, "a")
                    history.write(f"{c_eval},{min(F)},{np.mean(ALPHA)},{np.mean(BETA)}\n")
                    history.close()
                return X, F, c_eval

            if indicator == "dfii/fi":                                  # Cumulate indicator
                I[i] += max([fi - fi_, 0]) / (fi - min(F) + 1e-14)
            if indicator == "dfij/fj":
                I[i] += max([fj - fi_, 0]) / (fj - min(F) + 1e-14)
            if indicator == "dfii":
                I[i] += max([fi - fi_, 0])
            if indicator == "dfij":
                I[i] += max([fj - fi_, 0])

        ###########
        # REPLACE #
        ###########

        worst = sorted(np.arange(n_pop), key=lambda k: - F[k])          # Get worst individuals
        worst = worst[:int(pa * n_pop)]

        for k in worst:

            if replace_mode == "lf":                                    # Replace by Levy Flight
                xk_ = X[np.argmin(F)] + levy(0.1, 1.0, n_var)
                xk_ = np.clip(xk_, xl, xu)
            if replace_mode == "random":                                # Reinitialize individual
                xk_ = np.random.uniform(xl, xu, n_var)
            if replace_mode == "best-to-rand":
                rand = np.random.uniform(xl, xu, n_var)
                xk_ = X[np.argmin(F)] + 0.5 * (rand - X[np.argmin(F)])
                xk_ = np.clip(xk_, xl, xu)
            X[k], F[k] = xk_, evaluate_fitness(xk_)

            c_eval += 1
            if c_eval >= n_eval or min(F) - problem.optimalF <= epsilon:# Termination criteria
                if is_print == True:
                    print("{}, {:.2e}, {:.1e}, {:.1f}".                 # Print results on screen
                          format(c_eval, min(F), np.mean(ALPHA), np.mean(BETA)))
                if file != "none":                                      # Write results to file
                    history = open(file, "a")
                    history.write(f"{c_eval},{min(F)},{np.mean(ALPHA)},{np.mean(BETA)}\n")
                    history.close()
                return X, F, c_eval

        if is_print == True:
            print("{}, {:.2e}, {:.1e}, {:.1f}".                         # Print results on screen
                  format(c_eval, min(F), np.mean(ALPHA), np.mean(BETA)))
        if file != "none":                                              # Write results to file
            history = open(file, "a")
            history.write(f"{c_eval},{min(F)},{np.mean(ALPHA)},{np.mean(BETA)}\n")
            history.close()

        ############
        # ADAPTION #
        ############

        if c_gen % n_step == 0:

            if adapt_strategy == "none":                                # No adaption
                ALPHA, BETA = ALPHA, BETA

            if adapt_strategy == "random":                              # Random adaption
                ALPHA, BETA = ALPHA, BETA
                if "alpha" in adapt_params:
                    ALPHA = Adaption.rand_adapt(ALPHA, alpha[0], alpha[1])
                if "beta" in adapt_params:
                    BETA = Adaption.rand_adapt(ALPHA, beta[0], beta[1])

            """
            !TODO - design a better variation for best_levy
            """
            if adapt_strategy == "best_levy":                           # Best levy adaption
                ALPHA, BETA = ALPHA, BETA
                if "beta" in adapt_params:
                    BETA = Adaption.best_levy_adapt(BETA, I[:], beta[0], beta[1], params_of_adapt_strategy)

            if adapt_strategy == "ga":                                  # GA adaption
                ALPHA, BETA = ALPHA, BETA
                if "beta" in adapt_params:
                    BETA = Adaption.ga_adapt(BETA, I[:], beta[0], beta[1], n_step, params_of_adapt_strategy)

            if adapt_strategy == "cauchy":                              # Cauchy adaption (!TODO - equivalent to JADE???)
                ALPHA, BETA = ALPHA, BETA
                if "beta" in adapt_params:
                    BETA, beta_mu = Adaption.cauchy_adapt(BETA, I[:], beta[0], beta[2], beta_mu, n_step, params_of_adapt_strategy)

            if indicator != "none": I = I * 0                           # Clear indicator

    return X, F, c_eval

"""
##########
# SAMPLE #
##########

from Factory import set_problem

problem = set_problem("rastrigin")

X, F, c_eval = \
evolve(problem, n_var=30, n_eval=300000, n_pop=20,
       levy_mode="lf", replace_mode="levy",
       alpha=1e-09, beta=(0.1, 1.9), pa=0.25,
       adapt_params=["beta"], adapt_strategy="ga", indicator="dfii/fi", n_step=10,
       epsilon=1e-14, seed=1003, is_print=True, file="history.csv")
"""