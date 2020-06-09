import numpy as np
from LevyVector import levy
import AdaptiveStrategy as Adaption

def evolve(problem, n_eval=300000, n_pop=15,
           alpha=0.01, beta=1.5, pa=0.1,
           epsilon=1e-14, seed=1000, is_print=True, file="none"):


    if type(seed) == int: np.random.seed(seed)                          # Set seed for random number generator

    evaluate_fitness = problem.f                                        # Get problem information
    xl, xu = problem.boundaries
    n_var = problem.n_var

    #####################
    # PARAMETER SETTING #
    #####################

    beta_mu = beta[1]
    ALPHA = np.ones(n_pop) * alpha
    BETA = np.random.standard_cauchy(n_pop) * 0.1 + beta_mu
    BETA = np.clip(BETA, beta[0], beta[2])

    #################
    # START PROGRAM #
    #################

    X = np.random.uniform(xl, xu, (n_pop, n_var))                       # Initialize population
    F = np.array([evaluate_fitness(x) for x in X])                      # Evaluate fitness
    c_eval, c_gen = n_pop, 0

    #############
    # MAIN LOOP #
    #############

    while True:                                                         # Enter generation loop

        ###############
        # LEVY FLIGHT #
        ###############

        c_gen += 1

        S_beta = []

        for i in np.random.permutation(n_pop):                          # Traverse population

            xi = X[i,:]                                                 # Get current individual
            alphai, betai = ALPHA[i], BETA[i]                                             # Get current parameters

            xi_ = xi + levy(alphai, betai, n_var)
            xi_ = np.clip(xi_, xl, xu)                                  # Repair to satisfy constraint

            fi_ = evaluate_fitness(xi_)                                 # Evaluate offspring
            c_eval += 1

            j = np.random.choice(n_pop)                                 # Randamly select individual
            fj = F[j]
            if fi_ <= fj:                                               # Select better individual
                X[j], F[j] = xi_, fi_
                S_beta.append(betai)

            if c_eval >= n_eval or min(F) - problem.optimalF <= epsilon:# Termination criteria
                if is_print == True:
                    print("{}, {:.2e}, {:.1e}, {:.1f}".                 # Print results on screen
                          format(c_eval, min(F), alpha, np.mean(BETA)))
                if file != "none":                                      # Write results to file
                    history = open(file, "a")
                    history.write(f"{c_eval},{min(F)},{alpha},{np.mean(BETA)}\n")
                    history.close()
                return X, F, c_eval


        ###########
        # REPLACE #
        ###########

        worst = sorted(np.arange(n_pop), key=lambda k: - F[k])          # Get worst individuals
        worst = worst[:int(pa * n_pop)]

        for k in worst:
            rand = np.random.uniform(xl, xu, n_var)
            xk_ = X[np.argmin(F)] + 0.5 * (rand - X[np.argmin(F)])
            xk_ = np.clip(xk_, xl, xu)
            X[k], F[k] = xk_, evaluate_fitness(xk_)

            c_eval += 1

            if c_eval >= n_eval or min(F) - problem.optimalF <= epsilon:# Termination criteria
                if is_print == True:
                    print("{}, {:.2e}, {:.1e}, {:.1f}".                 # Print results on screen
                          format(c_eval, min(F), alpha, np.mean(BETA)))
                if file != "none":                                      # Write results to file
                    history = open(file, "a")
                    history.write(f"{c_eval},{min(F)},{alpha},{np.mean(BETA)}\n")
                    history.close()
                return X, F, c_eval

        if is_print == True:
            print("{}, {:.2e}, {:.1e}, {:.1f}".                         # Print results on screen
                  format(c_eval, min(F), alpha, np.mean(BETA)))
        if file != "none":                                              # Write results to file
            history = open(file, "a")
            history.write(f"{c_eval},{min(F)},{alpha},{np.mean(BETA)}\n")
            history.close()

        ############
        # ADAPTION #
        ############

        mu_ = np.mean(S_beta)              # Compute average of successful parameters
        beta_mu = (1 - 0.1) * beta_mu + 0.1 * mu_
        BETA = np.random.standard_cauchy(n_pop) * 0.1 + beta_mu
        BETA = np.clip(BETA, beta[0], beta[2])

    return X, F, c_eval



from PyBenchFCN import Factory

sphere = Factory.set_sop("sphere", n_var=30)

X, F, c_eval = evolve(sphere, n_eval=30000, n_pop=20,
                      alpha=1e-07, beta=[0.1, 1.0, 1.9], pa=0.1,
                      epsilon=1e-10, seed=1000, is_print=True, file="none")