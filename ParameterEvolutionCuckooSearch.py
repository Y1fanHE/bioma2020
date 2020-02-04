import numpy as np
import copy

from Population import initialize, evaluate
from LevyVector import default, mantegna, gutowski, fix_bound

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             alpha_1, levy_alg, pa_1,
             betal, betau, step_gen, indicator,
             alpha_2, beta_2,
             seed, record):

    # set numpy seed
    np.random.seed()

    # set record files
    if record != None:
        record_file = open(record, "a")

    # initialize parameters
    f = problem.f
    xl, xu = problem.boundaries
    if levy_alg == "default":
        levy = default
    elif levy_alg == "mantegna":
        levy = mantegna
    elif levy_alg == "gutowski":
        levy = gutowski
    else:
        levy = default
    betas = np.random.uniform(betal, betau, n_pop)
    betas_old = betas[:]
    
    # initialize evaluation counter
    n_eval = 0

    # initialize convergence counter
    n_conv = 0
    h_best = np.inf

    # initialize population
    X = np.random.uniform(xl, xu, (n_pop, n_var))

    # evaluate fitness
    F = evaluate(X, f)
    n_eval += n_pop
    # print(0, min(F))

    # initialize indicator at 0
    I = np.zeros(n_pop)
    I_old = np.zeros(n_pop)
    I_trial = np.zeros(n_pop)

    # enter generation loop
    for c_gen in range(1, max_gen):

        # record fitness and parameters by generations
        if record != None:
            record_file.write("{},{:.3e},{:.3e},".format(c_gen, min(F), np.mean(F)))
            for item in betas:
                record_file.write("{:.1f},".format(item))
            record_file.write("\n")

        # traverse population
        for i in range(n_pop):

            # reproduce by Levy Flight
            xi = copy.copy(X[i])
            xi_ = xi + levy(alpha_1, betas[i], n_var)
            xi_ = fix_bound(xi_, xl, xu)

            # evaluate offspring
            fi_ = f(xi_)

            # pick up a random individual
            j = np.random.choice(n_pop)
            fj = copy.copy(F[j])

            fi = copy.copy(F[i])

            # compare offspring with picked individual
            if fi_ <= fj:
                X[j] = xi_
                F[j] = fi_

            # calculate indicator
            if indicator == "df":
                tmp = np.max([fi - fi_, 0])
            elif indicator == "df/f":
                tmp = np.max([fi - fi_, 0]) / (fi + 1e-14)
            else:
                tmp = np.max([fi - fi_, 0]) / (fi + 1e-14)
            I[i] += tmp

            n_eval += 1
            if n_eval >= max_eval or min(F) <= 0.0: return X, F

        # generate trial parameters
        if c_gen % step_gen == int (step_gen / 2):

            # save original parameters old parameters
            betas_old = betas[:]

            # update parameters to trial parameters
            betas = betas + levy(alpha_2, beta_2, n_pop)
            betas = fix_bound(betas, betal, betau)

            # clear indicator
            I_old = I /  (step_gen / 2)
            I = I * 0

        # evaluate trial parameters
        if c_gen % step_gen == 0:
            
            # compute indicator for trial parameters
            I_trial = I / (step_gen / 2)

            # change if trial parameters are worse
            betas[I_trial <= I_old] = betas_old[I_trial <= I_old]

            # clear indicator
            I = I * 0

        # randomly abandon worst individuals
        idx_sorted = sorted(np.arange(n_pop), key=lambda k: - F[k])
        for i in range(n_pop):
            if i in idx_sorted[:int (n_pop * pa_1)]:

                # do a levy flight on worst individuals
                xi_ = X[i] + levy(alpha_1, betas[i], n_var)
                xi_ = fix_bound(xi_, xl, xu)
                X[i] = xi_
                F[i] = f(X[i])
                
                n_eval += 1
                if n_eval >= max_eval or min(F) <= 0.0: return X, F

        # set convergence criteria
        if np.abs(h_best - min(F)) >= 1e-14:
            h_best = min(F)
            n_conv = 0
        else:
            n_conv += 1
            if n_conv >= 10000:
                return X, F

    if record != None:
        record_file.close()

    return X, F