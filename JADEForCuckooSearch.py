import numpy as np
import copy

from Population import initialize, evaluate
from LevyVector import default, mantegna, gutowski, fix_bound

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             alpha, levy_alg, pa,
             beta_mu, beta_l, beta_u, c,
             seed, record):

    # set numpy seed
    np.random.seed(seed)

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

    # enter generation loop
    for c_gen in range(1, max_gen):

        # initialize success parameter pool
        S_beta = []

        # record fitness and parameters by generations
        if record != None:
            record_file.write("{},{:.3e},{:.3e},{:.1f}"
                              .format(c_gen, min(F), np.mean(F), beta_mu))
            record_file.write("\n")

        # traverse population
        for i in range(n_pop):
            
            # select a parameter from cauchy distribution
            # beta = np.random.normal(beta_mu, 0.1)
            beta = 0.1 * np.tan(np.random.uniform(-np.pi/2, np.pi/2)) + beta_mu
            beta = fix_bound(beta, beta_l, beta_u)

            # reproduce by Levy Flight
            xi = copy.copy(X[i])
            xi_ = xi + levy(alpha, beta, n_var)
            xi_ = fix_bound(xi_, xl, xu)

            # evaluate offspring
            fi_ = f(xi_)

            # pick up a random individual
            j = np.random.choice(n_pop)
            fj = copy.copy(F[j])

            # compare offspring with picked individual
            if fi_ <= fj:
                X[j] = xi_
                F[j] = fi_
                
                # add successful parameter into pool
                S_beta.append(beta)

            n_eval += 1
            if n_eval >= max_eval or min(F) <= 0.0: return X, F

        # update mean of parameters
        if len(S_beta) > 0:
            S_mu = np.sum(np.array(S_beta)**2) / np.sum(np.array(S_beta))
            beta_mu = (1 - c) * beta_mu + c * S_mu

        # randomly abandon worst individuals
        idx_sorted = sorted(np.arange(n_pop), key=lambda k: - F[k])
        for i in range(n_pop):
            if i in idx_sorted[:int (n_pop * pa)]:

                # do a levy flight on worst individuals
                xi_ = X[i] + levy(alpha, beta, n_var)
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

    return X, F