import numpy as np
import copy

from Population import initialize, evaluate
from LevyVector import default, mantegna, gutowski, fix_bound

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             alpha, beta, levy_alg, pa,
             epsilon, seed):

    # set numpy seed
    np.random.seed(seed)

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
    for _ in range(1, max_gen):

        # traverse population
        for i in range(n_pop):

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
            
            n_eval += 1
            if n_eval >= max_eval or min(F) <= epsilon: return X, F, n_eval

        # randomly abandon worst individuals
        idx_sorted = sorted(np.arange(n_pop), key=lambda k: - F[k])
        for i in range(n_pop):
            if i in idx_sorted[:int (n_pop * pa)]:

                # do a levy flight on worst individuals
                # xi_ = X[i] + levy(alpha, beta, n_var)
                # xi_ = fix_bound(xi_, xl, xu)

                # reinitialize worst individuals
                xi_ = initialize(1, n_var, xl, xu).flatten()

                X[i] = xi_
                F[i] = f(X[i])
                
                n_eval += 1
                if n_eval >= max_eval or min(F) <= epsilon: return X, F, n_eval

        # set convergence criteria
        if np.abs(h_best - min(F)) >= epsilon * 0.01:
            h_best = min(F)
            n_conv = 0
        else:
            n_conv += 1
            if n_conv >= 10000:
                return X, F, n_eval

    return X, F, n_eval