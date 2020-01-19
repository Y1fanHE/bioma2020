import numpy as np
import copy

from Population import initialize, evaluate
from LevyVector import default, mantegna, gutowski, fix_bound

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             alpha, betal, betau, levy_alg, pa,
             seed):
    
    # set numpy seed
    np.random.seed(seed)

    # initialize parameters
    f = problem.f
    xl, xu = problem.boundaries
    if levy_alg == "default": levy = default
    if levy_alg == "mantegna": levy = mantegna
    if levy_alg == "gutowski": levy = gutowski
    n_eval = 0

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
            beta = np.random.uniform(betal, betau)
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
            if n_eval >= max_eval or min(F) == 0.0: return X, F

        #randomly abandon worst individuals
        idx_sorted = sorted(np.arange(n_pop), key=lambda k: - F[k])
        for i in range(n_pop):
            if i in idx_sorted[:int (n_pop * pa)]:
                X[i] = np.random.uniform(xl, xu, n_var)
                F[i] = f(X[i])
                
                n_eval += 1
                if n_eval >= max_eval or min(F) == 0.0: return X, F
            
    return X, F