import numpy as np
import copy

from Population import initialize, evaluate
from LevyVector import mantegna, gutowski, fix_bound

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             alpha, beta, levy_alg, pa,
             seed):
    
    # set numpy seed
    np.random.seed(seed)

    # initialize parameters
    f = problem.f
    xl, xu = problem.boundaries
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
    for c_gen in range(1, max_gen):

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
            if n_eval >= max_eval: return X, F

        # randomly abandon worst individuals
        sorted_idx = sorted(np.arange(n_pop), key=lambda k: F[k])
        worst_idx = sorted_idx[- int (n_pop * pa):]
        for i in worst_idx:
            X[i] = np.random.uniform(xl, xu, n_var)
            F[i] = f(X[i])
            
            n_eval += 1
            if n_eval >= max_eval: return X, F
        
        # print(c_gen, min(F))
    
    return X, F