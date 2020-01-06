import numpy as np
import copy

from Population import initialize, evaluate
from LevyVector import mantegna, gutowski, fix_bound

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             alpha_1, levy_alg, pa,
             betal, betau, step_gen, indicator,
             alpha_2, beta_2,
             seed):
    # set numpy seed
    np.random.seed()

    # initialize parameters
    f = problem.f
    xl, xu = problem.boundaries
    if levy_alg == "mantegna": levy = mantegna
    if levy_alg == "gutowski": levy = gutowski
    betas = np.random.uniform(betal, betau, n_pop)
    n_eval = 0

    # initialize population
    X = np.random.uniform(xl, xu, (n_pop, n_var))

    # evaluate fitness
    F = evaluate(X, f)
    n_eval += n_pop
    # print(0, min(F))

    # initialize indicator at 0
    dF = np.zeros(n_pop)

    # enter generation loop
    for c_gen in range(1, max_gen):

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

            # compare offspring with picked individual
            if fi_ <= fj:
                X[j] = xi_
                F[j] = fi_

                # calculate indicator
                if indicator == "df":
                    df = np.abs(fi_ - fj)
                elif indicator == "df/f":
                    df = np.abs(fi_ - fj) / (fj + 1e-14)
                else:
                    df = np.abs(fi_ - fj) / (fj + 1e-14)
                
                dF[i] += df

            n_eval += 1
            if n_eval >= max_eval: return X, F

        # evolve parameters
        if c_gen % step_gen == step_gen - 1:

            # get best parameter
            beta_best = betas[np.argmax(dF)]
            dF = dF * 0
        
        # randomly abandon worst individuals
        sorted_idx = sorted(np.arange(n_pop), key=lambda k: F[k])
        worst_idx = sorted_idx[- int (n_pop * pa):]
        for i in worst_idx:
            X[i] = np.random.uniform(xl, xu, n_var)
            F[i] = f(X[i])
            
            n_eval += 1
            if n_eval >= max_eval: return X, F
        
        dF[worst_idx] = 0

        print(c_gen, "{:.3e}".format(min(F)), "{:.2f}".format(np.mean(dF)), "{:.2f}".format(np.median(betas)))
    
    return X, F