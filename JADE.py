import numpy as np
import copy
from Population import initialize, evaluate

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             mu_CR, mu_F, p,
             c,
             epsilon, seed):

    # set numpy seed
    np.random.seed(seed)

    # initialize parameters
    fit = problem.f
    xl, xu = problem.boundaries

    # initialize evaluation counter
    n_eval = 0

    # initialize convergence counter
    n_conv = 0
    h_best = np.inf

    # initialize population
    X = np.random.uniform(xl, xu, (n_pop, n_var))

    # evaluate fitness
    Fit = evaluate(X, fit)
    n_eval += n_pop

    for _ in range(1, max_gen):

        # initialize archive to store successful parameters
        SF, SCR = [], []

        # generate parameters by distributions
        CR = np.clip( np.random.normal(mu_CR, 0.1, n_pop), 0, 1)
        F = np.clip( np.random.standard_cauchy(n_pop) * 0.1 + mu_F, 0, 1 )

        # start generation loop
        for i in range(n_pop):

            # get parameters for the current individual
            Fi, CRi = F[i], CR[i]

            # get index for the individuals required in reproduction
            sorted_idx = sorted(np.arange(n_pop), key=lambda k: Fit[k])
            best = np.random.choice(sorted_idx[:int(n_pop * p)])
            r1 = np.random.choice(n_pop)
            while r1 == i:
                r1 = np.random.choice(n_pop)
            r2 = np.random.choice(n_pop)
            while r1 == i or r2 == i:
                r2 = np.random.choice(n_pop)

            # get individuals required in reproduction
            xi = copy.copy(X[i])
            xbest = copy.copy(X[best])
            xr1 = copy.copy(X[r1])
            xr2 = copy.copy(X[r2])

            # de/curr-to-pbest/1 mutatoon
            vi = np.clip( xi + Fi * (xbest - xi + xr1 - xr2), xl, xu )

            rand = np.random.uniform(0, 1, n_var)
            jrand = np.random.choice(n_var)

            # de crossover
            xi_ = np.full(n_var, np.nan)
            xi_[rand < CRi] = vi[rand < CRi]
            xi_[rand >= CRi] = xi[rand >= CRi]
            xi_[jrand] = vi[jrand]

            # de selection
            fiti = Fit[i]
            fiti_ = fit(xi_)
            n_eval += 1
            if fiti_ < fiti:
                X[i] = xi_
                Fit[i] = fiti_

                # add successful parameters to archive
                SCR.append(CRi)
                SF.append(Fi)

            if n_eval >= max_eval or min(Fit) <= epsilon: return X, Fit, n_eval

        # set convergence criteria
        if np.abs(h_best - min(Fit)) >= epsilon * 0.01:
            h_best = min(Fit)
            n_conv = 0
        else:
            n_conv += 1
            if n_conv >= 10000:
                return X, Fit, n_eval

        # update expectation values of distribution
        SCR, SF = np.array(SCR), np.array(SF)
        if len(SCR) > 0:
            mu_CR = (1 - c) * mu_CR + c * np.mean(SCR)
        if len(SF) > 0:
            mu_F = (1 - c) * mu_F + c * np.sum(SF ** 2) / np.sum(SF)

    return X, Fit, n_eval