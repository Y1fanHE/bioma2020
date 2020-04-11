import numpy as np
import copy
from Population import initialize, evaluate
from LevyVector import default, fix_bound

def optimize(problem, n_var, n_pop, max_gen, max_eval,
             p, Fl, Fu, CRl, CRu,
             n_step, alpha, beta,
             epsilon, seed, record):

    # set numpy seed
    np.random.seed(seed)

    # set record files
    if record != None:
        record_file = open(record, "a")

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
    # generate parameters by distributions
    P = np.random.uniform([Fl, CRl], [Fu, CRu], (n_pop, 2))
    P_parent = np.full((n_pop, 2), np.nan)
    P_offspring = np.full((n_pop, 2), np.nan)
    I = np.zeros(n_pop)
    I_parent = np.zeros(n_pop)
    I_offspring = np.zeros(n_pop)

    # evaluate fitness
    Fit = evaluate(X, fit)
    n_eval += n_pop

    # start generation loop
    for c_gen in range(1, max_gen):

        # record fitness and parameters by generations
        if record != None:
            record_file.write("{:.3e},{:.3e},".format(min(Fit), np.mean(Fit)))
            record_file.write("{:.3f},".format(np.mean(P[:, 0])))
            record_file.write("{:.3f}".format(np.mean(P[:, 1])))
            record_file.write("\n")

        for i in range(n_pop):

            # get parameters for the current individual
            Fi, CRi = P[i]

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
            vi = xi + Fi * (xbest - xi + xr1 - xr2)

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
                I[i] += np.abs(fiti_ - fiti) / (fiti + 1e-14)

            if n_eval >= max_eval or min(Fit) <= epsilon: return X, Fit, n_eval

        # set convergence criteria
        if np.abs(h_best - min(Fit)) >= epsilon * 0.01:
            h_best = min(Fit)
            n_conv = 0
        else:
            n_conv += 1
            if n_conv >= 10000:
                return X, Fit, n_eval

        # parameter adaption
        if c_gen % (2 * n_step) == n_step:
            P_parent = P
            I_parent = I / n_step
            P = fix_bound(P_parent + default(alpha, beta, (n_pop, 2)), [Fl, CRl], [Fu, CRu])
            I = I * 0

        if c_gen % (2 * n_step) == 0:
            P_offspring = P
            I_offspring = I / n_step
            P[I_offspring <= I_parent] = P_parent[I_offspring <= I_parent]
            P[I_offspring > I_parent] = P_offspring[I_offspring > I_parent]
            I = I * 0

    if record != None:
        record_file.close()

    return X, Fit, n_eval