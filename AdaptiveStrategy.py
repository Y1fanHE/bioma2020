import numpy as np
from LevyVector import levy

def no_adapt(PARAMETER):
    return PARAMETER

def rand_adapt(PARAMETER, pl, pu):
    return np.random.uniform( pl, pu, len(PARAMETER) )

def best_levy_adapt(PARAMETER, INDICATOR, pl, pu, params_of_adapt_strategy):
    F = params_of_adapt_strategy["F_adapt"]
    alpha = params_of_adapt_strategy["alpha_adapt"]
    beta = params_of_adapt_strategy["beta_adapt"]

    best = np.argmax(INDICATOR)
    PARAMETER = PARAMETER +  F * (PARAMETER[best] - PARAMETER)
    PARAMETER[best] += levy(alpha, beta, len(PARAMETER))
    PARAMETER = np.clip(PARAMETER, pl, pu)
    return PARAMETER

def ga_adapt(PARAMETER, INDICATOR, pl, pu, n_step, params_of_adapt_strategy):
    pc = params_of_adapt_strategy["pc_adapt"]
    pm = params_of_adapt_strategy["pm_adapt"]
    crx_mode = params_of_adapt_strategy["crx_mode_adapt"]
    mut_mode = params_of_adapt_strategy["mut_mode_adapt"]

    n_pop = len(PARAMETER)
    INDICATOR = INDICATOR / n_step + 1e-14
    prob = INDICATOR / np.sum(INDICATOR)

    p1 = PARAMETER[np.random.choice(n_pop, n_pop, p=prob)]  # Selection

    if crx_mode == "uniform_sum":
        sigma = np.random.uniform(0, 1, n_pop)
        PARAMETER_ = p1 * sigma + PARAMETER * (1 - sigma)   # Crossover
        rand = np.random.uniform(0, 1, n_pop)
        PARAMETER[rand <= pc] = PARAMETER_[rand <= pc]

    if mut_mode == "levy":
        alpha = params_of_adapt_strategy["alpha_adapt"]
        beta = params_of_adapt_strategy["beta_adapt"]
        PARAMETER = PARAMETER + levy(alpha, beta, n_pop) * \
            np.random.choice([0,1], n_pop, p=[1-pm, pm])    # Mutation
    if mut_mode == "gauss":
        sigma = params_of_adapt_strategy["sigma_adapt"]
        PARAMETER = PARAMETER + np.random.normal(0, 1, n_pop) * sigma * \
            np.random.choice([0,1], n_pop, p=[1-pm, pm])

    PARAMETER = np.clip(PARAMETER, pl, pu)                  # Repair to satisfy boundary constraint
    return PARAMETER

def cauchy_adapt(PARAMETER, INDICATOR, pl, pu, mu, n_step, params_of_adapt_strategy):
    c = params_of_adapt_strategy["c_adapt"]                 # learning rate
    gamma = params_of_adapt_strategy["gamma_adapt"]         # scaling factor
    avg_mode = params_of_adapt_strategy["avg_mode_adapt"]   # method to compute average

    n_pop = len(PARAMETER)
    INDICATOR = INDICATOR / n_step
    good_idx = np.where(INDICATOR > 0)                     # Collect successful parameters
    if np.sum(INDICATOR) > 0:
        if avg_mode == "arithmetic":
            mu_ = np.mean(PARAMETER[good_idx])              # Compute average of successful parameters
        if avg_mode == "lehmer":
            GOOD = PARAMETER[good_idx]
            mu_ = np.sum(GOOD ** 2) / np.sum(GOOD)
        if avg_mode == "weighted_arithmetic":
            mu_ = np.mean(INDICATOR[good_idx] / np.sum(INDICATOR[good_idx]) * PARAMETER[good_idx])
        if avg_mode == "weighted_lehmer":
            GOOD = PARAMETER[good_idx]
            weight = INDICATOR[good_idx] / np.sum(INDICATOR[good_idx])
            mu_ = np.sum(weight * GOOD ** 2) / np.sum(weight * GOOD)
        mu = (1 - c) * mu + c * mu_
    PARAMETER = np.random.standard_cauchy(n_pop) * gamma + mu
    PARAMETER = np.clip(PARAMETER, pl, pu)
    return PARAMETER, mu

def gauss_adapt(PARAMETER, INDICATOR, pl, pu, mu, n_step, params_of_adapt_strategy):
    c = params_of_adapt_strategy["c_adapt"]                 # learning rate
    sigma = params_of_adapt_strategy["sigma_adapt"]         # scaling factor
    avg_mode = params_of_adapt_strategy["avg_mode_adapt"]   # method to compute average

    n_pop = len(PARAMETER)
    INDICATOR = INDICATOR / n_step
    good_idx = np.where(INDICATOR != 0)                     # Collect successful parameters
    if np.sum(INDICATOR) > 0:
        if avg_mode == "arithmetic":
            mu_ = np.mean(PARAMETER[good_idx])              # Compute average of successful parameters
        if avg_mode == "lehmer":
            GOOD = PARAMETER[good_idx]
            mu_ = np.sum(GOOD ** 2) / np.sum(GOOD)
        if avg_mode == "weighted_arithmetic":
            mu_ = np.mean(INDICATOR[good_idx] / np.sum(INDICATOR[good_idx]) * PARAMETER[good_idx])
        if avg_mode == "weighted_lehmer":
            GOOD = PARAMETER[good_idx]
            weight = INDICATOR[good_idx] / np.sum(INDICATOR[good_idx])
            mu_ = np.sum(weight * GOOD ** 2) / np.sum(weight * GOOD)
        mu = (1 - c) * mu + c * mu_
    PARAMETER = np.random.normal(0, 1, n_pop) * sigma + mu
    PARAMETER = np.clip(PARAMETER, pl, pu)
    return PARAMETER, mu