from Factory import set_problem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_3D(problem_name, N=50, xbound=None):
    problem = set_problem(problem_name)                             # Set benchmark problem
    if xbound == None:
        xl, xu = problem.boundaries
    else:
        xl, xu = xbound
    f = problem.f

    x = np.linspace(xl, xu, N)                                      # Generate pairs of decision value
    X1, X2 = np.meshgrid(x, x)
    X = np.c_[np.ravel(X1), np.ravel(X2)]

    F = []                                                          # Compute fitness value for each pair
    for x in X:
        F.append(f(x))
    F = np.array(F).reshape(X1.shape)

    fig = plt.figure()                                              # Plotting 3D graph
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X1, X2, F, cmap='seismic')

    plt.savefig(f"{problem_name}3D.svg")
    plt.close("all")

def plot_2D(problem_name, N=50, xbound=None):
    problem = set_problem(problem_name)                             # Set benchmark problem
    if xbound == None:
        xl, xu = problem.boundaries
    else:
        xl, xu = xbound
    f = problem.f

    x = np.linspace(xl, xu, N)                                      # Generate pairs of decision value
    X1, X2 = np.meshgrid(x, x)
    X = np.c_[np.ravel(X1), np.ravel(X2)]

    F = []                                                          # Compute fitness value for each pair
    for x in X:
        F.append(f(x))
    F = np.array(F).reshape(X1.shape)

    plt.figure()                                              # Plotting 3D graph
    plt.contour(X1, X2, F, levels=30, cmap='seismic')
    plt.colorbar()
    plt.savefig(f"{problem_name}2D.svg")
    plt.close("all")

problems = ["sphere", "rotated_hyper_ellipsoid", "different_power", "weighted_sphere",
            "dixon_price", "rosenbrock_chain", "rosenbrock_star", "k_tablet", "zakharov",
            "ackley", "rastrigin", "griewank", "levy", "schwefel", "xin_she", "schaffer"]

for problem in problems:
    plot_2D(problem, 1000)
