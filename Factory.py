import SingleObjectiveProblem as SOP

def set_problem(problem_name):
    if problem_name == "ackley":
        return SOP.ackley()
    elif problem_name == "different_power":
        return SOP.different_power()
    elif problem_name == "dixon_price":
        return SOP.dixon_price()
    elif problem_name == "griewank":
        return SOP.griewank()
    elif problem_name == "k_tablet":
        return SOP.k_tablet()
    elif problem_name == "levy":
        return SOP.levy()
    elif problem_name == "perm":
        return SOP.perm()
    elif problem_name == "rastrigin":
        return SOP.rastrigin()
    elif problem_name == "rosenbrock_chain":
        return SOP.rosenbrock_chain()
    elif problem_name == "rosenbrock_star":
        return SOP.rosenbrock_star()
    elif problem_name == "rotated_hyper_ellipsoid":
        return SOP.rotated_hyper_ellipsoid()
    elif problem_name == "schwefel":
        return SOP.schwefel()
    elif problem_name == "sphere":
        return SOP.sphere()
    elif problem_name == "styblinski":
        return SOP.styblinski()
    elif problem_name == "trid":
        return SOP.trid()
    elif problem_name == "weighted_sphere":
        return SOP.weighted_sphere()
    elif problem_name == "xin_she":
        return SOP.xin_she()
    elif problem_name == "zakharov":
        return SOP.zakharov()
    elif problem_name == "schaffer":
        return SOP.schaffer()
    else:
        print("No such function!")
        return None
    