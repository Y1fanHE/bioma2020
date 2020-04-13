import numpy as np

# many local minima
class ackley():
    def __init__(self):
        self.boundaries = np.array([-32.768, 32.768])

    def f(self, x):
        t1 = 20
        t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
        t3 = np.e
        t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
        return t1 + t2 + t3 + t4

class griewank():
    def __init__(self):
        self.boundaries = np.array([-600, 600])

    def f(self, x):
        w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(x))])
        t1 = 1
        t2 = 1.0 / 4000.0 * np.sum(x ** 2)
        t3 = - np.prod(np.cos(x * w))
        return t1 + t2 + t3

class levy():
    def __init__(self):
        self.boundaries = np.array([-10, 10])

    def f(self, x):
        w = np.array(1. + (x - 1) / 4.)
        t1 = np.sin(np.pi * w[0]) ** 2
        t2 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        t3 = np.sum( (w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2) )
        return t1 + t2 + t3

class rastrigin():
    def __init__(self):
        self.boundaries = np.array([-5, 5])

    def f(self, x):
        t1 = 10 * len(x)
        t2 = np.sum(x ** 2)
        t3 = - 10 * np.sum(np.cos(2 * np.pi * x))
        return t1 + t2 + t3

class schwefel():
    def __init__(self):
        self.boundaries = np.array([-500, 500])

    def f(self, x):
        x = np.round(x, 13)
        return 418.9828872724339*len(x) - np.sum(x * np.sin( np.sqrt( np.abs(x) ) ) )

class xin_she():
    def __init__(self):
        self.boundaries = np.array([-2 * np.pi, 2 * np.pi])

    def f(self, x):
        t1 = np.sum( np.abs(x) )
        e1 = - np.sum( np.sin(x ** 2) )
        t2 = np.exp(e1)
        return t1 * t2

class schaffer():
    def __init__(self):
        self.boundaries = np.array([-100, 100])

    def f(self, x):
        val = 0
        for i in range(len(x)-1):
            t1 = (x[i]**2 + x[i+1]**2)**0.25
            t2 = np.sin(50*(x[i]**2 + x[i+1]**2)**0.1)**2+1
            val += t1 * t2
        return val

# bowl-shaped
class perm():
    def __init__(self, n_dim = 10):
        self.boundaries = np.array([- n_dim, n_dim])

    def f(self, x):
        val = 0
        for j in range(len(x)):
            v = 0
            for i in range(len(x)):
                v += (i + 2) * (x[i] ** (j + 1) - ( ( 1 / (i + 1) ) ** (j + 1) ) )
            val += v ** 2
        return val

class rotated_hyper_ellipsoid():
    def __init__(self):
        self.boundaries = np.array([-65, 65])

    def f(self, x):
        val = 0
        for i in range(len(x)):
            val += np.sum(x[:i + 1] ** 2)
        return val

class sphere():
    def __init__(self):
        self.boundaries = np.array([-5, 5])

    def f(self, x):
        return np.sum(x ** 2)

class different_power():
    def __init__(self):
        self.boundaries = np.array([-1, 1])

    def f(self, x):
        val = 0
        for i, v in enumerate(x):
            val += np.abs(v) ** (i + 2)
        return val

class weighted_sphere():
    def __init__(self):
        self.boundaries = np.array([-5, 5])

    def f(self, x):
        val = np.array([ (i + 1) * xi ** 2 for i, xi in enumerate(x)])
        return np.sum(val)

class trid():
    def __init__(self, n_dim = 50):
        self.boundaries = np.array([- n_dim ** 2, n_dim ** 2])

    def f(self, x):
        n_dim = len(x)
        t1 = np.sum( (x - 1) ** 2 )
        t2 = - np.sum( x[1:n_dim] * x[0:n_dim - 1] )
        return t1 + t2

# plate-shaped
class zakharov():
    def __init__(self):
        self.boundaries = np.array([-5, 5])

    def f(self, x):
        t1 = np.sum(x ** 2)
        w = np.array([ i + 1 for i in range(len(x))])
        wx = np.dot(w, x)
        t2 = 0.5 ** 2 * wx ** 2
        t3 = 0.5 ** 4 * wx ** 4
        return t1 + t2 + t3

# valley-shaped
class dixon_price():
    def __init__(self):
        self.boundaries = np.array([-10, 10])

    def f(self, x):
        n_dim = len(x)
        c = np.array([i + 2 for i in range(n_dim - 1)])
        t1 = (x[0] - 1) ** 2
        t2 = np.sum( c * (2 * x[1:n_dim] ** 2 - x[0:n_dim - 1] ) ** 2 )
        return t1 + t2

class rosenbrock_chain():
    def __init__(self):
        self.boundaries = np.array([-5, 5])

    def f(self, x):
        val = 0
        for i in range(0, len(x) - 1):
            t1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
            t2 = (x[i] - 1) ** 2
            val += t1 + t2
        return val

class rosenbrock_star():
    def __init__(self):
        self.boundaries = np.array([-5, 5])

    def f(self, x):
        val = 0
        for i in range(1, len(x)):
            t1 = 100 * (x[0] - x[i] ** 2) ** 2
            t2 = (x[i] - 1) ** 2
            val += t1 + t2
        return val

class k_tablet():
    def __init__(self):
        self.boundaries = np.array([-5.12, 5.12])

    def f(self, x):
        k = int(np.ceil(len(x) / 4.0))
        t1 = np.sum(x[:k] ** 2)
        t2 = 100 ** 2 * np.sum(x[k:] ** 2)
        return t1 + t2

# other
class styblinski():
    def __init__(self):
        self.boundaries = np.array([-5, 5])

    def f(self, x):
        t1 = np.sum(x ** 4)
        t2 = - 16 * np.sum(x ** 2)
        t3 = 5 * np.sum(x)
        return 39.16599 * len(x) + 0.5 * (t1 + t2 + t3)