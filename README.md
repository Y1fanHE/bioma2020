# Self Adaptive Cuckoo Search

## How to Use

1. Edit ```.yml``` files and run the following command.

```
python [test algorithm .py] [problem .yml] [algorithm .yml]
```

- E.g. Test **Cuckoo Search** on **SOP Set 1** with **ParameterEvolved** adaptive strategy.

```
python TestCS.py SOPyml/SOP1.yml ALGOyml/PECS.yml
```

2. Check results in ```tmp/{problem}/{algorithm}_{seed}.csv```.

   - sample line of output in ```.csv``` file.

```
1,2.032e+02,0.5,0.5     //{generations},{fitness},{mean of param1},{mean of param2}
```

## Self-adaptive Algorithms

- Base algorithm
    - differential evolution
    - cuckoo search
- Self-adaptive strategy
    - none
    - best levy
    - ga (with levy)
    - cauchy
    - jade

## Benchmark Functions

- Many local minima
    - ackley
    - griewank
    - levy
    - rastrigin
    - schwefel
    - xin she
    - schaffer
- Bowl-shaped
    - perm
    - rotated hyper ellipsoid
    - sphere
    - different power
    - weighted sphere
    - trid
- Plate-shaped
    - zakharov
- Valley-shaped
    - dixon price
    - rosenbrock chain
    - rosenbrock star
    - k tablet
- Others
    - styblinski
