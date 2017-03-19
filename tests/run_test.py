import jet
import jet.intake as intake
from jet.jit import jit
import numpy as np
from random import randint
from inspect import getargspec


jet.set_options(merge=False)

ERROR_MAX = 1.e-6

dimensions = []
for f_name in ['0d', '1d', '2d']:
    with open(f_name) as file:
        dimensions_raw = file.readlines()
    dimensions.append([line.strip() for line in dimensions_raw])

def equal(x, y):
    assert(np.all(abs(x - y) < ERROR_MAX))

def compare(dim, member):
    jet_func = intake.__dict__[member]
    if not member.startswith("_") and callable(jet_func):
        numpy_func = np.__dict__[member]
        args = getargspec(jet_func).args
        if dim == 0: 
            shape = ()
        elif dim == 1:
            shape = (3,)
        elif dim == 2:
            shape = (3, 3)

        mat = np.random.rand(*shape)
        if args in [['array'], ['x']]:
            equal(jit(jet_func)(mat), numpy_func(mat))
        elif args == ['x', 'y']:
            equal(jit(jet_func)(mat, mat), numpy_func(mat, mat)) 
        elif 'input_tuple' in args:
            equal(jit(lambda x, y: jet_func((x, y)))(mat, mat), numpy_func((mat, mat)))

def test():
    for dim, members in enumerate(dimensions):
        for member in members:
            compare(dim, member)


if __name__ == '__main__':
    test()
