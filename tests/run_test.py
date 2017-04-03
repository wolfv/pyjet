import jet
import sys
from random import randint
from inspect import getargspec
from helpers import cprint
from jet.jit import jit
import numpy as np


ERROR_MAX = 1.e-12

mode = 0 if 'continue' in sys.argv else 1

dimensions = []
for f_name in ['0d', '1d', '2d']:
    with open(f_name) as file:
        dimensions_raw = file.readlines()
    dimensions.append([line.strip() for line in dimensions_raw if line.strip()])

mix = []
for f_name in ['mix0d', 'mix1d']:
    with open(f_name) as file:
        mix_raw = file.readlines()
    mix.append([line.strip() for line in mix_raw if line.strip()])

def equal(x, y, member, tol=ERROR_MAX):
    if np.any(np.isnan(x)):
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
    if mode:
        assert(np.all(abs(x - y) < tol))
    else:
        try:
            cprint(member, np.all(abs(x - y) < tol), (x, y))
        except:
            cprint(member, False, (x, y))

def rand_mat(shape):
    return 2.0 * np.random.rand(*shape) - 1.0

def compare(dim, member):
    jet_func = jet.__dict__[member]
    numpy_func = np.__dict__[member]
    args = getargspec(jet.intake.__dict__[member]).args
    if dim == 0: 
        shape = ()
    elif dim == 1:
        shape = (3,)
    elif dim == 2:
        shape = (4, 4)

    mat_1 = rand_mat(shape)
    mat_2 = rand_mat(shape)

    try:
        if args in [['array'], ['x']]:
            equal(numpy_func(mat_1),
                  jit(jet_func)(mat_1), member)
        elif args == ['x', 'y']:
            equal(numpy_func(mat_1, mat_2),
                  jit(jet_func)(mat_1, mat_2), member) 
        elif 'input_tuple' in args:
            equal(numpy_func((mat_1, mat_2)),
                  jit(lambda x, y: jet_func((x, y)))(mat_1, mat_2), member)
        elif 'size' in args:
            equal(numpy_func(shape),
                  jit(lambda: jet_func(shape))(), member)
        elif member == 'eye':
            equal(numpy_func(shape[0]),
                  jit(lambda: jet_func(shape[0]))(), member)
        elif member == 'where':
            equal(numpy_func(False, mat_1, mat_2),
                  jit(lambda x, y: jet_func(False, x, y))(mat_1, mat_2), member + ', False')
            equal(numpy_func(True, mat_1, mat_2),
                  jit(lambda x, y: jet_func(True, x, y))(mat_1, mat_2), member + ', True')
        elif member == 'clip':
            equal(numpy_func(mat_1, -0.5, 0.5),
                  jit(lambda x: jet_func(x, -0.5, 0.5))(mat_1), member)
        elif member == 'reshape':
            equal(numpy_func(mat_1, (shape[0] / 2, shape[1] * 2)),
                  jit(lambda x: jet_func(x, (shape[0] / 2, shape[1] * 2)))(mat_1), member + ', 2D')
            equal(numpy_func(mat_1, (shape[0] ** 2,)),
                  jit(lambda x: jet_func(x, (shape[0] ** 2,)))(mat_1), member + ', 1D')
        else:
            assert(False)
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        e = sys.exc_info()[0]
        if mode:
            raise(e)
        else:
            cprint(member, False, e)

def test_0D():
    print("-------")
    print("0D")
    print("-------")

    for member in dimensions[0]:
        compare(0, member)

def test_1D():
    print("-------")
    print("1D")
    print("-------")

    for member in dimensions[1]:
        compare(1, member)

def test_2D():
    print("-------")
    print("2D")
    print("-------")

    for member in dimensions[2]:
        compare(2, member)

def test_mix():
    for dim, members in enumerate(mix):
        print("-------")
        print("2D x {dim}D".format(dim=dim))
        print("-------")

        mat = rand_mat((4, 4))
        vec = rand_mat((4,))
        scalar = rand_mat(())

        for member in members:
            jet_func = jet.__dict__[member]
            numpy_func = np.__dict__[member]
            try:
                if dim == 0:
                    equal(numpy_func(mat, scalar),
                          jit(jet_func)(mat, scalar), member)
                else:
                    equal(numpy_func(mat, vec),
                          jit(jet_func)(mat, vec), member)
            except SystemExit:
                 cprint(member, False)

def test_linalg():
    mat = rand_mat((3, 3))
    vec = rand_mat((3,))

    print("-------")
    print("linalg")
    print("-------")

    equal(np.linalg.solve(mat, vec),
          jit(jet.linalg.solve)(mat, vec), 'solve')
    equal(np.linalg.norm(vec),
          jit(lambda x: jet.linalg.norm(x))(vec), 'norm')

def test_random():
    print("-------")
    print("random")
    print("-------")

    scalar_1 = rand_mat(())
    scalar_2 = rand_mat(()) + 1.0

    n = int(1e5)
    random = np.zeros((n,))
    jet_normal = jit(jet.random.normal)
    for i in xrange(n):
        random[i] = jet_normal(scalar_1, scalar_2)
    equal(np.sum(random) / n,
          scalar_1, 'normal, mean', tol=1.e-1)
    equal(np.std(random),
          scalar_2, 'normal, sd', tol=1.e-2)


if __name__ == '__main__':
    test_0D()
    test_1D()
    test_2D()
    test_mix()
    test_random()
    test_linalg()
