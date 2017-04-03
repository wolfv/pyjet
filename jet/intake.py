import numpy
from jet import config
from jet import utils
from jet import expander
from jet import helpers


##########################################################################
####                       General Functions                          ####
##########################################################################

def clip(array, a_min, a_max):
    op = expander.ClipOp([array, a_min, a_max])
    return op.get_output()

def where(condition, x, y):
    # when True, yield x, otherwise yield y
    op = expander.WhereOp([condition, x, y])
    return op.get_output()

def concatenate(input_tuple, axis=0):
    # join elements along axis
    if len(input_tuple) > 2:
        con = [concatenate(input_tuple[1:], axis=axis)]
        op = expander.ConcatenateOp(tuple([input_tuple[0]] + con), axis=axis)
    else:
        op = expander.ConcatenateOp(input_tuple, axis=axis)
    return op.get_output()

def vstack(input_tuple):
    return concatenate(input_tuple, axis=0)

def hstack(input_tuple):
    return concatenate(input_tuple, axis=1)

def zeros(size, dtype=config.DTYPE):
    if isinstance(size, int):
        shape = (size,)
    else:
        shape = size
    op = expander.ZerosOp([None], shape, dtype)
    return op.get_output()

def ones(size, dtype=config.DTYPE):
    if isinstance(size, int):
        shape = (size,)
    else:
        shape = size
    op = expander.OnesOp([None], shape, dtype)
    return op.get_output()

def eye(n, dtype=config.DTYPE):
    op = expander.EyeOp([None], (n, n), dtype)
    return op.get_output()

##########################################################################
####                          Binary Functions                        ####
##########################################################################

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def power(x, y):
    return x ** y

def divide(x, y):
    return x / y

def true_divide(x, y):
    return x / y

def multiply(x, y):
    return x * y

def maximum(x, y):
    return where(x > y, x, y)

def minimum(x, y):
    return where(x < y, x, y)

def matmul(x, y):
    op = expander.MatMulOp([x, y])
    return op.get_output()

def dot(x, y):
    if len(x.shape) == 1 and len(y.shape) == 1:
        op = expander.DotOp([x, y])
    else:
        op = expander.MatMulOp([x, y])
    return op.get_output()

def cross(x, y):
    op = expander.CrossOp([x, y])
    return op.get_output()

def mod(x, y):
    op = expander.ModOp([x, y])
    return op.get_output()

##########################################################################
####                      Trigonometric Functions                     ####
##########################################################################

def sin(x):
    op = expander.SinOp([x])
    return op.get_output()

def arcsin(x):
    op = expander.ArcSinOp([x])
    return op.get_output()

def cos(x):
    op = expander.CosOp([x])
    return op.get_output()

def arccos(x):
    op = expander.ArcCosOp([x])
    return op.get_output()

def tan(x):
    op = expander.TanOp([x])
    return op.get_output()

def arctan(x):
    op = expander.ArcTanOp([x])
    return op.get_output()

def arctan2(x, y):
    op = expander.ArcTan2Op([x, y])
    return op.get_output()

##########################################################################
####                          Unary Functions                         ####
##########################################################################

def negative(x):
    return - x

def reciprocal(x):
    return 1.0 / x

def fabs(x):
    op = expander.AbsOp([x])
    return op.get_output()

def sign(x):
    return where(x > 0.0, 1.0, where(x < 0.0, -1.0, 0.0))

def sqrt(x):
    op = expander.SqrtOp([x])
    return op.get_output()

def square(x):
    op = expander.SquareOp([x])
    return op.get_output()

def exp(x):
    op = expander.ExpOp([x])
    return op.get_output()

def log(x):
    op = expander.LogOp([x])
    return op.get_output()

##########################################################################
####                           Array Functions                        ####
##########################################################################

def transpose(array):
    return array.transpose()

def ravel(array):
    return array.ravel()

def reshape(array, shape):
    return array.reshape(shape)

def max(array):
    return array.max()

def amax(array):
    return array.amax()

def min(array):
    return array.min()

def amin(array):
    return array.amin()

##########################################################################
####                         Logical Functions                        ####
##########################################################################

def logical_and(x, y):
    op = expander.AndOp([x, y])
    return op.get_output()

def logical_or(x, y):
    op = expander.OrOp([x, y])
    return op.get_output()

def logical_xor(x, y):
    return logical_and(logical_or(x, y),
                       logical_or(logical_not(x), logical_not(y)))

def logical_not(x):
    op = expander.NotOp([x])
    return op.get_output()

##########################################################################
####                         Other Functions                          ####
##########################################################################

@helpers.jet_static_class
class linalg(object):
    def solve( lhs, rhs):
        op = expander.SolveOp([lhs, rhs])
        return op.get_output()

    def norm(x, order=2):
        op = expander.NormOp([x], order)
        return op.get_output()

@helpers.jet_static_class
class random(object):
    def normal(mean=0, sd=1):
        op = expander.RandomNormalOp([mean, sd])
        return op.get_output()

##########################################################################
####                         Array Objects                            ####
##########################################################################

# not compatible with numpy's ndarray
class ndarray(object): # TODO dominique: PEP8 requires CamelCase for class names
    def __init__(self, value=None, name='array', shape=(), dtype=config.DTYPE,
                 producer=None, **kwargs):
        if value is None:
            self.value = numpy.array(None)
            self.shape = shape
        else:
            self.value = numpy.array(value, dtype=object)
            self.shape = self.value.shape
            if len(shape) > 2:
                raise NotImplementedError('Only up to 2 dimensions supported.')
            value = expander.check_type(*self.value.flatten().tolist())
            self.value = numpy.array(value, dtype=object).reshape(self.shape)
            op = expander.CreateArrayOp(value, self, self.shape)
            producer = op

        self.ndim = len(shape)
        self.dtype = numpy.dtype(dtype)
        self.name = utils.get_unique_name(name)
        self.producer = producer
        self.assignment = []
        self.kwargs = kwargs

    # base class for all variable objects
    def __getitem__(self, keys):
        if not hasattr(keys, '__iter__'):
            keys = [keys]
        if all(type(s) == int for s in keys):
            # This is just a plain array access ...
            if any(a >= b for a, b in zip(keys, self.shape)): # TODO dominique: what happens if len(keys) != len(self.shape)?
                raise StopIteration # TODO dominique: should be IndexError, see https://docs.python.org/2/reference/datamodel.html#object.__getitem__
            if len(self.shape) == len(keys):
                op = expander.ArrayAccessOp([self], at_idx=keys)
                return op.get_output()
        op = expander.ViewOp(inputs=[self], keys=list(keys))

        return op.get_output()

    def __setitem__(self, keys, value):
        if not hasattr(keys, '__iter__'):
            keys = [keys]
        if all(type(s) == int for s in keys):
            # This is just a plain array access ...
            if any(a >= b for a, b in zip(keys, self.shape)): # TODO dominique: what happens if len(keys) != len(self.shape)?
                raise StopIteration # TODO dominique: should be IndexError, same as for __getitem__
            if len(self.shape) == len(keys):
                op = expander.AssignOp([self, value], at_idx=keys)
        else:
            op = expander.AssignOp(inputs=[self, value], slices=keys)
        self.assignment.append(op)

        return op.get_output()

    def __add__(self, rhs):
        op = expander.AddOp([self, rhs])
        return op.get_output()

    def __mul__(self, rhs):
        op = expander.MultiplyOp([self, rhs])
        return op.get_output()

    def __sub__(self, rhs, rsub=True):
        if rsub:
            op = expander.SubtractOp([self, rhs])
        else:
            op = expander.SubtractOp([rhs, self])
        return op.get_output()

    def __div__(self, rhs, rdiv=True):
        if rdiv:
            op = expander.DivideOp([self, rhs])
        else:
            op = expander.DivideOp([rhs, self])
        return op.get_output()

    def __truediv__(self, rhs):
        # TODO check handling
        op = expander.DivideOp([self, rhs])
        return op.get_output()

    # TODO: enable power with integers without upcasting
    def __pow__(self, rhs):
        op = expander.PowOp([self, rhs])
        return op.get_output()

    __rmul__ = __mul__

    __radd__ = __add__

    def __rsub__(self, lhs):
        return self.__sub__(lhs, rsub=False)

    def __rdiv__(self, lhs):
        return self.__div__(lhs, rdiv=False)

    def __lt__(self, rhs):
        op = expander.LessOp([self, rhs])
        return op.get_output()

    def __le__(self, rhs):
        op = expander.LessEqualOp([self, rhs])
        return op.get_output()

    def __eq__(self, rhs):
        op = expander.EqualOp([self, rhs])
        return op.get_output()

    def __ne__(self, rhs):
        op = expander.NotEqualOp([self, rhs])
        return op.get_output()

    def __gt__(self, rhs):
        op = expander.GreaterOp([self, rhs])
        return op.get_output()

    def __ge__(self, rhs):
        op = expander.GreaterEqualOp([self, rhs])
        return op.get_output()

    def __iadd__(self, rhs):
        self = self + rhs
        return self.producer.get_output()

    def __idiv__(self, rhs):
        self = self / rhs
        return self.producer.get_output()

    def __imul__(self, rhs):
        self = self * rhs
        return self.producer.get_output()

    def __isub__(self, rhs):
        self = self - rhs
        return self.producer.get_output()

    def __ipow__(self, rhs):
        self = self ** rhs
        return self.producer.get_output()

    def __neg__(self):
        op = expander.NegOp(inputs=[self])
        return op.get_output()   

    def __repr__(self):
        return '<array: {}/{} {}>'.format(self.dtype.name, self.shape, self.name)

    @property
    def last_producer(self):
        if len(self.assignment):
            return self.assignment[-1]
        else:
            return self.producer

    @property
    def T(self):
        return self.transpose()

    def copy(self):
        op = expander.CopyOp([self])
        return op.get_output

    def transpose(self):
        op = expander.TransposeOp([self])
        return op.get_output()

    def ravel(self):
        op = expander.RavelOp([self])
        return op.get_output()

    def reshape(self, shape):
        op = expander.ReshapeOp([self], shape)
        return op.get_output()

    def any(self):
        op = expander.AnyOp([self])
        raise NotImplementedError()
        return op.get_output()

    def max(self):
        op = expander.MaxOp([self])
        return op.get_output()

    def amax(self):
        return self.max()

    def min(self):
        op = expander.MinOp([self])
        return op.get_output()

    def amin(self):
        return self.min()

    def __len__(self):
        if self.shape:
            return self.shape[0]
        else:
            raise TypeError()

    def __abs__(self):
        op = expander.AbsOp([self])
        return op.get_output()

    __array_priority__ = 10000
    def __array_wrap__(self, result):
        return constant(result)

def array(*args, **kwargs):
    return ndarray(*args, **kwargs)

class variable(ndarray): # TODO dominique: PEP8 requires CamelCase for class names
    # variable results in member variable in generated class
    def __init__(self, value=None, name='variable', shape=(), dtype=config.DTYPE):
        _check_not_jet_type(value, name)
        self.value = numpy.array(value)
        super(variable, self).__init__(value=None, name=name,
                                       shape=self.value.shape,
                                       dtype=dtype)
        self.value = numpy.array(value)
        op = expander.VariableOp([self])
        self.producer = op

class placeholder(ndarray): # TODO dominique: PEP8 requires CamelCase for class names
    # placeholder results in argument to be passed in generated function
    def __init__(self, name='placeholder', shape=(), dtype=config.DTYPE):
        super(placeholder, self).__init__(name=name, shape=shape, dtype=dtype)
        op = expander.PlaceholderOp([self],
                                placeholder_count=expander.placeholder_count)
        expander.placeholder_count += 1

        self.producer = op

    def __repr__(self):
        return '<placeholder: {}/{} {}>'.format(self.dtype.name, self.shape, 
                                                self.name)

class constant(ndarray): # TODO dominique: PEP8 requires CamelCase for class names
    # constant results in constant expression in generated class
    def __init__(self, value, name='constant', dtype=config.DTYPE):
        _check_not_jet_type(value, name)
        self.value = numpy.array(value)
        super(constant, self).__init__(value=None, name=name,
                                       shape=self.value.shape,
                                       dtype=config.DTYPE)
        self.value = numpy.array(value)
        op = expander.ConstOp([self])
        self.producer = op
        self.name +=  '/* {value} */'.format(value=self.value) if config.debug else ''

    def __repr__(self):
        return '<constant: {}/{} {}>'.format(self.dtype.name, self.shape, self.name)

##########################################################################
####                              Other                               ####
##########################################################################

def _check_not_jet_type(obj, name):
    if isinstance(obj, ndarray) or (isinstance(obj, numpy.ndarray) and
                                  obj.dtype == object):
        raise ValueError('Can\'t add this object to jet {}.'.format(name))

##########################################################################
####                              main                                ####
##########################################################################

if __name__ == '__main__':
    # test jet functionality here
    # example:
    import jet

    ph = jet.placeholder(name='holder', shape=(3, 3))
    var = jet.variable(name='variable', value=numpy.zeros((2,1)))
    const = jet.constant(name='constant', value=1.5)
    op = ph[1, 1] + ph[0:2, 0:2] * var + const
    out0 = jet.concatenate((var, op), axis=1)
    out1 = jet.linalg.norm(var)
    ph[0:2, 0:1] = var

    from jet.burn import draw
    draw(jet.graph, name='graph')

    from jet.compressor import JetBuilder

    jb = JetBuilder(out=[out0, out1], fun_name='test')
    test_module = jb.build()
    test_class = test_module.TestClass()

    print(test_class.variable)
    test_class.variable = numpy.array([1, 2]) # (n,)-shaped numpy vectors are represented 
                                   # as (n, 1)-shaped matrices in Armadillo
    print(test_class.test(numpy.ones((3,3))))

    # numpy to compare
    ph = numpy.ones((3,3))
    var = numpy.array([[1], [2]])
    const = 1.5
    op = ph[1, 1] + ph[0:2, 0:2] * var + const
    out0 = numpy.concatenate((var, op), axis=1)
    out1 = numpy.linalg.norm(var)
    ph[0:2, 0:1] = var
    print(out0, out1)
