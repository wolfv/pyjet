import numpy as np
from pprint import pprint
import networkx as nx
from jet.utils import get_unique_name, get_caller_info, slice_to_str
from jet import config


# global expander variables
constants = []
placeholder_count = 0
graph = nx.DiGraph()

# hack to fix issues with circular dependencies
intake = None
def import_intake():
    global intake
    if intake is None:
        from jet import intake

##########################################################################
####                          Operations                              ####
##########################################################################

class Op(object):
    # Base class for all operations
    def __init__(self, inputs, *args, **kwargs):
        self.inputs_raw = inputs
        self.inputs = check_type(*inputs)
        self.output = None
        self.init_op(self.inputs, *args, **kwargs)
        if config.debug or config.group_class or config.group_func:
            self.caller_info = get_caller_info('expander.py', 'intake.py')
        else:
            self.caller_info = None
        self.add_to_graph()

    def init_op(self, inputs, *args, **kwargs):
        pass

    def add_to_graph(self):
        if hasattr(self, 'name'): # TODO dominique: is this check here? the class has a property 'name'
            graph.add_node(self, name=self.name)
        else:
            name=get_unique_name('uniquename')
            graph.add_node(self, name=name)
        if self.inputs is not None:
            for arr in self.inputs:
                if hasattr(arr, 'assignment') and arr.assignment:
                    graph.add_edge(arr.assignment[-1], self, array=arr.name)
                elif hasattr(arr, 'producer'):
                    graph.add_edge(arr.producer, self, array=arr.name)

    def get_output(self):
        return self.output

    @property
    def name(self):
        return self.output.name

    @property
    def dtype(self):
        return self.output.dtype

    def __repr__(self):
        out_name = ''
        if self.output is not None:
            out_name = self.output.name + '\\n' + str(self.output.shape)
        return out_name

class ConstOp(Op):
    def init_op(self, inputs):
        self.op = 'Const'
        self.inputs = None
        self.value_holder = inputs[0]
        self.output = inputs[0]

    def values(self):
        return self.inputs

class VariableOp(Op):
    def init_op(self, inputs):
        self.op = 'Variable'
        self.inputs = None
        self.value_holder = inputs[0]
        self.output = inputs[0]

class PlaceholderOp(Op):
    def init_op(self, inputs, placeholder_count):
        self.op = 'Placeholder'
        self.placeholder = inputs[0]
        self.inputs = None
        self.output = inputs[0]
        self.placeholder_count = placeholder_count

class CreateArrayOp(Op):
    def __init__(self, inputs, producer, shape):
        self.op = 'CreateArray'
        self.inputs = inputs
        self.nested_input = np.array(inputs, dtype=object).reshape(shape)
        self.output = producer # TODO dominique: this is some weird naming convention: a producer is an output? isn't a producer normally responsible for the input?
        self.shape = shape
        if config.debug or config.group_class or config.group_func:
            self.caller_info = get_caller_info('expander.py', 'intake.py')
        else:
            self.caller_info = None
        self.add_to_graph()

class AssignOp(Op):
    def __init__(self, inputs, at_idx=None, slices=None):
        self.op = 'Assign'
        self.inputs = check_type(*inputs)
        self.slices = slices
        if slices is not None:
            if len(slices) == 1:
                slice_shape = np.zeros(inputs[0].shape)[slices[0]].shape
            else:
                slice_shape = np.zeros(inputs[0].shape)[slices[0], slices[1]].shape
            self.output = intake.array(name=self.op, dtype=inputs[0].dtype, 
                                       shape=inputs[0].shape,
                                       producer=self,
                                       slice_shape=slice_shape)
            self.__repr__ = lambda : '{}[{}, {}]'.format(self.inputs[0].name, 
                    slice_to_str(slices[0]),
                    slice_to_str(slices[1])) + '\\n' + str(self.output.shape)
        else:
            self.output = intake.array(name=self.op,
                                       dtype=inputs[0].dtype,
                                       shape=inputs[0].shape,
                                       producer=self)
        self.at_idx = at_idx
        if at_idx:
            if len(at_idx) == 1:
                self.__repr__ = lambda : '{}[{}]'.format(self.inputs[0].name,
                                    at_idx[0]) + '\\n' + str(self.output.shape)
            else:
                self.__repr__ = lambda : '{}[{}, {}]'.format(self.inputs[0].name,
                        at_idx[0], at_idx[1]) + '\\n' + str(self.output.shape)
        # special thing for assign operator... have to add the others
        # as dependency
        successors = graph.successors(inputs[0].last_producer)
        for s in successors:
            if s != self.inputs[0].last_producer:
                graph.add_edge(s, self, edge_type='helper')
        if config.debug or config.group_class or config.group_func:
            self.caller_info = get_caller_info('expander.py', 'intake.py')
        else:
            self.caller_info = None
        self.add_to_graph()
        self.inputs[0].assignment.append(self)

    def __repr__(self):
        return self.__repr__()

class ViewOp(Op):
    def init_op(self, inputs, keys):
        self.op = 'View'
        # find shape by slicing zeros vector
        self.slices = keys
        if len(self.slices) == 1:
            new_shape = np.zeros(inputs[0].shape)[self.slices[0]].shape
        else:
            new_shape = np.zeros(inputs[0].shape)[self.slices[0], self.slices[1]].shape

        self.output = intake.array(name=self.op, shape=new_shape, producer=self)

    def __repr__(self):
        if len(self.slices) == 1:
            return '{}[{}]'.format(self.inputs[0].name,
                    slice_to_str(self.slices[0])) + '\\n' + str(self.output.shape)
        return '{}[{}, {}]'.format(self.inputs[0].name,
                slice_to_str(self.slices[0]), 
                slice_to_str(self.slices[1])) + '\\n' + str(self.output.shape)

class ArrayAccessOp(Op):
    def init_op(self, inputs, at_idx):
        self.op = 'ArrayAccess'
        if len(at_idx) == 1:
            shape = np.zeros(inputs[0].shape)[at_idx[0]].shape
        else:
            shape = np.zeros(inputs[0].shape)[at_idx[0], at_idx[1]].shape
        self.output = intake.array(name=self.op, shape=shape, producer=self)
        self.at_idx = at_idx

    def __repr__(self):
        if len(self.at_idx) == 1:
            return '{}[{}]'.format(self.inputs[0].name, 
                                self.at_idx[0]) + '\\n' + str(self.output.shape)
        return '{}[{}, {}]'.format(self.inputs[0].name, 
                self.at_idx[0], self.at_idx[1]) + '\\n' + str(self.output.shape)

class ConcatenateOp(Op):
    def init_op(self, inputs, axis):
        self.op = 'Concatenate'
        self.axis=axis
        self.shape = np.concatenate((np.zeros(inputs[0].shape),
                                    np.zeros(inputs[1].shape)), axis=axis).shape
        self.output = intake.array(name=self.op, shape=self.shape,
                              dtype=upcast(inputs),
                              producer=self)

class WhereOp(Op):
    def init_op(self, inputs):
        self.op = 'Where'
        assert(inputs[1].shape == inputs[2].shape)
        shape = inputs[1].shape
        self.output = intake.array(name=self.op, shape=shape,
                              dtype=upcast(inputs[1:3]),
                              producer=self)
# not implemented yet
class WhileOp(Op):
        def init_op(self, inputs):
            self.op = 'While'
            self.output = inputs[1:]

class ZerosOp(Op):
    def init_op(self, inputs, shape, dtype=config.DTYPE):
        self.op = 'Zeros'
        self.shape = shape
        self.output = intake.array(name='zeros_mat', shape=self.shape,
                              dtype=dtype, producer=self)

class OnesOp(Op):
    def init_op(self, inputs, shape, dtype=config.DTYPE):
        self.op = 'Ones'
        self.shape = shape
        self.output = intake.array(name='ones_mat', shape=self.shape,
                              dtype=dtype, producer=self)

class EyeOp(Op):
    def init_op(self, inputs, shape, dtype=config.DTYPE):
        self.op = 'Eye'
        self.shape = shape
        self.output = intake.array(name='eye_mat', shape=self.shape,
                              dtype=dtype, producer=self)

class MatMulOp(Op):
    def init_op(self, inputs):
        self.op = 'MatMul'
        # result of mat mul = new matrix
        # n x m * m x p -> n * p
        shape = np.dot(np.zeros(inputs[0].shape), np.zeros(inputs[1].shape)).shape
        self.output = intake.array(name=self.op,
                                   shape=shape,
                                   dtype=upcast(inputs),
                                   producer=self)

class DotOp(Op):
    """
    Note:

    We use an explicit DOT operation instead, but we parse the numpy
    dot correctly (ie. dot between 2D matrices -> matmul, dot between vectors -> inner product)
    """
    def init_op(self, inputs):
        self.op = 'Dot'
        # result of mat mul = new matrix
        # n x m * m x p -> n * p
        shape = np.dot(np.zeros(inputs[0].shape), np.zeros(inputs[1].shape)).shape
        self.output = intake.array(name=self.op,
                                   shape=shape,
                                   dtype=upcast(inputs),
                                   producer=self)

class ModOp(Op):
    def init_op(self, inputs):
        self.op = 'Mod'
        shape = self.inputs[0].shape
        self.output = intake.array(name=self.op, shape=shape,
                              dtype=config.DTYPE,
                              producer=self)

class ArcTan2Op(Op):
    def init_op(self, inputs):
        self.op = 'ArcTan2'
        shape = self.inputs[0].shape
        self.output = intake.array(name=self.op, shape=shape,
                              dtype=config.DTYPE,
                              producer=self)

class ClipOp(Op):
    def init_op(self, inputs):
        self.op = 'Clip'
        assert(inputs[1].shape == inputs[2].shape)
        shape = inputs[0].shape
        self.output = intake.array(name=self.op, shape=shape,
                                  dtype=upcast(inputs),
                                  producer=self)

##########################################################################
#### Unary Operations ####
##########################################################################

class UnaryOp(Op):
    def init_op(self, inputs):
        # TODO check why this happens and what it does
        shape = self.inputs[0].shape
        if hasattr(self, 'dtype'):
            # dtype for sin etc. has to change
            dtype = self.dtype
        else:
            dtype = self.inputs[0].dtype
        self.output = intake.array(name=self.op, shape=shape, dtype=dtype,
                              producer=self)

class NegOp(UnaryOp):
    op = 'Neg'

class CopyOp(UnaryOp):
    op = 'Copy'

class SinOp(UnaryOp):
    op = 'Sin'
    dtype = config.DTYPE

class CosOp(UnaryOp):
    op = 'Cos'
    dtype = config.DTYPE

class TanOp(UnaryOp):
    op = 'Tan'
    dtype = config.DTYPE

class SqrtOp(UnaryOp):
    op = 'Sqrt'
    dtype = config.DTYPE

class SquareOp(UnaryOp):
    op = 'Square'
    dtype = config.DTYPE

class AbsOp(UnaryOp):
    op = 'Abs'
    dtype = config.DTYPE

class ExpOp(UnaryOp):
    op = 'Exp'
    dtype = config.DTYPE

class LogOp(UnaryOp):
    op = 'Log'
    dtype = config.DTYPE

class ArcSinOp(UnaryOp):
    op = 'ArcSin'
    dtype = config.DTYPE

class ArcCosOp(UnaryOp):
    op = 'ArcCos'
    dtype = config.DTYPE

class ArcTanOp(UnaryOp):
    op = 'ArcTan'
    dtype = config.DTYPE

##########################################################################
####                       Binary Operations                          ####
##########################################################################

class BinOp(Op):
    """
    Base class for binary ops that don't modify the shape
    ie: 3 + [1, 2, 3] = [4, 5, 6]
    or [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
    """
    def init_op(self, inputs):
        self.output = intake.array(name=self.op, shape=self.shape_op(inputs),
                              dtype=upcast(inputs), producer=self)

    def shape_op(self, inputs):
        return ()

class AddOp(BinOp):
    op = 'Add'
    def shape_op(self, inputs):
        return (np.zeros(inputs[0].shape) +
                np.zeros(inputs[1].shape)).shape

class MultiplyOp(BinOp):
    op = 'Mul'
    def shape_op(self, inputs):
        return (np.zeros(inputs[0].shape) *
                np.zeros(inputs[1].shape)).shape

class DivideOp(BinOp):
    op = 'Div'
    def shape_op(self, inputs):
        return (np.ones(inputs[0].shape) /
                np.ones(inputs[1].shape)).shape

class SubtractOp(BinOp):
    op = 'Sub'
    def shape_op(self, inputs):
        return (np.zeros(inputs[0].shape) -
                np.zeros(inputs[1].shape)).shape

class PowOp(BinOp):
    # Elementwise pow
    op = 'Pow'
    def shape_op(self, inputs):
        return (np.zeros(inputs[0].shape) **
                np.ones(inputs[1].shape)).shape

class CrossOp(BinOp):
    op = 'Cross'
    def shape_op(self, inputs):
        return np.cross(np.zeros(inputs[0].shape),
               np.zeros(inputs[1].shape)).shape


##########################################################################
####                       Boolean Operations                         ####
##########################################################################

class BoolOp(BinOp):
    def init_op(self, inputs):
        super(BoolOp, self).init_op(inputs)
        self.output.dtype = np.dtype(np.bool)

class LessOp(BoolOp):
    op = 'Less'

class LessEqualOp(BoolOp):
    op = 'LessEqual'

class EqualOp(BoolOp):
    op = 'Equal'

class NotEqualOp(BoolOp):
    op = 'NotEqual'

class GreaterOp(BoolOp):
    op = 'Greater'

class GreaterEqualOp(BoolOp):
    op = 'GreaterEqual'

class AndOp(BoolOp):
    op = 'And'

class OrOp(BoolOp):
    op = 'Or'

class NotOp(UnaryOp):
    def init_op(self, inputs):
        self.op = 'Not'
        super(NotOp, self).init_op(inputs)
        self.output.dtype = np.dtype(np.bool)

##########################################################################
####                          Linalg Operations                       ####
##########################################################################


class SolveOp(Op):
    def init_op(self, inputs):
        self.op = 'Solve'
        self.add_to_graph()
        if not (inputs[0].shape[0] == inputs[0].shape[1] ==
                inputs[1].shape[0]):
            raise ValueError(
                'LHS shape {} and RHS shape {} not compatible for solve.'
                .format(inputs[0].shape, inputs[1].shape))
        self.output = intake.array(name=self.op, shape=inputs[1].shape,
                              dtype=upcast(inputs), producer=self)

class NormOp(Op):
    def init_op(self, inputs, order):
        self.op = 'Norm'
        if order != 2:
            raise NotImplementedError('Only 2 order norm supported currently.')
        if inputs[0].ndim > 1:
            if inputs[0].shape[1] > 1:
                raise NotImplementedError('Only vectors supported.')
        self.output = intake.array(name=self.op, shape=(), dtype=upcast(inputs),
                              producer=self)

##########################################################################
####                        Random Operations                         ####
##########################################################################

class RandomNormalOp(Op):
    def init_op(self, inputs):
        self.op = 'RandomNormal'
        shape = (np.zeros(inputs[0].shape) + np.zeros(inputs[1].shape)).shape
        self.output = intake.array(name=self.op,
                                   shape=shape,
                                   dtype=config.DTYPE,
                                   producer=self)

##########################################################################
####                      Functions on Objects                        ####
##########################################################################

class RavelOp(Op):
    def init_op(self, inputs):
        self.op = 'Ravel'
        shape = np.zeros(inputs[0].shape).ravel().shape
        self.output = intake.array(name=self.op,
                                   shape=shape, 
                                   dtype=inputs[0].dtype,
                                   producer=self)

class AnyOp(Op):
    def init_op(self, inputs):
        self.op = 'Any'
        self.output = intake.array(name=self.op,
                                   shape=inputs[0].shape,
                                   dtype=bool,
                                   producer=self)

class TransposeOp(Op):
    def init_op(self, inputs):
        self.op = 'Transpose'
        shape = np.zeros(inputs[0].shape).transpose().shape
        self.output = intake.array(name=self.op,
                                   shape=shape,
                                   dtype=inputs[0].dtype,
                                   producer=self)

class ReshapeOp(Op):
    def init_op(self, inputs, shape):
        self.op = 'Reshape'
        self.output = intake.array(name=self.op,
                                   shape=shape, 
                                   dtype=inputs[0].dtype, 
                                   producer=self)

class MaxOp(Op):
    def init_op(self, inputs):
        self.op = 'Max'
        self.output = intake.array(name=self.op,
                                   shape=(), 
                                   dtype=inputs[0].dtype, 
                                   producer=self)

class MinOp(Op):
    def init_op(self, inputs):
        self.op = 'Min'
        self.output = intake.array(name=self.op,
                                   shape=(), 
                                   dtype=inputs[0].dtype, 
                                   producer=self)

##########################################################################
####                             Utils                                ####
##########################################################################

def upcast(inputs):
    return np.find_common_type([i.dtype for i in inputs], [])

def print_graph():
    pprint(graph.graph)
    pprint("Nodes of graph: ")
    pprint(graph.nodes())
    pprint("Edges of graph: ")
    pprint(graph.edges())

def find_node(n):
    for node in graph.nodes_iter():
        if node == n.producer:
            return node

def check_type(*args):
    arg_list = list(args)
    num_types = [float, int, bool, np.ndarray, np.float64, np.float32]
    jet_types = [intake.variable, intake.constant, intake.placeholder, intake.ndarray]

    for i, arg in enumerate(args):
        if arg is not None and type(arg) not in jet_types:
            if type(arg) in num_types:
                # this is a constant
                arg_list[i] = intake.constant(arg)
                constants.append(arg)
            else:
                raise ValueError('Unexpected type \'{}\'.'.format(type(arg)))
    return arg_list
