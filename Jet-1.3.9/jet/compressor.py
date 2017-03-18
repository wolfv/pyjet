# This layer translates the OP graph to C++11

import numpy as np
import networkx as nx
from jet.utils import get_unique_name, beauty, sanitize_name
from jet import expander
from jet import config
from jet import exhaust


class_template = """
#define DEBUG {debug}

#include <cmath>
#include <tuple>
#include <vector>
#include <string>
#include <armadillo>
#if DEBUG
#include <iostream>
#include <limits>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "armadillo.h"
#include "supportlib.h"

using namespace arma;
using namespace std;


// constants
{consts}

class {class_name} {{
public:

    // member variables
    {states}

#if DEBUG
    // cell width for matrix
    unsigned int cell_width;

    // constructor
    {class_name}() {{
        // number of digits in double
        unsigned int double_digits = numeric_limits<double>::max_digits10; // 17
        cell_width = double_digits + 4;
        // full precision print
        cout.precision(double_digits - 1);
        // fixed format print out
        cout.setf(ios::scientific);
    }}
#endif

    // function
    tuple<{ret_types}> {fun_name}({args}) {{


        {ops}


        return make_tuple({ret_names});
    }}
}};

namespace py = pybind11;
void pyexport(py::module& m) {{
    py::class_<{class_name}>(m, "{class_name}")
        .def(py::init<>())
        .def("{fun_name}", &{class_name}::{fun_name})
        .def("args", []({class_name}&) {{
            return vector<string> {{ {arg_names} }};
        }})
        {state_accessors}
    ;
}}
"""
fmt_print_double = 'cout << {name} << endl;\n'
fmt_print_mat = 'cout.width(cell_width); {name}.raw_print(); cout.width(0);\n'
fmt_caller = '/* caller: {caller_class}, function {caller_fun}, {caller_line} */'
graph = None

dtype_map = {
    np.dtype(np.float32): 'float',
    np.dtype(np.float64): 'double',
    np.dtype(np.int32): 'int',
    np.dtype(np.int64): 'long',
    np.dtype(np.bool): 'bool',
}

##########################################################################
#### Top-Level ####
##########################################################################

class JetBuilder(object):
    """
    The class builder class is responsible for turning a jet-graph into a C++
    class.
    """
    def __init__(self, out, fun_name='func', file_name=None):
        """
        Construct a class builder

        Args:
            graph (nx.DiGraph): The jet graph containing all Ops
            out (list): List of nodes which form the output of the compiled
                        function
            fun_name (str): Name suffix of the compiled class.
                            If fun_name = FunName, the compiled class
                            would be called JetFunName.
        """
        if file_name is None:
            file_name = get_unique_name('jet_autogenerated')

        self.graph = expander.graph
        self.args = []
        self.ops = []
        self.variables = []
        self.constants = []
        self.Op = OpCollector()
        self.file_name = file_name
        self.fun_name = fun_name
        self.class_name = fun_name.title() + 'Class'
        self.extract_return(out)

        # extract sequential list, starting at output
        current_nodes = self._extract_nodes_for_return(out)
        self.subgraph = self.graph.subgraph(current_nodes)
        topo_sorted = nx.topological_sort(self.subgraph)
        self._merge_ops(self.subgraph, topo_sorted)

        for el in topo_sorted:
            if el not in current_nodes:
                continue
            if el.op == 'Placeholder':
                self.args.append(el)
            elif el.op == 'Variable':
                self.variables.append(el)
            elif el.op == 'Const':
                self.constants.append(el)
            else:
                self.ops.append(el)

        self.args = sorted(self.args, key=lambda x: x.placeholder_count)

        if config.draw_graph:
            import burn
            burn.draw(self.subgraph, outputs=out, name=fun_name)
        if config.draw_graph_raw:
            import burn
            burn.draw(self.graph, outputs=out, name=fun_name + '_raw')

    def extract_return(self, out):
        """
        Get the return names and types from the out argument.

        Args:
            out (list): list of graph nodes
        """
        self.return_names = [r.name for r in out]
        self.return_types = [get_type(r) for r in out]

    def _extract_nodes_for_return(self, out):
        """
        Starting at the out-nodes, this function traverses the graph backwards
        and collects all nodes necessary to compute the output-values.

        Args:
            out (list): list of nodes which are returned from the generated
                        function.

        Returns:
            Set of all nodes which are necessary to compute all outputs.
        """
        pres = set()
        for node in out:
            other_nodes = set()
            other_nodes.add(node.last_producer)
            pres.add(node.last_producer)

            while True:
                next_other_nodes = set()
                for node in other_nodes:
                    for pre in self.graph.predecessors(node):
                        edge_data = self.graph.get_edge_data(pre, node)
                        if not edge_data or edge_data.get('edge_type') != 'helper':
                                pres.add(pre)
                                next_other_nodes.add(pre)

                if not next_other_nodes:
                    break

                other_nodes = next_other_nodes
        return pres

    def _merge_ops(self, graph, ordered_nodes):
        """
        The unmerged output contains one Op per linem. But if Ops have only one
        successor, they can be merged. For example:

            Unmerged:
                auto Add_1 = Placeholder_1 + constant;
                auto Add_2 = Add_1 + constant_2;

            Merged:
                auto Add_2 = (Placeholder_1 + constant) + constant_2;

        Args:
            graph (nx.DiGraph): The jet graph
            ordered_nodes (list): The topologically sorted nodes
        """
        if config.debug or not config.merge:
            return

        mergeables = ['Add', 'Sub', 'Mul', 'Div', 'ArrayAccess', 'Pow']

        for node in ordered_nodes:
            if node.op not in mergeables:
                continue
            if len(graph.successors(node)) == 1:
                successor = graph.successors(node)[0]
                if successor.op in mergeables:
                    rhs = repr(self.Op(node)).split('=')[-1]
                    rhs_without_semicolon = rhs.strip(' ;')
                    merge_string = '({})'.format(rhs_without_semicolon)

                    # merge this node into the successor
                    if not hasattr(successor, 'merged_inputs'):
                        successor.merged_inputs = {}
                    successor.merged_inputs[node.name] = merge_string
                    node.node_merged = True

    def to_cpp(self):
        """
        Create the C++ representation of the graph

        Returns:
            str: The compilable C++
        """
        fmt_state_acc = '.def_readwrite("{name}", &{class_name}::{name})'
        all_ops = [op for op in self.ops if not hasattr(op, 'node_merged')]
        all_accessors = [fmt_state_acc.format(name=v.name,
                                              class_name=self.class_name) 
                                                        for v in self.variables]
        return beauty(class_template.format(
            debug='true' if config.debug else 'false',
            class_name = self.class_name,
            fun_name = self.fun_name,
            consts='\n'.join([repr(Constant(const)) for const in self.constants]),
            args=', '.join([repr(Placeholder(p)) for p in self.args]),
            states='\n'.join([repr(Variable(v)) for v in self.variables]),
            ops=('\ncout << "' + self.fun_name + ' --" << %s << "--" << endl;\n'
                ).join([repr(self.Op(op)) for op in all_ops]) % \
                            tuple(range(len(all_ops)-1)) if config.debug else \
                '\n'.join([repr(self.Op(op)) for op in all_ops]),
            ret_types=', '.join(self.return_types),
            ret_names=', '.join(self.return_names),
            arg_names=', '.join(['"{}"'.format(arg.name) for arg in self.args]),
            state_accessors='\n'.join(all_accessors))
        )

    def build(self):
        cpp = self.to_cpp()
        return exhaust.compile_cpp(cpp, self.file_name)

class BaseArray(object):
    def get_dtype(self):
        """
        Obtain the data type for the first output.

        Returns:
          A string containing the the dtype for the first output.
        """
        output = self.el.output
        shape = None
        if output.shape is not None:
            shape = output.shape
        dtype = output.dtype
        if not shape or len(shape) == 0:
            return dtype_map[dtype]
        elif len(shape) == 1:
            return 'Mat<{}>::fixed<{}, 1>'.format(dtype_map[dtype], shape[0])
        else:
            tmpl = 'Mat<{}>::fixed<{}>'.format(dtype_map[dtype], ', '.join(str(s) for s in
                [s for s in shape]))
            return '{type}'.format(type=tmpl)
        return 'auto /* get_dtype */'

##########################################################################
#### Variable Objects ####
##########################################################################

class Array(BaseArray):
    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.fmt_caller = fmt_caller.format(name=self.name,
                                caller_class=el.caller_info[0] if el.caller_info else '',
                                caller_fun=el.caller_info[2] if el.caller_info else '',
                                caller_line=el.caller_info[3] if el.caller_info else '') \
                          if config.debug else ''
        self.init_op()

    def init_op(self):
        pass

class Placeholder(Array):
    # placeholder results in argument to be passed in generated function
    def init_op(self):
        self.fmt = '{type} {name}' + self.fmt_caller
        # extract shape from node def
        shape = self.el.get_output().shape
        self.dtype = get_type(self.el, shape)

    def __repr__(self):
        return self.fmt.format(type=self.dtype, name=self.name)

class Constant(Array):
    # constant results in constant expression in generated class
    def init_op(self):
        self.fmt = 'constexpr {dtype} {name} = {val};' + self.fmt_caller
        self.fmt_mat = '{dtype} {name} = {init};' + self.fmt_caller
        self.value = self.el.value_holder.value
        self.dtype = self.el.output.dtype
        self.is_array = False
        self.shape = self.value.shape

    def __repr__(self):
        if type(self.value) == np.ndarray and len(self.shape):
            value_str = array2string(self.value)
            return self.fmt_mat.format(
                dtype=get_type(self.el, shape=self.shape),
                name=self.name,
                init=value_str
            )
        else:
            return self.fmt.format(
                dtype=self.get_dtype(),
                name=self.name,
                val=float_nsf(self.value))

class Variable(Array):
    # variable results in member variable in generated class
    def init_op(self):
        self.fmt = '{dtype} {name} = {val};' + self.fmt_caller
        self.fmt_empty = '{type} {name};' + self.fmt_caller
        self.value = self.el.value_holder.value
        self.dtype = self.el.output.dtype
        self.is_array = False
        self.shape = self.value.shape

    def __repr__(self):
        if type(self.value) == np.ndarray and len(self.shape):
            if self.value is not None:
                return self.fmt.format(
                    dtype=get_type(self.el, shape=self.shape),
                    name=self.name,
                    val=array2string(self.value)
            )
            return self.fmt_empty.format(
                dtype=get_type(self.el, shape=self.shape),
                name=self.name,
            )
        else:
            if self.value is not None:
                return self.fmt.format(
                    dtype=dtype_map[self.dtype],
                    name=self.name,
                    val=float_nsf(self.value))
            return self.fmt_empty.format(
                dtype=dtype_map[self.dtype],
                name=self.name,
            )

##########################################################################
#### Operations ####
##########################################################################

class OpRegister(BaseArray):
    """
    The OpRegister is a meta-object that is the super-class of all Ops.
    It contains some utility functions.
    """
    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.fmt_base = (' ' + fmt_caller + '\n' + (fmt_print_double 
                if el.output.shape == () else fmt_print_mat)).format(
                                name=self.name,
                                caller_class=el.caller_info[0] if el.caller_info else '',
                                caller_fun=el.caller_info[2] if el.caller_info else '',
                                caller_line=el.caller_info[3] if el.caller_info else '') \
                if config.debug else ''

        self.init_op()

    def init_op(self):
        pass

    def get_sanitized_inputs(self):
        """
        Obtain sanitized inputs for a Op. This is particularly necessary to use
        if the Op is defined as merge-able, as this function will return the
        properly merged input string.

        Returns:
          A list of strings representing the inputs, if necessary in merged
          form. For example: ['Add_2', '(Add_1 - Mul_1)'], where the second
          list value is a merged input.
        """
        has_merged = hasattr(self.el, 'merged_inputs')
        ret_inputs = []
        for inp in self.el.inputs:
            if has_merged and inp.name in self.el.merged_inputs:
                ret_inputs.append(self.el.merged_inputs[inp.name])
            else:
                ret_inputs.append(inp.name)
        return ret_inputs

class OpCollector:
    """
    Helper class to collect all subclasses of OpRegister. All classes that
    inherit from OpRegister are collected in this class. When an instance is
    created, it can be called with an Graph node and the corresponding Op for
    code generation is returned.

    This class also performs some "magic" by:
     - Checking the class name of the subclass. The suffix-"Op" is removed and
       the Op-type that is matched is chosen as the remainder. E.g. for the
       `ClipOp` the corresponding Op name is `Clip` which is registered here.
     - If the subclass has the class method `registers` all returned Op names
       are added as resolving to that specific class.

    Example (pseudo code): # TODO dominique: what does this example have to do with the OpCollector?
      >>> op_register_instance = OpRegister()
      >>> a = op_register_instance(Node(type='Add')) # TODO dominique: OpRegister does not seem to be callable
      AddOp( ... )
      >>> repr(a)
      "auto Add_1 = 2 + 2;"

    """
    def __init__(self):
        self.register = {}
        for cls in OpRegister.__subclasses__():
            # Check if subclass has the class method `registers`
            if hasattr(cls, 'registers'):
                regs = cls.registers()
                # if we got some sort of list ...
                if hasattr(regs, '__iter__'):
                    # add all op-names
                    for el in regs:
                        self.register[el] = cls
                else:
                    self.register[regs] = cls
            else:
                # this adds the class name minus the Op as op-name
                # e.g. -> `class ClipOp` registers `Clip`
                self.register[cls.__name__[:-2]] = cls

    def __call__(self, el):
        if self.register.get(el.op):
            return self.register[el.op](el)
        else:
            print('WARNING ::: Found no Op for {}'.format(el))
            return None

# not implemented yet
class WhileOp(OpRegister):
    def init_op(self):
        self.fmt = 'while ({cond}) {{\n{body}\n}}'

    # def __repr__(self):
    #     inputs = self.get_sanitized_inputs()
    #     return self.fmt.format(cond=inputs[0], body=inputs[1:])

class RavelOp(OpRegister):
    def init_op(self):
        self.ravel_fmt = '{dtype} {name} = mat(vectorise({inp1}, {dim}).t());' + \
                    self.fmt_base
        self.ravel_fmt_0d = '{dtype} {name} = {{{inp1}}};' + self.fmt_base

    def __repr__(self):
        # currently ravel always along rows ...
        inputs = self.get_sanitized_inputs()
        if self.el.inputs[0].shape == ():
            return self.ravel_fmt_0d.format(dtype=self.get_dtype(),
                                     name=self.name, inp1=inputs[0])
        return self.ravel_fmt.format(dtype=self.get_dtype(),
                                     name=self.name, inp1=inputs[0], dim=1)

class MaxOp(OpRegister):
    def init_op(self):
        self.ravel_fmt = '{dtype} {name} = {inp1}.max();' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.ravel_fmt.format(dtype=self.get_dtype(),
                                     name=self.name, inp1=inputs[0])

class MinOp(OpRegister):
    def init_op(self):
        self.ravel_fmt = '{dtype} {name} = {inp1}.min();' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.ravel_fmt.format(dtype=self.get_dtype(),
                                     name=self.name, inp1=inputs[0])

class ClipOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = clamp({inp1}, {min}, {max});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(),
            name=self.name, inp1=inputs[0], min=inputs[1], max=inputs[2])

class WhereOp(OpRegister):
    """
    The Where Op is currently implemented as ternary operator.
    It does not work as expected with numpy-arrays.

    Todo:
      -  Implement where for matrices.
      -  Implement while loop (need subgraph)
    """
    def init_op(self):
        self.fmt = '{dtype} {name} = {cond} ? {option1} : {option2};' + \
                   self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(),
            name=self.name, cond=inputs[0], option1=inputs[1], option2=inputs[2])

class TransposeOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = {in1}{t};' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(),
                               name=self.name, in1=inputs[0],
                               t='.t()' if len(self.el.output.shape) > 1 else '')

class ArcTan2Op(OpRegister):
    # TODO(wolf): implement atan2 for matrices
    def init_op(self):
        self.fmt = '{dtype} {name} = atan2({inp1}, {inp2});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(),
            name=self.name, inp1=inputs[0], inp2=inputs[1])

class PowOp(OpRegister):
    def init_op(self):
        self.pow_fmt = '{dtype} {name} = pow({inp1}, {exp});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.pow_fmt.format(
            dtype=self.get_dtype(), name=self.name, inp1=inputs[0],
            exp=inputs[1])

class BinaryOp(OpRegister):
    op_map = {
        'Add': '+',
        'Sub': '-',
        'ElemMul': '%%' if config.debug else '%',
        'Mul': '*',
        'Div': '/',
        'Less': '<',
        'LessEqual': '<=',
        'Equal': '==',
        'NotEqual': '!=',
        'Greater': '>',
        'GreaterEqual': '>=',
        'And': '&&',
        'Or': '||',
    }

    def init_op(self):
        self.fmt = '{dtype} {name} = {in1}{t0}{suffix} {op} {in2}{t1};{note}' + \
                   self.fmt_base
        self.inputs = self.el.inputs
        self.elementwise = False
        self.reverse_inputs = False
        self.suffix = ''
        self.t = ['', '']
        # in ardamillo no 1d vectors -> convert shape to 2d shape
        if len(self.inputs[0].shape) == 1:
            arma_shape0 = (self.inputs[0].shape[0], 1)
        else:
            arma_shape0 = self.inputs[0].shape
        if len(self.inputs[1].shape) == 1:
            arma_shape1 = (self.inputs[1].shape[0], 1)
        else:
            arma_shape1 = self.inputs[1].shape
        # specify element-wise operation
        if len(arma_shape0) and len(arma_shape1):
            if arma_shape0 != arma_shape1:
                self.elementwise = True
                # check for np.array([[][]]) x np.array([])
                if len(self.inputs[0].shape) == 1 and len(self.inputs[1].shape) != 1:
                    self.t[0] = '.t()'
                    arma_shape0 = (1, self.inputs[0].shape[0])
                elif len(self.inputs[1].shape) == 1 and len(self.inputs[0].shape) != 1:
                    self.t[1] = '.t()'
                    arma_shape1 = (1, self.inputs[1].shape[0])

                if arma_shape0[0] == arma_shape1[0]:
                    self.suffix = '.eval().each_col()'
                    if arma_shape1[1] > 1:
                        self.reverse_inputs = not self.reverse_inputs
                        self.inputs = list(reversed(self.inputs))
                        self.t = list(reversed(self.t))
                elif arma_shape0[1] == arma_shape1[1]:
                    self.suffix = '.eval().each_row()'
                    if arma_shape1[0] > 1:
                        self.reverse_inputs = not self.reverse_inputs
                        self.inputs = list(reversed(self.inputs))
                        self.t = list(reversed(self.t))

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        operator = self.op_map[self.el.op]
        if self.elementwise:
            if self.reverse_inputs and self.el.op in ('Sub', 'Div'):
                raise NotImplementedError('Non-commutative op.')
            if self.el.op == 'Mul':
                operator = self.op_map['ElemMul']
        return self.fmt.format(dtype=self.get_dtype(),
                               name=self.name,
                               in1=inputs[0],
                               suffix=self.suffix,
                               t0=self.t[0],
                               t1=self.t[1],
                               in2=inputs[1],
                               op=operator,
                               note=' /* reversed */' if self.reverse_inputs else ''
                               )

    @classmethod
    def registers(cls,):
        return cls.op_map.keys()

class DotOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = dot({in1}, {in2});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(),
                               name=self.name,
                               in1=inputs[0],
                               in2=inputs[1])

class MatMulOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = {in1} * {in2};' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(),
                               name=self.name,
                               in1=inputs[0],
                               in2=inputs[1])

class UnaryOp(OpRegister):
    op_map = {
        'Neg': '-',
        'Not': '!',
    }

    def init_op(self):
        self.fmt = '{dtype} {name} = {op}{in1};' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(), name=self.name,
                               in1=inputs[0], op=self.op_map[self.el.op])

    @classmethod
    def registers(cls):
        return cls.op_map.keys()

class AssignOp(OpRegister):
    def init_op(self):
        self.fmt = '{in1} = {in2}{tranpose};' + self.fmt_base
        self.fmt_with_idx = '{in1}({idx}) = {in2};'
        self.at_idx = self.el.at_idx
        self.transpose = ''
        self.slices = self.el.slices
        self.input1 = sanitize_name(self.el.inputs[0].name)
        self.input2 = sanitize_name(self.el.inputs[1].name)
        shape0 = self.el.inputs[0].shape
        shape1 = self.el.inputs[1].shape

        if len(shape0) and len(shape1) and shape0[0] != shape1[0]:
            self.transpose = '.t()'

        if self.el.slices:
            self.slice_tuples = []
            slices = self.el.slices
            input_shape = self.el.inputs[0].shape
            nslices = len(slices)
            slice_ndim = len(input_shape)

            # in ardamillo no 1d vectors -> convert to 2d row vector
            if slice_ndim == 1:
                    input_shape = (input_shape[0], 1)
                    slice_ndim += 1

            # fill up empty slices with ':'' (-> slice[None])
            while nslices < slice_ndim:
                slices.append(slice(None))
                nslices += 1

            # convert to armadillo slices
            for idx in range(nslices):
                if type(slices[idx]) == int:
                    self.slice_tuples.append((slices[idx], slices[idx]))
                else:
                    sl = slices[idx]
                    stp = sl.stop - 1 if sl.stop else input_shape[idx] - 1
                    self.slice_tuples.append((sl.start or 0, stp))

            out_shape = self.el.output.kwargs['slice_shape']
            slice_shape = out_shape if len(out_shape) !=  1 else (out_shape[0], 1)
            arma_shape = (self.slice_tuples[0][1] - self.slice_tuples[0][0] + 1,
                          self.slice_tuples[1][1] - self.slice_tuples[1][0] + 1)

            if slice_shape == arma_shape or (slice_shape == () and arma_shape == (1, 1)):
                self.slice_fmt = 'set_items({lhs}, {rhs}, {{{start}}}, {{{end}}});'
            elif slice_shape == (arma_shape[1], arma_shape[0]):
                self.slice_fmt = 'set_items({lhs}, {rhs}, {{{start}}}, {{{end}}}, true);'
            else:
                raise NotImplementedError("You should not end up here.")

    def __repr__(self):
        inputs = self.get_sanitized_inputs()

        if self.at_idx:
            return self.fmt_with_idx.format(
                name=self.name,
                in1=self.input1,
                idx=', '.join([str(i) for i in self.at_idx]),
                in2=self.input2)

        elif self.slices:
            lhs_eval = get_unique_name(inputs[0] + '_eval')
            rhs_eval = get_unique_name(inputs[1] + '_eval')
            return self.slice_fmt.format(name=self.name,
                                         lhs_eval=lhs_eval,
                                         rhs_eval=rhs_eval,
                                         lhs=inputs[0],
                                         rhs=inputs[1],
                                         start='{}, {}'.format(self.slice_tuples[0][0],
                                                               self.slice_tuples[1][0]),
                                         end='{}, {}'.format(self.slice_tuples[0][1],
                                                             self.slice_tuples[1][1]))

        return self.fmt.format(in1=self.input1,
                               in2=self.input2,
                               tranpose=self.transpose)

class IdentityOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = {in1};' + self.fmt_base

    def __repr__(self):
        input1 = sanitize_name(self.el.inputs[0].name)
        return self.fmt.format(dtype=self.get_dtype(), name=self.name,
                               in1=input1)

class SqueezeOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = {in1};' + self.fmt_base

    def __repr__(self):
        input1 = sanitize_name(self.el.inputs[0].name)
        return self.fmt.format(dtype=self.get_dtype(), name=self.name,
                               in1=input1)

class CrossOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = cross({in1}, {in2});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(), name=self.name,
                               in1=inputs[0], in2=inputs[1])

class CreateArrayOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = {init_val};' + self.fmt_base

    def get_value(self):
        inputs = self.el.inputs
        if self.el.shape != ():
            # convert to initializer list
            return array2string(self.el.nested_input)
        else:
            return inputs[0].value

    def __repr__(self):
        return self.fmt.format(
            dtype=self.get_dtype(),
            name=self.name,
            init_val=self.get_value())

class ModOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = mod({in1}, {in2});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(dtype=self.get_dtype(), name=self.name,
                               in1=inputs[0], in2=inputs[1])

class SolveOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = solve({a}, {b});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(name=self.name, dtype=self.get_dtype(),
                               a=inputs[0], b=inputs[1])

class NormOp(OpRegister):
    def init_op(self):
        # self.fmt = '{dtype} {name} = norm({in1});' + self.fmt_base
        self.fmt = '{dtype} {name} = sqrt(dot({in1}, {in1}));' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(name=self.name, dtype=self.get_dtype(),
                               in1=inputs[0])

class ConcatenateOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = join_{join_type}({in1}, {in2});'

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        if self.el.axis == 0:
            return self.fmt.format(name=self.name, dtype=self.get_dtype(), 
                               in1=inputs[0], in2=inputs[1], join_type='cols') + \
                    self.fmt_base
        elif self.el.axis == 1:
            return self.fmt.format(name=self.name, dtype=self.get_dtype(), 
                               in1=inputs[0], in2=inputs[1], join_type='rows') + \
                    self.fmt_base
        else:
            raise ValueError('You shouldn\'t be here.')

class RandomNormalOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name}; {name} = {in2} {mul_op} {name}.randn() + {in1};' + \
                   self.fmt_base
        self.fmt_single_val = '{dtype} {name}; {name} = {in2} * randn<vec>(1)[0] + {in1};' + \
                              self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        if self.el.output.shape == ():
            return self.fmt_single_val.format(name=self.name, dtype=self.get_dtype(),
                                              in1=inputs[0], in2=inputs[1])

        mul_op = '*' if self.el.inputs[1].shape == () else '%%' if config.debug else '%'
        return self.fmt.format(name=self.name, dtype=self.get_dtype(),
                               in1=inputs[0], in2=inputs[1], mul_op=mul_op)

class ElementWiseOp(OpRegister):
    op_map = {
            'Sin': 'sin',
            'Cos': 'cos',
            'Tan': 'tan',
            'Sqrt': 'sqrt',
            'Square': 'square',
            'Abs': 'abs',
            'Exp': 'exp',
            'Log': 'log',
            'ArcSin': 'asin',
            'ArcCos': 'acos',
            'ArcTan': 'atan'
        }

    def init_op(self):
        self.fmt = '{dtype} {name} = {operator}({in1});' + self.fmt_base

    def __repr__(self):
        input1 = sanitize_name(self.el.inputs[0].name)
        return self.fmt.format(dtype=self.get_dtype(), name=self.name,
                               in1=input1, operator=self.op_map[self.el.op])

    @classmethod
    def registers(cls):
        return cls.op_map.keys()

class NoOp(OpRegister):
    def init_op(self):
        self.fmt = '/* No Op {name} */;' + fmt_base

    def __repr__(self):
        return self.fmt.format(name=self.name)

class ReshapeOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = {input}.eval(); {name}.reshape({new_shape});' + \
                   self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(name=self.name,
                               input=inputs[0],
                               new_shape=', '.join([str(i) for i in self.el.output.shape]),
                               dtype=self.get_dtype())

class ZerosOnesEyeOp(OpRegister):
    op_map = {
        'Ones': 'ones',
        'Zeros': 'zeros',
        'Eye': 'eye'
    }

    def __repr__(self):
        output = self.el.output
        return '{type} {name}; {name}.{op}();'.format(type=self.get_dtype(),
                                                       name=output.name,
                                                       op=self.op_map[self.el.op])

    @classmethod
    def registers(cls):
        return cls.op_map.keys()

class ViewOp(OpRegister):
    def init_op(self):
        self.in1 = self.el.inputs[0]
        self.output = self.el.output
        self.slice_tuples = []

        slices = self.el.slices
        input_shape = self.in1.shape
        nslices = len(slices)
        input_ndim = len(input_shape)

        # in ardamillo no 1d vectors -> convert to 2d row vector
        if input_ndim == 1:
                input_shape = (input_shape[0], 1)
                input_ndim += 1
        # fill up empty slices with ':'' (-> slice[None])
        while nslices < input_ndim:
            slices.append(slice(None))
            nslices += 1
        # convert to armadillo slices
        for idx in range(nslices):
            if type(slices[idx]) == int:
                self.slice_tuples.append((slices[idx], slices[idx]))
            else:
                sl = slices[idx]
                stp = sl.stop - 1 if sl.stop else input_shape[idx] - 1
                self.slice_tuples.append((sl.start or 0, stp))
        out_shape = self.el.output.shape
        np_shape = out_shape if len(out_shape) != 1 else (out_shape[0], 1)
        arma_shape = (self.slice_tuples[0][1] - self.slice_tuples[0][0] + 1,
                      self.slice_tuples[1][1] - self.slice_tuples[1][0] + 1)
        if np_shape == arma_shape:
            self.fmt_block = '{dtype} {name} = {in1}({spans}).eval();' + \
                             self.fmt_base
        elif np_shape == (arma_shape[1], arma_shape[0]):
            self.fmt_block = '{dtype} {name} = {in1}({spans}).t().eval();' + \
                             self.fmt_base
        elif np_shape == () and arma_shape == (1, 1):
            self.fmt_block = '{dtype} {name} = {in1}({spans})[0];' + \
                             self.fmt_base
        else:
            raise NotImplementedError("You shouldn't be here")

    def __repr__(self):
        return self.fmt_block.format(
            dtype=self.get_dtype(),
            name=self.name,
            in1=self.in1.name,
            spans=', '.join('span({}, {})'.format(a, b) for a, b in self.slice_tuples)
        )

class ArrayAccessOp(OpRegister):
    def init_op(self):
        self.fmt = '{dtype} {name} = {in1}.eval()({idx});' + self.fmt_base

    def __repr__(self):
        inputs = self.get_sanitized_inputs()
        return self.fmt.format(
            dtype=self.get_dtype(),
            name=self.name,
            in1=inputs[0],
            idx=', '.join([str(i) for i in self.el.at_idx]))

##########################################################################
#### Helper Functions ####
##########################################################################

def array2string(array, _depth=0):
    """
    Recursively create a initializer list style string from an iterable with
    multiple dimensions.

    Args:
        array (iterable): input iterable which is expected to have elements that
                          can be converted to strings with `str()`.
        _depth (int): variable tracking the current recursion depth
    """
    if hasattr(array, 'name'):
        return array.name
    elif not hasattr(array, '__len__'):
        return float_nsf(array)
    else:
        string = ''
        array_len = len(array)
        for i in range(array_len):
            string += array2string(array[i], _depth=_depth + 1) + ', '
        if (array_len > 1) or (_depth == 0) :
            return '{' + string[0:-2] + '}'
        else:
            return string[0:-2]

def get_type(el, shape=None):
    if shape is None:
        shape = el.shape
    dtype = el.dtype
    if not shape or len(shape) == 0:
        return dtype_map[dtype]
    elif len(shape) == 1:
        return 'Mat<{}>::fixed<{}, 1>'.format(dtype_map[dtype], shape[0])
    else:
        # TODO restrict to 2D
        tmpl = 'Mat<{}>::fixed<{}>'.format(dtype_map[dtype], ', '.join(str(s) for s in
            [s for s in shape]))
        return '{type}'.format(type=tmpl)

def float_nsf(num, precision=17):
    """n-Significant Figures"""
    return ('{0:.%ie}' % (precision - 1)).format(float(num))    
