from sys import modules as _modules
import types as _types
import numpy
from jet import config
jet_mode = config.jet_mode
from jet import helpers as _helpers
from jet import intake as _intake
from jet.expander import import_intake as _import_intake, graph;
_import_intake()


# jet module
module = _modules[__name__]

# decorate numpy attributes 
for name, attr in numpy.__dict__.iteritems():
    if not name.startswith("_") and not isinstance(attr, _types.BuiltinFunctionType):
        if callable(attr):
            module.__dict__[name] = _helpers.numpy_mode(attr)
        else:
            module.__dict__[name] = attr

# decorate jet-intake attributes
for name, attr in _intake.__dict__.iteritems():
    if not name.startswith("_") and \
            (callable(attr) or hasattr(attr, '__class__')) and \
            not isinstance(attr, _types.BuiltinFunctionType):
        module.__dict__[name] = _helpers.jet_mode(attr)

def set_options(jet_mode=True,
                debug=False,
                merge=True,
                print_banner=True,
                draw_graph=False,
                draw_graph_raw=False,
                group_class=False,
                group_func=False,
                DTYPE=numpy.float64):

    jet_mode = jet_mode
    config.jet_mode = jet_mode
    config.debug = debug
    config.draw_graph = draw_graph
    config.draw_graph_raw = draw_graph_raw
    config.group_class = group_class
    config.group_func = group_func
    config.merge = merge
    config.print_banner = print_banner

####
# functions to be available in jet mode as well as in pure-python mode
def while_loop(cond, body, args):
    if not jet_mode:
        while(cond(*args)):
            body(*args)
    else:
        raise NotImplementedError()
        op = WhileOp#([cond] + [body(*args, register as while subgraph)])
        return op.get_output()
