import sys
import inspect
import numpy
from jet import config

if sys.version_info >= (3, 4):
    import importlib
    reload = importlib.reload
elif sys.version_info[0] > 3:
    import imp
    reload = imp.reload
    


jet_mode = config.jet_mode

class Error(object):
        def __init__(self, name, jet_mode):
            self.name = name
            self.jet_mode = jet_mode
        
        def raise_error_jet(self, *args, **kwargs):
            raise NotImplementedError('\'{}\' is available in Numpy but not in '
                                      'JET yet.'.format(self.name))
        
        def raise_error_numpy(self, *args, **kwargs):
            raise NotImplementedError('\'{}\' is a special member function of '
                    'JET which is not part of Numpy\'s API.'.format(self.name))
        
        raise_error = raise_error_jet if jet_mode else raise_error_numpy

# Not working yet, since it overwrites also some global members of Numpy.
# TODO: find way to detect global members (and do not overwrite them)
def _overload_funcs_deep(members, obj, base_name=None, depth=0):
    '''
    Overlaods functions which are only available in JET but not in Numpy or only
    in Numpy but not in JET with an error function depending on the jet_mode.
    '''
    if depth > 2:
        return
    exceptions = ['Error', 'inspect', 'config', 'sys', 'intake', 'utils', 
                  'expander', 'compressor', 'numpy']
    for name, el in members.items():
        full_name = base_name + '.' + name if base_name else name
        if inspect.isfunction(el):
            attr = el.__module__.split('.')[0]
            if attr == ('numpy' if jet_mode else 'jet') and \
                        not name.startswith('__') and name != '_overload_funcs':
                setattr(obj, name, Error(full_name, jet_mode).raise_error)
        elif (inspect.isclass(el) or inspect.ismodule(el)) and \
                            hasattr(el, '__dict__') and name not in exceptions:
            _overload_funcs_deep(el.__dict__, el, full_name, depth=depth + 1)

# TODO: currently the error function also overwrites classes/modules. In future
# it should detect classes/modules and override its member functions
# See _overload_funcs_deep
def _overload_funcs():
    '''
    Overlaods functions which are only available in JET but not in Numpy or only
    in Numpy but not in JET with an error function depending on the jet_mode.
    '''
    members =  globals()
    for name, el in members.items():
        if hasattr(el, '__call__'):
            attr = el.__module__.split('.')[0] if hasattr(el, '__module__') else \
                   el.__class__.__name__ if hasattr(el, '__class__') else None
            if (attr == ('numpy' if jet_mode else 'jet') or \
                        attr == ('ufunc' if jet_mode else False)) and \
                        name != 'Error':
                    members[name] = Error(name, jet_mode).raise_error

if jet_mode:
    from numpy import *
    from jet.intake import *
    if config.print_banner:
        print(config.BANNER)
    if config.debug:
        print("Config: ")
        print("-------")
        print("JET Mode: %r" % config.jet_mode)
        print("Debug: %r" % config.debug)
        print("Merge Ops: %r" % (config.merge and not config.debug))
        print("Drawing Graph: %r" % config.draw_graph)
        print("Drawing Raw Graph: %r" % config.draw_graph_raw)
        print("Grouping Classes in Drawing: %r" % config.group_class)
        print("Grouping Functions in Drawing: %r" % config.group_func)
else:
    from jet.intake import *
    from numpy import *

_overload_funcs()
# _overload_funcs_deep(globals(), obj=sys.modules[__name__])

def set_options(jet_mode=False,
                debug=False,
                merge=True,
                print_banner=True,
                draw_graph=False,
                draw_graph_raw=False,
                group_class=False,
                group_func=False,
                DTYPE=numpy.float64):

    config.jet_mode = jet_mode
    config.debug = debug
    config.draw_graph = draw_graph
    config.draw_graph_raw = draw_graph_raw
    config.group_class = group_class
    config.group_func = group_func
    config.merge = merge
    config.print_banner = print_banner

    reload(sys.modules[__name__])

# functions to be available in jet mode as well as in pure-python mode
def while_loop(cond, body, args):
    if not jet_mode:
        while(cond(*args)):
            body(*args)
    else:
        raise NotImplementedError()
        op = WhileOp#([cond] + [body(*args, register as while subgraph)])
        return op.get_output()




