from jet.compressor import JetBuilder
from jet.utils import sanitize_name, get_caller_info
import jet


func_cached_dict = {}

def jit(*shapes):
    def jit_core(func):
        if (jet.jet_mode):
            func_cached = func_cached_dict.get(id(func))
            if not func_cached:            
                arg_names = func.__code__.co_varnames[0:func.__code__.co_argcount]

                if len(shapes) != len(arg_names) and len(shapes) != 0:
                    raise ValueError('Shapes length does not match the arguments length.')

                ph = map(lambda (idx, name): jet.placeholder(name=name, shape=shapes[idx] if shapes else ()), enumerate(arg_names))
                jb = JetBuilder(out=[func(*ph)], file_name=sanitize_name('{}_{}_{func_name}'.format(*get_caller_info('jit.py')[1:-1], func_name=func.__code__.co_name)))

                func_cached = jb.build().JetClass().func

            return func_cached
        else:
            return func
    return jit_core

