import inspect
from jet.compressor import JetBuilder
from jet.utils import sanitize_name, get_caller_info
from jet.intake import placeholder
import jet
from jet.utils import get_unique_name


_func_cached_dict = {}

def jit(*shapes):
    def decorator(func):
        _func_cached_dict[id(func)] = {'func': None, 'shapes': shapes}

        def wrapper(*args):
            if not jet.jet_mode:
                return func(*args)

            func_id = id(func)
            func_cached = _func_cached_dict[func_id]['func']
            if func_cached is not None:
                return func_cached(*args)

            shapes = _func_cached_dict[func_id]['shapes']

            if inspect.ismethod(func):
                arg_names = func.__code__.co_varnames[1:func.__code__.co_argcount]
            else:
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]

            if len(arg_names) != len(args):
                assert(len(arg_names) == 0)
                arg_names = [get_unique_name('ph') for each in args]

            if len(shapes) != len(arg_names) and shapes:
                raise ValueError('Shapes length does not match the arguments length.')

            if not shapes:
                shapes = [arg.shape if hasattr(arg, 'shape') else () for arg in args]
                _func_cached_dict[func_id]['shapes'] = shapes

            ph = [placeholder(name=arg[1], shape=shapes[arg[0]]) for arg in enumerate(arg_names)]
            fun_name = func.__code__.co_name
            if fun_name == '<lambda>':
                fun_name = get_unique_name('lambda')

            jb = JetBuilder(args=ph, out=func(*ph),
                    file_name=get_unique_name(sanitize_name('{}_{}_{func_name}'.format(
                            *get_caller_info('jit.py')[1:-1],
                            func_name=fun_name))),
                    fun_name=get_unique_name(fun_name))
            
            jet_class = getattr(jb.build(), jb.class_name)
            jet_func = getattr(jet_class(), jb.fun_name)
            _func_cached_dict[func_id]['func'] = jet_func

            return jet_func(*args)
        return wrapper

    if shapes and callable(shapes[0]):
        func = shapes[0]
        shapes = ()
        return decorator(func)
    return decorator

if __name__ == "__main__":
    import numpy
    
    @jit((2,), ())
    def test_func(a, b):
        return a + b

    @jit()
    def test_func2(a, b):
        return a - b

    def sub_func(a):
        return a * 2

    @jit
    def test_func3(a, b):
        return (sub_func(a) * b, b)

    b = 1.0
    print(test_func(numpy.array([1, 2]), b))
    print(test_func2(numpy.array([1, 2]), b))
    print(test_func3(numpy.array([1, 2]), b))
