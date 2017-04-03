import numpy
import jet

def jet_error(attr_name):
    raise NotImplementedError('\'{}\' is a special member of '
            'JET which is not part of Numpy\'s API.'.format(attr_name))

def numpy_error(attr_name):
    raise NotImplementedError('Attribute \'{}\' is available in Numpy but not '
            'in JET yet.'.format(attr_name))

def jet_method(attr):
    def wrapper(*args, **kwargs):
        if jet.jet_mode:
            return attr(*args, **kwargs)
        attr_name = attr.__name__
        np_obj = getattr(numpy, attr_name, None)
        if np_obj is not None:
            return np_obj(*args, **kwargs)
        jet_error(attr_name)
    return wrapper

def numpy_method(np_obj):
    def wrapper(*args, **kwargs):
        if not jet.jet_mode:
            return np_obj(*args, **kwargs)
        attr_name = np_obj.__name__
        numpy_error(attr_name)
    return wrapper

def jet_class_method(cls):
    def decorator(attr):
        def wrapper(*args, **kwargs):
            if jet.jet_mode:
                return attr(*args, **kwargs)
            attr_name = attr.__name__
            np_obj = getattr(getattr(numpy, cls, None), attr_name, None)
            if np_obj is not None:
                return np_obj(*args, **kwargs)
            jet_error(attr_name)
        return wrapper
    return decorator

def jet_static_class(cls):
    for name, attr in cls.__dict__.items():
        if callable(attr):
            setattr(cls, name, staticmethod(jet_class_method(cls.__name__)(attr)))
    return cls
