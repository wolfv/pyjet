import numpy
from jet import jet_mode
import jet.intake as intake

def jet_error(attr_name):
    raise NotImplementedError('\'{}\' is a special member of '
            'JET which is not part of Numpy\'s API.'.format(attr_name))


def numpy_error(attr_name):
    raise NotImplementedError('Attribute \'{}\' is available in Numpy but not '
            'in JET yet.'.format(attr_name))

def jet_mode(attr):
    if jet_mode:
        return attr
    attr_name = attr.__name__
    np_obj = getattr(numpy, attr_name, None)
    if np_obj is not None:
        return np_obj
    return lambda *args, **kwargs: jet_error(attr_name)

def numpy_mode(np_obj):
    if not jet_mode:
        return np_obj
    attr_name = np_obj.__name__
    attr = getattr(intake, attr_name, None)
    return lambda *args, **kwargs: numpy_error(attr_name)
