import sys
import os
import importlib
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dirname, 'thirdparty/cppimport/'))
sys.path.insert(0, os.path.join(dirname, 'thirdparty/pybind11/'))
from cppimport.import_hook import build_plugin


def compile_cpp(src_code, name, force_build=False):
    """
    Compile source code to a python module

    Args:
      src_code (str)     The source code
      name (str)         Name of python module
      force_build (str)  Force rebuild instead of using a previous build

    Returns:
      Compiled and imported Python module
    """
    if not os.path.exists(os.path.dirname('jet_generated/')):
        try:
            os.makedirs(os.path.dirname('jet_generated/'))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    elif not force_build:
        try:
            with open('jet_generated/' + name + '.cpp', 'r') as fi:
                sorted_fi_lines = set(fi.read().splitlines())
            sorted_code = set(src_code.splitlines())
            if sorted_fi_lines == sorted_code:
                return importlib.import_module('jet_generated.' + name)
        except:
            pass

    with open('jet_generated/' + name + '.cpp', 'w+') as fo: # TODO dominique: I think the mode should be just 'w'
        fo.write(src_code)

    with open('jet_generated/__init__.py', 'w+') as fo: # TODO dominique: I think the mode should be just 'w'
        pass

    build_plugin(name, 'jet_generated/' + name + '.cpp')

    return importlib.import_module('jet_generated.' + name)

def import_cpp(name):
    """
    Import jet generated cpp as module

    Args:
      name (str)  The name (corresponding to the compile_cpp name given)

    Returns:
      Compiled and imported Python module
    """
    try:
        return importlib.import_module('jet_generated.' + name)
    except ImportError:
        raise ImportError("The module %s doesn't exist. Has it been built?" % name)
