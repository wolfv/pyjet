import inspect
import sys

if sys.version_info[0] >= 3:
    import functools
    reduce = functools.reduce

# Below are functions related to generating unique names. Every graph object has
# to have a unique name, which is re-used in the generated C++ Output
registered_name_generators = {} # TODO dominique: use collections.defaultdict for this

def get_unique_name(name):
    if registered_name_generators.get(name):
        return next(registered_name_generators[name])
    else:
        registered_name_generators[name] = unique_name_generator(name)
        return next(registered_name_generators[name])

# function taken from sympy
def unique_name_generator(prefix='x', start=0, exclude=[]):
    """
    Generate an infinite stream of names consisting of a prefix and
    increasing subscripts provided that they do not occur in `exclude`.

    Args:
        prefix: str, optional
            The prefix to use. By default, this function will generate symbols of
            the form "x0", "x1", etc.
        start: int, optional
            The start number.  By default, it is 0.

    Returns:
        name: str
            The subscripted, unique name.
    """
    exclude = set(exclude or [])

    # return first without suffix -> for declared names like
    # function parameters
    if start == 0:
        yield prefix

    while True:
        name = '%s_%s' % (prefix, start)
        if name not in exclude:
            yield name
        start += 1

def beauty(text):
    import subprocess
    try:
        p = subprocess.Popen(['clang-format-3.6'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
        stdout, stderr = p.communicate(input=text)
        return stdout
    except:
        pass
    try:
        p = subprocess.Popen(['clang-format'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
        stdout, stderr = p.communicate(input=text)
        return stdout
    except:
        return text

def slice_to_str(slice_obj):
    if not isinstance(slice_obj, slice):
        return str(slice_obj)
    elif slice_obj.step:
        return '{}:{}:{}'.format(slice_obj.start, slice_obj.step, slice_obj.stop)
    else:
        return '{}:{}'.format(slice_obj.start, slice_obj.stop)

def sanitize_name(name):
    return name.split(':')[0].replace('/', '_', ).replace('.', '_').replace('-', '_').replace('+', '_').replace('*', '_').replace('<', '_').replace('>', '_')

def get_caller_info(*exclude_list):
        exclude_list = list(exclude_list) + ['utils.py']
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe)
        i = 0
        while i + 1 < len(calframe) and reduce(lambda bool_1st, bool_2nd: bool_1st or bool_2nd, map(lambda name: (len(calframe[i][1]) >= len(name) and calframe[i][1][-len(name):] == name) , exclude_list)):
            i += 1
        class_name = get_class_from_frame(calframe[i][0])
        line = 'line {}'.format(calframe[i][2])
        file_path_full = calframe[i][1]
        if class_name:
            return class_name[0], file_path_full.split('/')[-1].split('.')[0], calframe[i][3], line
        else:
            return file_path_full, file_path_full.split('/')[-1].split('.')[0], \
                   calframe[i][3], line

def get_class_from_frame(fr):
    args, _, _, value_dict = inspect.getargvalues(fr)
    # we check the first parameter for the frame function is named 'self'
    if len(args) and args[0] == 'self':
        # in that case, 'self' will be referenced in value_dict
        instance = value_dict.get('self', None)
        if instance:
            # return its class
            class_name_full = str(getattr(instance, '__class__', ''))
            return class_name_full, class_name_full.split('.')[-1]
    # return None otherwise
    instance = value_dict.get('__module__', None)
    if instance:
        # return its class
        module_name_full = instance
        return module_name_full, module_name_full.split('.')[-1]
    return None
