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
