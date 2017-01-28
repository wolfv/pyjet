#!/usr/bin/env python
import sys, os
from fnmatch import fnmatch
from setuptools import setup
from setuptools.command.install import install


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))

    ignore_files = []
    for line in open(".gitignore"):
        li = line.strip()
        if not li.startswith("#") and li != '':
            ignore_files.append(li)

    # attempt to imitate gitignore
    paths_filtered = [path for path in paths 
            if not any(
                    any(fnmatch(part, ignore) for part in path.split('/'))
                            for ignore in ignore_files)]
    return paths_filtered

long_description = \
"""The JET library offers a simple way to make Python, and especially NumPy code
run faster. This is achieved by transparently converting Python/NumPy operations
to performant C++."""

setup(
    name='Jet',
    version='0.0',
    description='JET, a framework for faster numeric Python',
    long_description=long_description,
    author='Wolf Vollprecht, Orestis Zambounis',
    author_email='w.vollprecht@gmail.com',
    url='https://github.com/wolfv/pyjet/',
    license='MIT',
    install_requires=[
            'numpy',
            'networkx',],
    packages=['jet'],
    scripts=['bin/jet'],
    package_data={
            'jet':
                ['post_install/*',
                'include/*.h',
                '../.gitignore'] + 
                package_files('jet/thirdparty')
            },
)
