#!/usr/bin/env python
import os, sys, glob

from setuptools import setup
from distutils.command.install import install as _install

# copied from https://stackoverflow.com/questions/17806485/execute-a-python-script-post-install-using-distutils-setuptools
def _post_install(dir):
    from subprocess import call
    call(['pip', 'install', 'pygraphviz',
        '--install-option="--include-path=/usr/include/graphviz"',
        '--install-option="--library-path=/usr/lib/graphviz/"',])
    call([sys.executable, 'install_armadillo.py'])

class install(_install):
    def run(self):
        _install.run(self)
        self.execute(_post_install, (self.install_lib,),
                     msg="Running post install task")
long_description = \
"""The JET library offers a simple way to make Python, and especially NumPy code
run faster. This is achieved by transparently converting Python/NumPy operations
to performant C++."""

setup(
    name='Jet',
    version='1.0',
    description='JET, a framework for faster numeric Python',
    long_description=long_description,
    author='Wolf Vollprecht, Orestis Zambounis',
    author_email='w.vollprecht@gmail.com',
    url='https://github.com/wolfv/pyjet/',
    license='MIT',
    install_requires=[
        "numpy",
        "networkx",
    ],
    packages=['jet'],
    package_data={
            'jet': ['include/*.h', 
            'thirdparty/armadillo/installed/**/**/**/*', 
            'thirdparty/armadillo/installed/**/**/*', 
            'thirdparty/armadillo/installed/**/*', 
            'thirdparty/armadillo/installed/*', 
            'thirdparty/cppimport/**/*',
            'thirdparty/cppimport/*',
            'thirdparty/pybind11/**/**/*'
            'thirdparty/pybind11/**/*'
            'thirdparty/pybind11/*'
            'thirdparty/pybind11/pybind11/*'
       ]},
    cmdclass={'install': install},
)
