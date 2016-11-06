#!/usr/bin/env python
import os, sys, glob

from distutils.core import setup
from distutils.command.install import install as _install

# copied from https://stackoverflow.com/questions/17806485/execute-a-python-script-post-install-using-distutils-setuptools
def _post_install(dir):
    from subprocess import call
    call([sys.executable, 'install_armadillo.py'])

class install(_install):
    def run(self):
        _install.run(self)
        self.execute(_post_install, (self.install_lib,),
                     msg="Running post install task")

setup(name='Jet',
      version='1.0',
      description='JET, a framework for faster numeric Python',
      author_email='w.vollprecht@gmail.com',
      install_requires=[
        "numpy",
        "networkx",
        "pygraphviz"
      ],
      packages=['jet'],
      package_data={'jet': ['include/*.h', 
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
