#!/usr/bin/env python
import sys, os
from fnmatch import fnmatch
from setuptools import setup
from setuptools.command.install import install
if sys.version_info[0] >= 3:
    from configparser import  ConfigParser
else:
    from ConfigParser import  ConfigParser


config = ConfigParser()
config.read("jet/.metadata")

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
    name='jet',
    version=config.get('Version', 'version'),
    author='Wolf Vollprecht, Orestis Zambounis',
    author_email='w.vollprecht@gmail.com',
    description='JET, a framework for faster numeric Python',
    long_description=long_description,

    url='https://github.com/wolfv/pyjet/',
    license='MIT',
    
    platforms=['Ubuntu'],
    classifiers=[
              'Development Status :: 4 - Beta',
              'Natural Language :: English',
              'Operating System :: Unix',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              'Topic :: Software Development :: Code Generators',
              'Topic :: Software Development :: Libraries :: Python Modules',
              ],

    install_requires=[
            'numpy',
            'networkx'],
    packages=['jet'],
    scripts=['bin/jet'],
    package_data={
            'jet':
                ['post_install/*',
                'include/*.h',
                '.metadata',
                '../.gitignore'] + 
                package_files('jet/thirdparty')
            },
)
