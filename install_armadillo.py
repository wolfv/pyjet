from distutils.core import setup
import distutils.spawn as ds

import os, sys

cwd = os.path.dirname(os.path.realpath(__file__))

try:
    ds.spawn(['sed', '-i',
        's/set(ARMA_USE_WRAPPER true)/set(ARMA_USE_WRAPPER false)/',
        cwd + '/jet/thirdparty/armadillo/CMakeLists.txt'])
except ds.DistutilsExecError:
    print("Error while setting ARMA_USE_WRAPPER to false.")
    print("Please try to set ARMA_USE_WRAPPER in the CmakeLists.txt to false.")
    sys.exit(-1)

if ds.find_executable('cmake') is None:
    print("CMake  is required to build JET Armadillo")
    print("Please install cmake version >= 2.6 and re-run setup")
    sys.exit(-1)

arma_path = os.path.join(cwd, 'jet', 'thirdparty', 'armadillo')
os.chdir(arma_path)

print("Configuring JET Armadillo build with CMake.... ")

cmake_args = "-DCMAKE_INSTALL_PREFIX={path}".format(path=os.path.join(arma_path, 'installed'))
try:
    ds.spawn(['cmake', '.'] + cmake_args.split())
except ds.DistutilsExecError:
    print("Error while running cmake")
    print("Please make sure you have all dependencies for Armadillo installed.")
    print("E.g. by running `sudo build-dep libarmadillo`.")
    sys.exit(-1)

try:
    ds.spawn(['make'])
except ds.DistutilsExecError:
    print("Error while running make for Armadillo")
    sys.exit(-1)

try:
    ds.spawn(['make', 'install'])
except ds.DistutilsExecError:
    print("Error while installing armadillo")
    print("We tried to install armadillo to %s" % os.path.join(arma_path, 'installed'))
    sys.exit(-1)
