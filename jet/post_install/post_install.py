import sys, os
from subprocess import call


cwd = os.path.dirname(os.path.realpath(__file__))
def post_install():
    call([sys.executable, cwd + '/install_armadillo.py'])
    call(['bash', cwd + '/install_dependencies.sh'])

if __name__ == '__main__':
    post_install()
