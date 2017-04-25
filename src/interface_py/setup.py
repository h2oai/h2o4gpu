
import os
from sys import platform
from setuptools import setup
from setuptools.command.install import install
from distutils.command.build import build
from subprocess import call
from multiprocessing import cpu_count

BASEPATH = os.path.dirname(os.path.abspath(__file__))
POGSPATH = os.path.join(BASEPATH, '../interface_c/')

class PogsBuild(build):
  def run(self):
    NVCC = os.popen("which nvcc").read()!=""
    CPULIB='cpogs_cpu'
    GPULIB='cpogs_gpu'
    EXT=".dylib" if os.uname()[0] == "Darwin" else ".so"

    # run original build code
    build.run(self)

    # build POGS
    cmd = [ 'make' ]

    targets= [ CPULIB, GPULIB ] if NVCC else [ CPULIB ]
    cmd.extend(targets)

    CPU_LIBPATH = os.path.join(POGSPATH, CPULIB + EXT)
    GPU_LIBPATH = os.path.join(POGSPATH, GPULIB + EXT)

    target_files = [ CPU_LIBPATH, GPU_LIBPATH ] if NVCC else [ CPU_LIBPATH ]
    message = 'Compiling POGS---CPU and GPU' if NVCC else 'Compiling POGS---CPU only'
 
    def compile():
      # compile CPU version of POGS
      call(cmd, cwd=POGSPATH)

    self.execute(compile, [], message)

    # copy resulting tool to library build folder
    self.mkpath(self.build_lib)
    for target in target_files:
          self.copy_file(target, self.build_lib)


class PogsInstall(install):
    def initialize_options(self):
        install.initialize_options(self)
        self.build_scripts = None

    def finalize_options(self):
        install.finalize_options(self)
        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))

    def run(self):
        # run original install code
        install.run(self)

        # install POGS executables
        self.copy_tree(self.build_lib, self.install_lib)

setup(
    name='pogs',
    version='0.0.1',
    author='H2O.ai, Chris Fougner, Baris Ungun, Stephen Boyd',
    author_email='fougner@stanford.edu, ungun@stanford.edu, boyd@stanford.edu',
    url='http://github.com/h2oai',
    package_dir={'interface_py': 'pogs'},
    packages=['pogs',
              'pogs.libs',
              'pogs.solvers'],
    license='GPLv3',
    zip_safe=False,
    description='Proximal Operator Graph Solver---Python Interface',
    install_requires=["numpy >= 1.8",
                      "scipy >= 0.13"],
    cmdclass={'build' : PogsBuild, 'install' : PogsInstall}
)
