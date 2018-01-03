"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
from subprocess import call
from distutils.command.build import build
from setuptools.command.install import install
from setuptools import setup, Extension, find_packages
from pip.req import parse_requirements
import numpy

BASEPATH = os.path.dirname(os.path.abspath(__file__))
H2O4GPUPATH = os.path.join(BASEPATH, '../interface_c/')


class H2O4GPUBuild(build):
    """H2O4GPU library compiler"""
    def run(self):
        """Run the compilation"""
        NVCC = os.popen("which nvcc").read() != ""
        CPULIB = 'ch2o4gpu_cpu'
        GPULIB = 'ch2o4gpu_gpu'
        EXT = ".dylib" if os.uname()[0] == "Darwin" else ".so"

        # run original build code
        build.run(self)

        # build H2O4GPU
        cmd = ['make']

        targets = [CPULIB, GPULIB] if NVCC else [CPULIB]
        cmd.extend(targets)

        CPU_LIBPATH = os.path.join(H2O4GPUPATH, CPULIB + EXT)
        GPU_LIBPATH = os.path.join(H2O4GPUPATH, GPULIB + EXT)

        target_files = [CPU_LIBPATH, GPU_LIBPATH] if NVCC else [CPU_LIBPATH]
        message = 'Compiling H2O4GPU CPU and GPU' if NVCC \
            else 'Compiling H2O4GPU CPU only'

        def compile_cpu():
            # compile_cpu CPU version of H2O4GPU
            call(cmd, cwd=H2O4GPUPATH)

        self.execute(compile_cpu, [], message)

        # copy resulting tool to library build folder
        self.mkpath(self.build_lib)
        for target in target_files:
            self.copy_file(target, self.build_lib)


class H2O4GPUInstall(install):
    def initialize_options(self):
        install.initialize_options(self)
        self.build_scripts = None

    def finalize_options(self):
        install.finalize_options(self)
        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))

    def run(self):
        # run original install code
        install.run(self)

        # install H2O4GPU executables
        self.copy_tree(self.build_lib, self.install_lib)

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('../../requirements_runtime.txt', session='hack')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

def get_packages(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory, followlinks=True):
        if './build' in path or './dist' in path or 'h2o4gpu.egg-info' in path or '__pycache__' in path or path == './' or path in paths:
            pass
        else:
            paths.append(path[2:])
    return paths

packages = get_packages('./')

package_data = {}
for package in packages:
   package_data[package] = ['*']

import os
from setuptools import setup
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

# Read version
about_info={}
with open('__about__.py') as f: exec(f.read(), about_info)

swig_extra_compile_args = ["-std=c++11"]
swig_extra_link_args = []

setup(
    name='h2o4gpu',
    version=about_info['__version__'],
    author='H2O.ai, Inc.',
    author_email='h2ostream@googlegroups.com',
    url='http://h2o.ai',
    distclass=BinaryDistribution,
    #platforms=['linux_x86_64'], # scikit-learn: h2o4gpu-0.20.dev0-cp36-cp36m-linux_x86_64.whl
    # from:
    # find -L -type d -printf '%d\t%P\n'| sort -r -nk1| cut -f2-|grep -v pycache
    packages=packages,
    package_data=package_data,
    license='Apache v2.0',
    zip_safe=False,
    description='H2O.ai GPU Edition',
    install_requires=reqs,
    cmdclass={'build': H2O4GPUBuild, 'install': H2O4GPUInstall},
    
    ext_modules=[
    Extension(name='h2o4gpu.util._roc_opt',
              sources=['h2o4gpu/util/roc_opt.i', '../cpu/metrics/metrics.cpp'],
              include_dirs=[numpy.get_include(), '../include/'],
              extra_compile_args=swig_extra_compile_args,
              extra_link_args=swig_extra_link_args,
              swig_opts=["-c++"])
    ]
)
