from setuptools.dist import Distribution
from setuptools import setup
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
from subprocess import call
from distutils.command.build import build
from setuptools.command.install import install

BASEPATH = os.path.dirname(os.path.abspath(__file__))
H2O4GPUPATH = os.path.join(BASEPATH, '../interface_c/')


class H2O4GPUBuild(build):
    """H2O4GPU library compiler"""

    def run(self):
        """Run the compilation"""
        NVCC = os.popen("which nvcc").read() != ""
        CPULIB = '_ch2o4gpu_cpu'
        GPULIB = '_ch2o4gpu_gpu'
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


# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
requirements_file = 'requirements_runtime.txt'
if os.environ.get('CONDA_BUILD_STATE') is not None:
    requirements_file = 'requirements_conda.txt'
with open(requirements_file, "r") as fs:
    reqs = [r for r in fs.read().splitlines() if (
        len(r) > 0 and not r.startswith("#"))]


def get_packages(directory):
    paths = set()
    for (path, directories, filenames) in os.walk(directory, followlinks=True):
        if '.github' in path or './build' in path or './dist' in path or 'h2o4gpu.egg-info' in path or '__pycache__' in path or path == './' or path in paths:
            pass
        else:
            paths.add(path[2:])
    return list(paths)


packages = get_packages('./')

package_data = {}
for package in packages:
    package_data[package] = ['*']


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


# Read version
about_info = {}
with open('__about__.py', 'r', encoding="utf-8") as f:
    exec(f.read(), about_info)

lines = []
lines.append("__version__ = '" +
             about_info['__build_info__']['base_version'] + "'")
lines.append("__git_revision__ = '" +
             about_info['__build_info__']['commit'] + "'")
lines.append("__cuda_version__ = '" +
             about_info['__build_info__']['cuda_version'] + "'")
lines.append("__cuda_nccl__ = '" +
             about_info['__build_info__']['cuda_nccl'] + "'")
with open('build_info.txt', 'w') as fp:
    fp.write('\n'.join(lines)+'\n')

# Make the .whl contain required python and OS as we are version and distro specific
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

setup(
    name='h2o4gpu',
    version=about_info['__version__'],
    author='H2O.ai, Inc.',
    author_email='h2ostream@googlegroups.com',
    url='http://h2o.ai',
    distclass=BinaryDistribution,
    # platforms=['linux_x86_64'], # scikit-learn: h2o4gpu-0.20.dev0-cp36-cp36m-linux_x86_64.whl
    # from:
    # find -L -type d -printf '%d\t%P\n'| sort -r -nk1| cut -f2-|grep -v pycache
    packages=packages,
    package_data=package_data,
    license='Apache v2.0',
    zip_safe=False,
    description='H2O.ai GPU Edition',
    long_description='**H2O4GPU** is a collection of GPU solvers by [H2Oai](https://www.h2o.ai/) with APIs in Python and R. The Python API builds upon the easy-to-use [scikit-learn](http://scikit-learn.org) API and its well-tested CPU-based algorithms.  It can be used as a drop-in replacement for scikit-learn (i.e. `import h2o4gpu as sklearn`) with support for GPUs on selected (and ever-growing) algorithms.  H2O4GPU inherits all the existing scikit-learn algorithms and falls back to CPU algorithms when the GPU algorithm does not support an important existing scikit-learn class option.  The R package is a wrapper around the H2O4GPU Python package, and the interface follows standard R conventions for modeling.',
    long_description_content_type='text/markdown',
    install_requires=reqs,
    cmdclass={'bdist_wheel': bdist_wheel,
              'build': H2O4GPUBuild, 'install': H2O4GPUInstall},
)
