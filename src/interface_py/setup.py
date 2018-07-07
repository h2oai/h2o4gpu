"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# If you want to build the python module directly by setuptools, you can
# specify CMake build flags by environment variables, including
#
# CMAKE flags:
#
#   USE_CUDA
#     Build with GPU acceleration
#
#   BUILD_TESTS
#     Build cpp tests, not related to python tests.
#
#   USE_SYSTEM_GTEST
#     Use system google tests
#
#   DEV_BUILD
#     Use compute_61
#
#   CMAKE_BUILD_TYPE
#     Choose from RelWithDebInfo, Release, Debug, default is Release.
#
#   GPU_COMPUTE_VER
#     Semicolon separated list of compute versions to be built against,
#     e.g. -DGPU_COMPUTE_VER='35;61'
#
#
# If you want to reuse your pre-built cpp modules, you can specify the
# environment variable:
#
#   BINARY_DIR
#     It should be the path to your cmake build directory.
#
#
# Miscellaneous
#
#   TEST_SK
#     Test scikit-learn during build. Enabled when set to  ON or TRUE or 1.

import os
import subprocess
from multiprocessing import cpu_count
from distutils.command.build import build
from distutils.command.clean import clean
from setuptools.command.install import install
from setuptools import setup
from setuptools.dist import Distribution
import shutil
from sklearn_transformer import SkTransformer, SkCache

BASEPATH = os.path.dirname(os.path.abspath(__file__))
BINARIES = ['_ch2o4gpu_gpu.so', '_ch2o4gpu_cpu.so']


class EnvFlags(object):
    def __init__(self):
        self.flags = {
            # CMake
            'USE_CUDA': 'ON',
            'GPU_COMPUTE_VER': '',
            'BUILD_TESTS': 'OFF',
            'USE_SYSTEM_GTEST': 'ON',
            'DEV_BUILD': 'OFF',
            'CMAKE_BUILD_TYPE': 'Release',
            # Others
            'TEST_SK': 'OFF',
            'BINARY_DIR': None}
        for k, default in self.flags.items():
            value = self.check(k)
            setattr(self, k, value)

    def check(self, flag):
        try:
            value = os.getenv(flag).upper()
        except AttributeError:
            return self.flags[flag]
        return value

    def get_cmake_flags(self):
        '''Get the build flags for cmake and value read from `BINARY_DIR`'''

        def _flag(key):
            return '-D' + key + '=' + getattr(self, key)

        cmake_flags = [_flag('USE_CUDA'),
                       _flag('BUILD_TESTS'),
                       _flag('USE_SYSTEM_GTEST'),
                       _flag('DEV_BUILD'),
                       _flag('CMAKE_BUILD_TYPE'),
                       _flag('GPU_COMPUTE_VER')]

        path = os.getenv('BINARY_DIR')
        return cmake_flags, path


class PreBuild(object):
    def __init__(self):
        self._about_info = {}

    @property
    def about_info(self):
        with open('__about__.py') as f:
            exec(f.read(), self._about_info)
        return self._about_info


class H2O4GPUBuild(build):
    '''H2O4GPU library compiler'''

    build_buffer = os.path.join(BASEPATH, './build_lib')

    def _get_prebuilds(self, path):
        '''Return list of paths for found pre-build binaries.

        Parameters
        ---------
        path: str
           The directory to search.
        '''
        result = []
        if not os.path.exists(path):
            print(path, 'doesn\'t exists.')
            exit(1)
        for root, dirs, files in os.walk(path):
            for name in files:
                if name in BINARIES:
                    result.append(os.path.join(root, name))
        if len(result) == 0:
            print('No binaries is found in', path)
            exit(1)
        return result

    def _copy_binaries(self, paths):
        for p in paths:
            self.copy_file(p, os.path.join(self.build_lib, 'h2o4gpu'))

    def _cmake_build(self, cmake_args):
        cmake_cmd = ['cmake']
        cmd = cmake_cmd + cmake_args
        cmd.append('../../../')
        print('CMake command:', cmd)

        def configure():
            subprocess.call(cmd)

        def compile():
            subprocess.call(['make', '-j' + str(cpu_count())])

        if not os.path.exists(self.build_buffer):
            os.makedirs(self.build_buffer)
        os.chdir(self.build_buffer)
        self.execute(configure, [])
        self.execute(compile, [], 'Compiling H2O4GPU cpp code.')
        os.chdir('..')
        binaries_path = [os.path.append(self.build_buffer,
                                        'src/swig/_ch2o4gpu_cpu.so'),
                         os.path.append(self.build_buffer,
                                        'src/swig/_ch2o4gpu_gpu.so')]
        self._copy_binaries(binaries_path)

    def run(self):
        '''Run the compilation'''

        with SkCache(self.build_buffer) as cache:
            SkTransformer(cache=cache).transform(self.build_buffer)

        env_flags = EnvFlags()

        # build scikit-learn
        if env_flags.TEST_SK in ['ON', 'TRUE', '1']:
            sk_target = 'all'
        else:
            sk_target = 'inplace'
        self.execute(
            lambda: subprocess.call(['make', sk_target],
                                    cwd=os.path.join(self.build_buffer,
                                                     'h2o4gpu')), [])

        # Merge scikit-learn with h2o4gpu
        SkTransformer.append_imports(
            os.path.join(self.build_buffer, 'h2o4gpu'))
        SkTransformer.merge_dir(
            os.path.join(self.build_buffer, 'h2o4gpu/h2o4gpu'),
            os.path.join(self.build_lib, 'h2o4gpu'))
        SkTransformer.merge_sklearn_init(self.build_lib)

        # run original build code
        build.run(self)

        SkTransformer.merge_sklearn_init(self.build_lib)

        # build H2O4GPU
        cmake_args, prebuild_dir = env_flags.get_cmake_flags()

        if prebuild_dir is not None:
            print('Try to find binaries from:', prebuild_dir)
            binaries_path = self._get_prebuilds(prebuild_dir)
            self._copy_binaries(binaries_path)
        else:
            self._cmake_build(cmake_args)


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


class H2O4GPUClean(clean):
    def run(self):
        clean.run(self)
        build_lib = os.path.join(BASEPATH, './build_lib')
        if os.path.exists(build_lib):
            shutil.rmtree(build_lib)
        xgb = os.path.join(BASEPATH, './xgboost')
        if os.path.exists(xgb):
            os.remove(xgb)
        py3nvml = os.path.join(BASEPATH, './py3nvml')
        if os.path.exists(py3nvml):
            os.remove(py3nvml)


# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
with open("requirements_runtime.txt", "r") as fs:
    reqs = [r for r in fs.read().splitlines()
            if (len(r) > 0 and not r.startswith("#"))]


def get_packages(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory, followlinks=True):
        if ('./build' in path
            or './dist' in path
            or 'h2o4gpu.egg-info' in path
            or '__pycache__' in path
            or path == './'
                or path in paths):
            pass
        else:
            paths.append(path[2:])
    return paths


packages = get_packages('./')

package_data = {}
for package in packages:
    package_data[package] = ['*']


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


# Make the .whl contain required python and OS as we are version and distro
# specific
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    print('bdist_wheel == None')
    bdist_wheel = None

setup(
    name='h2o4gpu',
    version=PreBuild().about_info['__version__'],
    author='H2O.ai, Inc.',
    author_email='h2ostream@googlegroups.com',
    url='http://h2o.ai',
    distclass=BinaryDistribution,
    # scikit-learn: h2o4gpu-0.20.dev0-cp36-cp36m-linux_x86_64.whl
    # platforms=['linux_x86_64'],
    # from:
    # find -L -type d -printf '%d\t%P\n'| sort -r -nk1| cut -f2-|grep -v pycache
    packages=packages,
    package_data=package_data,
    license='Apache v2.0',
    zip_safe=False,
    description='H2O.ai GPU Edition',
    install_requires=['pandas', 'numpy', 'dateutils', 'pytz', 'scipy',
                      'tabulate', 'future', 'scikit-learn', 'xgboost',
                      'py3nvml'],
    tests_require=['pytest', 'pytest-xdist', 'pytest-cov', 'pylint'],
    cmdclass={'bdist_wheel': bdist_wheel,
              'build': H2O4GPUBuild,
              'install': H2O4GPUInstall,
              'clean': H2O4GPUClean})
