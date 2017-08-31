"""
:copyright: (c) 2017 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
from subprocess import call
from distutils.command.build import build
from setuptools import setup
from setuptools.command.install import install
from pip.req import parse_requirements

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
install_reqs = parse_requirements('../../requirements.txt', session='hack')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='h2o4gpu',
    version='0.0.3',
    author='H2O.ai, Inc.',
    author_email='h2ostream@googlegroups.com',
    url='http://h2o.ai',
    # from:
    # find -L -type d -printf '%d\t%P\n'| sort -r -nk1| cut -f2-|grep -v pycache
    packages=['h2o4gpu',
              'xgboost',
              'py3nvml',
              'h2o4gpu.utils/sparsetools/tests',
              'h2o4gpu.metrics/cluster/tests',
              'h2o4gpu.datasets/tests/data',
              'h2o4gpu.utils/tests',
              'h2o4gpu.utils/sparsetools',
              'h2o4gpu.tree/tests',
              'h2o4gpu.svm/tests',
              'h2o4gpu.semi_supervised/tests',
              'h2o4gpu.preprocessing/tests',
              'h2o4gpu.neural_network/tests',
              'h2o4gpu.neighbors/tests',
              'h2o4gpu.model_selection/tests',
              'h2o4gpu.mixture/tests',
              'h2o4gpu.metrics/tests',
              'h2o4gpu.metrics/cluster',
              'h2o4gpu.manifold/tests',
              'h2o4gpu.linear_model/tests',
              'h2o4gpu.gaussian_process/tests',
              'h2o4gpu.feature_selection/tests',
              'h2o4gpu.feature_extraction/tests',
              'h2o4gpu.externals/joblib',
              'h2o4gpu.ensemble/tests',
              'h2o4gpu.decomposition/tests',
              'h2o4gpu.datasets/tests',
              'h2o4gpu.datasets/images',
              'h2o4gpu.datasets/descr',
              'h2o4gpu.datasets/data',
              'h2o4gpu.cross_decomposition/tests',
              'h2o4gpu.covariance/tests',
              'h2o4gpu.cluster/tests',
              'h2o4gpu.utils',
              'h2o4gpu.util',
              'h2o4gpu.tree',
              'h2o4gpu.tests',
              'h2o4gpu.svm',
              'h2o4gpu.solvers',
              'h2o4gpu.semi_supervised',
              'h2o4gpu.preprocessing',
              'h2o4gpu.neural_network',
              'h2o4gpu.neighbors',
              'h2o4gpu.model_selection',
              'h2o4gpu.mixture',
              'h2o4gpu.metrics',
              'h2o4gpu.manifold',
              'h2o4gpu.linear_model',
              'h2o4gpu.libs',
              'h2o4gpu.gaussian_process',
              'h2o4gpu.feature_selection',
              'h2o4gpu.feature_extraction',
              'h2o4gpu.externals',
              'h2o4gpu.ensemble',
              'h2o4gpu.decomposition',
              'h2o4gpu.datasets',
              'h2o4gpu.cross_decomposition',
              'h2o4gpu.covariance',
              'h2o4gpu.cluster',
              'h2o4gpu.__check_build',
              'h2o4gpu._build_utils'
             ],
    package_data={'h2o4gpu': ['*'],
                  'h2o4gpu.xgboost': ['*'],
                  'h2o4gpu.py3nvml': ['*'],
                  'h2o4gpu.utils/sparsetools/tests': ['*'],
                  'h2o4gpu.metrics/cluster/tests': ['*'],
                  'h2o4gpu.datasets/tests/data': ['*'],
                  'h2o4gpu.utils/tests': ['*'],
                  'h2o4gpu.utils/sparsetools': ['*'],
                  'h2o4gpu.tree/tests': ['*'],
                  'h2o4gpu.svm/tests': ['*'],
                  'h2o4gpu.semi_supervised/tests': ['*'],
                  'h2o4gpu.preprocessing/tests': ['*'],
                  'h2o4gpu.neural_network/tests': ['*'],
                  'h2o4gpu.neighbors/tests': ['*'],
                  'h2o4gpu.model_selection/tests': ['*'],
                  'h2o4gpu.mixture/tests': ['*'],
                  'h2o4gpu.metrics/tests': ['*'],
                  'h2o4gpu.metrics/cluster': ['*'],
                  'h2o4gpu.manifold/tests': ['*'],
                  'h2o4gpu.linear_model/tests': ['*'],
                  'h2o4gpu.gaussian_process/tests': ['*'],
                  'h2o4gpu.feature_selection/tests': ['*'],
                  'h2o4gpu.feature_extraction/tests': ['*'],
                  'h2o4gpu.externals/joblib': ['*'],
                  'h2o4gpu.ensemble/tests': ['*'],
                  'h2o4gpu.decomposition/tests': ['*'],
                  'h2o4gpu.datasets/tests': ['*'],
                  'h2o4gpu.datasets/images': ['*'],
                  'h2o4gpu.datasets/descr': ['*'],
                  'h2o4gpu.datasets/data': ['*'],
                  'h2o4gpu.cross_decomposition/tests': ['*'],
                  'h2o4gpu.covariance/tests': ['*'],
                  'h2o4gpu.cluster/tests': ['*'],
                  'h2o4gpu.utils': ['*'],
                  'h2o4gpu.util': ['*'],
                  'h2o4gpu.tree': ['*'],
                  'h2o4gpu.tests': ['*'],
                  'h2o4gpu.svm': ['*'],
                  'h2o4gpu.solvers': ['*'],
                  'h2o4gpu.semi_supervised': ['*'],
                  'h2o4gpu.preprocessing': ['*'],
                  'h2o4gpu.neural_network': ['*'],
                  'h2o4gpu.neighbors': ['*'],
                  'h2o4gpu.model_selection': ['*'],
                  'h2o4gpu.mixture': ['*'],
                  'h2o4gpu.metrics': ['*'],
                  'h2o4gpu.manifold': ['*'],
                  'h2o4gpu.linear_model': ['*'],
                  'h2o4gpu.libs': ['*'],
                  'h2o4gpu.gaussian_process': ['*'],
                  'h2o4gpu.feature_selection': ['*'],
                  'h2o4gpu.feature_extraction': ['*'],
                  'h2o4gpu.externals': ['*'],
                  'h2o4gpu.ensemble': ['*'],
                  'h2o4gpu.decomposition': ['*'],
                  'h2o4gpu.datasets': ['*'],
                  'h2o4gpu.cross_decomposition': ['*'],
                  'h2o4gpu.covariance': ['*'],
                  'h2o4gpu.cluster': ['*'],
                  'h2o4gpu.__check_build': ['*'],
                  'h2o4gpu._build_utils': ['*']
                 },
    license='Apache v2.0',
    zip_safe=False,
    description='H2O.ai GPU Edition',
    install_requires=reqs,
    cmdclass={'build': H2O4GPUBuild, 'install': H2O4GPUInstall}
)
