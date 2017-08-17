import os
from sys import platform
from setuptools import setup
from setuptools.command.install import install
from distutils.command.build import build
from subprocess import call
from multiprocessing import cpu_count

BASEPATH = os.path.dirname(os.path.abspath(__file__))
H2OGPUMLPATH = os.path.join(BASEPATH, '../interface_c/')


class H2OGPUMLBuild(build):
    def run(self):
        NVCC = os.popen("which nvcc").read() != ""
        CPULIB = 'ch2ogpuml_cpu'
        GPULIB = 'ch2ogpuml_gpu'
        EXT = ".dylib" if os.uname()[0] == "Darwin" else ".so"

        # run original build code
        build.run(self)

        # build H2OGPUML
        cmd = ['make']

        targets = [CPULIB, GPULIB] if NVCC else [CPULIB]
        cmd.extend(targets)

        CPU_LIBPATH = os.path.join(H2OGPUMLPATH, CPULIB + EXT)
        GPU_LIBPATH = os.path.join(H2OGPUMLPATH, GPULIB + EXT)

        target_files = [CPU_LIBPATH, GPU_LIBPATH] if NVCC else [CPU_LIBPATH]
        message = 'Compiling H2OGPUML---CPU and GPU' if NVCC else 'Compiling H2OGPUML---CPU only'

        def compile():
            # compile CPU version of H2OGPUML
            call(cmd, cwd=H2OGPUMLPATH)

        self.execute(compile, [], message)

        # copy resulting tool to library build folder
        self.mkpath(self.build_lib)
        for target in target_files:
            self.copy_file(target, self.build_lib)


class H2OGPUMLInstall(install):
    def initialize_options(self):
        install.initialize_options(self)
        self.build_scripts = None

    def finalize_options(self):
        install.finalize_options(self)
        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))

    def run(self):
        # run original install code
        install.run(self)

        # install H2OGPUML executables
        self.copy_tree(self.build_lib, self.install_lib)


setup(
    name='h2ogpuml',
    version='0.0.3',
    author='H2O.ai, Inc.',
    author_email='h2ostream@googlegroups.com',
    url='http://h2o.ai',
    package_dir={'interface_py': 'h2ogpuml','interface_py': 'xgboost'},
    package_data={'xgboost': ['*']},
    packages=['h2ogpuml',
              'h2ogpuml.libs',
              'h2ogpuml.solvers',
	          'h2ogpuml.util',
              'xgboost'
    ],
    license='Apache v2.0',
    zip_safe=False,
    description='H2O.ai Generalized Linear Modeling with Proximal Operator Graph Solver',
    install_requires=["numpy >= 1.8",
                      "scipy >= 0.13"],
    cmdclass={'build': H2OGPUMLBuild, 'install': H2OGPUMLInstall}
)
