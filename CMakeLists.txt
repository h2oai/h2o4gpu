CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(h2o4gpu)
INCLUDE(cmake/Utils.cmake)

SET_DEFAULT_CONFIGURATION_RELEASE()

FIND_PACKAGE(OpenMP)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(SWIG REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED) # SWIG

INCLUDE(${SWIG_USE_FILE})

# Options
OPTION(USE_CUDA  "Build with GPU acceleration" ON)
OPTION(RELEASE  "Release build" OFF)

SET(CUDA_BUILD_VERSION 8.0)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(RELEASE)
        SET(GPU_COMPUTE_VER 35;50;52;60;61 CACHE STRING
        "Space separated list of compute versions to be built against")
else()
        SET(GPU_COMPUTE_VER 61 CACHE STRING
        "Space separated list of compute versions to be built against")
endif()

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# TODO probably to be removed after POGS is out in favor of XGboost GLM
ADD_DEFINITIONS(
        -D_GITHASH_=0
        -DH2O4GPU_DOUBLE
        -DH2O4GPU_SINGLE
)

# BUILD COMMON CPU/GPU CODE
FILE(GLOB_RECURSE COMMON_SOURCES
        src/common/*.cpp
        src/common/*.h
        )

INCLUDE_DIRECTORIES (
        src/include
        src/cpu/include
        src/gpu/include
        ${PYTHON_INCLUDE_PATH}
)

ADD_LIBRARY(commonh2o4gpu OBJECT ${COMMON_SOURCES})

# BUILD CPU LIBRARY
FILE(GLOB_RECURSE CPU_SOURCES
        src/cpu/*.cpp
        src/cpu/*.h
        )

# TODO should be SHARED??
ADD_LIBRARY(cpuh2o4gpu ${CPU_SOURCES} $<TARGET_OBJECTS:commonh2o4gpu>)

### SWIG

SET(CMAKE_SWIG_FLAGS "")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../src/interface_c/")
SET(CMAKE_SWIG_OUTDIR "../src/interface_py/h2o4gpu/libs")
SET(SWIG_OUTFILE_DIR .)
SET_SOURCE_FILES_PROPERTIES(src/swig/*.i PROPERTIES CPLUSPLUS ON)

# Add swig module
if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
        SWIG_ADD_MODULE(ch2o4gpu_cpu python src/swig/ch2o4gpu_cpu.i)
else()
        SWIG_ADD_LIBRARY(ch2o4gpu_cpu LANGUAGE python SOURCES src/swig/ch2o4gpu_cpu.i)
endif()

SWIG_LINK_LIBRARIES(ch2o4gpu_cpu cpuh2o4gpu ${PYTHON_LIBRARIES})

SET_TARGET_PROPERTIES(${SWIG_MODULE_ch2o4gpu_cpu_REAL_NAME} PROPERTIES
        LINK_FLAGS ${OpenMP_CXX_FLAGS})

### SWIG

if(USE_CUDA)
        # BUILD GPU LIBRARY
        FIND_PACKAGE(CUDA ${CUDA_BUILD_VERSION} REQUIRED)

        ADD_DEFINITIONS(
                -DCUDA_MAJOR=${CUDA_VERSION_MAJOR}
                -DHAVECUDA
        )

        SET(GENCODE_FLAGS "")
        FORMAT_GENCODE_FLAGS("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -std=c++11;--expt-extended-lambda;--expt-relaxed-constexpr;${GENCODE_FLAGS};-lineinfo;")

        if(((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9)) AND RELEASE)
                MESSAGE("CUDA 9.0 detected, adding Volta compute capability (7.0).")
                SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};70")
        endif()

        FILE(GLOB_RECURSE GPU_SOURCES
                src/*.cu
                src/*.cuh
                src/common/*.cpp
                src/common/*.h
                )

        CUDA_ADD_LIBRARY(cuda_static ${GPU_SOURCES} STATIC)

        TARGET_LINK_LIBRARIES(cuda_static ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})

        ADD_LIBRARY(gpuh2o4gpu SHARED $<TARGET_OBJECTS:commonh2o4gpu>)
        TARGET_LINK_LIBRARIES(gpuh2o4gpu ${BLAS_LIBRARIES} cuda_static)

        if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
                SWIG_ADD_MODULE(ch2o4gpu_gpu python src/swig/ch2o4gpu_gpu.i)
        else()
                SWIG_ADD_LIBRARY(ch2o4gpu_gpu LANGUAGE python SOURCES src/swig/ch2o4gpu_gpu.i)
        endif()
        SWIG_LINK_LIBRARIES(ch2o4gpu_gpu gpuh2o4gpu ${PYTHON_LIBRARIES})

        SET_TARGET_PROPERTIES(${SWIG_MODULE_ch2o4gpu_gpu_REAL_NAME} PROPERTIES
                LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()