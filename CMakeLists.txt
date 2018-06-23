CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(h2o4gpu)
INCLUDE(cmake/Utils.cmake)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

SET_DEFAULT_CONFIGURATION_RELEASE()

FIND_PACKAGE(OpenMP)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(SWIG REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED) # SWIG

INCLUDE(${SWIG_USE_FILE})

#============= OPTIONS & SETTINGS
OPTION(USE_CUDA  "Build with GPU acceleration" ON)
OPTION(DEV_BUILD  "Dev build" OFF)

# Compiler flags
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")

# PythonLibs' PYTHON_INCLUDE_PATH doesn't take into account virtualenv etc.
# Open to suggestions how to do this better.
EXECUTE_PROCESS(COMMAND python -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_PATH_CUST
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#============= OPTIONS & SETTINGS

# TODO probably to be removed after POGS is out in favor of XGboost GLM
ADD_DEFINITIONS(
        -D_GITHASH_=0
        -DH2O4GPU_DOUBLE
        -DH2O4GPU_SINGLE
)

#============= BUILD COMMON CPU/GPU CODE
FILE(GLOB_RECURSE COMMON_SOURCES
        src/common/*.cpp
        src/common/*.h
        src/interface_c/*.cpp
        src/interface_c/*.h
        )

INCLUDE_DIRECTORIES(
        src/include
        src/cpu/include
        # Here and not in target_include_directories b/c cmake < 3.7 which we use in Dockerfiles does not support it
        src/gpu/include
        ${PYTHON_INCLUDE_PATH}
        ${PYTHON_INCLUDE_PATH_CUST}
)

ADD_LIBRARY(commonh2o4gpu OBJECT ${COMMON_SOURCES})
#============= BUILD COMMON CPU/GPU CODE

#============= BUILD CPU LIBRARY
FILE(GLOB_RECURSE CPU_SOURCES
        src/cpu/*.cpp
        src/cpu/*.h
        )

ADD_LIBRARY(cpuh2o4gpu STATIC ${CPU_SOURCES} $<TARGET_OBJECTS:commonh2o4gpu>)
TARGET_LINK_LIBRARIES(cpuh2o4gpu ${BLAS_LIBRARIES})
#============= BUILD CPU LIBRARY

#============= SWIG
SET(CMAKE_SWIG_FLAGS -Werror)
#============= SWIG

#============= CPU SWIG
SET_SOURCE_FILES_PROPERTIES(src/swig/ch2o4gpu_cpu.i PROPERTIES CPLUSPLUS ON)

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
        SWIG_ADD_MODULE(ch2o4gpu_cpu python src/swig/ch2o4gpu_cpu.i)
else()
        SWIG_ADD_LIBRARY(ch2o4gpu_cpu LANGUAGE python SOURCES src/swig/ch2o4gpu_cpu.i)
endif()

SWIG_LINK_LIBRARIES(ch2o4gpu_cpu cpuh2o4gpu ${PYTHON_LIBRARIES})

SET_TARGET_PROPERTIES(${SWIG_MODULE_ch2o4gpu_cpu_REAL_NAME} PROPERTIES
        LINK_FLAGS ${OpenMP_CXX_FLAGS})
#============= CPU SWIG

if(USE_CUDA)
        FIND_PACKAGE(CUDA 8.0 REQUIRED)
        FIND_PACKAGE(NVML REQUIRED)

        #============= BUILD GPU LIBRARY
        ADD_DEFINITIONS(
                -DCUDA_MAJOR=${CUDA_VERSION_MAJOR}
                -DHAVECUDA
        )

        if(DEV_BUILD)
                MESSAGE(STATUS "Building DEVELOPER compute capability version.")
                SET(GPU_COMPUTE_VER 61 CACHE STRING
                        "Space separated list of compute versions to be built against")
        else()
                MESSAGE(STATUS "Building RELEASE compute capability version.")
                SET(GPU_COMPUTE_VER 35;50;52;60;61 CACHE STRING
                        "Space separated list of compute versions to be built against")
        endif()

        if(((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9)) AND NOT DEV_BUILD)
                MESSAGE(STATUS "CUDA 9.0 detected, adding Volta compute capability (7.0).")
                SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};70")
        endif()

        SET(GENCODE_FLAGS "")
        FORMAT_GENCODE_FLAGS("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -std=c++11;--expt-extended-lambda;--expt-relaxed-constexpr;${GENCODE_FLAGS};-lineinfo; -w;")

        FILE(GLOB_RECURSE GPU_SOURCES
                src/*.cu
                src/*.cuh
                src/common/*.cpp
                src/common/*.h
                )

        CUDA_ADD_LIBRARY(gpuh2o4gpu ${GPU_SOURCES} $<TARGET_OBJECTS:commonh2o4gpu> STATIC)

        if($ENV{USENVTX})
                MESSAGE(STATUS "Building with NVTX support on.")
                SET(NVTX_LIBRARY nvToolsExt)
        endif()

        TARGET_LINK_LIBRARIES(gpuh2o4gpu
                ${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${CUDA_cusparse_LIBRARY}
                ${BLAS_LIBRARIES}
                ${NVTX_LIBRARY}
                ${NVML_LIBRARY})
        #============= BUILD GPU LIBRARY

        #============= GPU SWIG
        SET_SOURCE_FILES_PROPERTIES(src/swig/ch2o4gpu_gpu.i PROPERTIES CPLUSPLUS ON)

        if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
                SWIG_ADD_MODULE(ch2o4gpu_gpu python src/swig/ch2o4gpu_gpu.i)
        else()
                SWIG_ADD_LIBRARY(ch2o4gpu_gpu LANGUAGE python SOURCES src/swig/ch2o4gpu_gpu.i)
        endif()
        SWIG_LINK_LIBRARIES(ch2o4gpu_gpu gpuh2o4gpu ${PYTHON_LIBRARIES})

        SET_TARGET_PROPERTIES(${SWIG_MODULE_ch2o4gpu_gpu_REAL_NAME} PROPERTIES
                LINK_FLAGS ${OpenMP_CXX_FLAGS})
        #============= GPU SWIG
endif()
