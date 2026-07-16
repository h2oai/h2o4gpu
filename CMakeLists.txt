CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
PROJECT(h2o4gpu)
INCLUDE(cmake/Utils.cmake)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

SET_DEFAULT_CONFIGURATION_RELEASE()

FIND_PACKAGE(OpenMP)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(SWIG REQUIRED)
FIND_PACKAGE (Python COMPONENTS Interpreter Development)

# SWIG behaviour was changed
if (${CMAKE_VERSION} VERSION_GREATER "3.13")
        cmake_policy(SET CMP0078 OLD)
endif()

INCLUDE(${SWIG_USE_FILE})

#============= OPTIONS & SETTINGS
OPTION(USE_CUDA  "Build with GPU acceleration" ON)
OPTION(DEV_BUILD  "Dev build" OFF)
OPTION(DEV_SYNC  "call cudaDeviceSynchronize after each cuda call" OFF)

# Compiler flags
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")

# FIND_PACKAGE Python can't properly find numpy.
# TODO: use python from make/config.mk
EXECUTE_PROCESS(COMMAND python -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_PATH_CUST
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenBLAS ships cblas.h under an openblas/ subdir on RHEL/Rocky 8 rather than
# directly on the default include path, but the sources do #include <cblas.h>.
FIND_PATH(CBLAS_INCLUDE_DIR cblas.h PATHS /usr/include/openblas /usr/include)
if(CBLAS_INCLUDE_DIR)
        INCLUDE_DIRECTORIES(${CBLAS_INCLUDE_DIR})
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
        ${Python_INCLUDE_DIRS}
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

        # DAI provides the CUDA 12.8 runtime, so link the math libraries
        # DYNAMICALLY. Static linking (libcublasLt_static.a et al.) baked ~950 MB
        # of per-arch cubins into the extension; dynamic linking defers them to
        # DAI's runtime (libcublas/libcublasLt/libcusolver/libcusparse .so.12)
        # and keeps the wheel small. The explicit .so NAMES avoid picking up the
        # static archives. cudart stays static (via CUDA_ADD_LIBRARY), which is
        # why libculibos is still linked.
        set(CUDA_MATH_LIB_PATHS
                "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
                "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/"
                "/usr/lib64")
        find_library(CUBLAS_LIB   NAMES libcublas.so   PATHS ${CUDA_MATH_LIB_PATHS})
        find_library(CUBLASLT_LIB NAMES libcublasLt.so PATHS ${CUDA_MATH_LIB_PATHS})
        find_library(CUSPARSE_LIB NAMES libcusparse.so PATHS ${CUDA_MATH_LIB_PATHS})
        find_library(CUSOLVER_LIB NAMES libcusolver.so PATHS ${CUDA_MATH_LIB_PATHS})
        find_library(CULIBOS_STATIC_LIB NAMES libculibos.a
                PATHS   "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
                        "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/")

        set(STATIC_LIBS ${CUBLAS_LIB})
        list(APPEND STATIC_LIBS ${CUBLASLT_LIB})
        list(APPEND STATIC_LIBS ${CUSPARSE_LIB})
        list(APPEND STATIC_LIBS ${CUSOLVER_LIB})
        list(APPEND STATIC_LIBS ${CULIBOS_STATIC_LIB})

        if(DEV_BUILD)
            MESSAGE(STATUS "Building DEVELOPER compute capability version.")
            SET(GPU_COMPUTE_VER 61)
            SET(CMAKE_BUILD_TYPE Debug)
        elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 12)
            # DAI-aligned ladder (matches DAI's xgboost/lightgbm wheels): Pascal
            # through Blackwell. sm_86/87/89 run on the sm_80 cubin (binary
            # compatible), so only sm_80 is built; sm_62 (Tegra) is omitted.
            MESSAGE(STATUS "Building RELEASE compute capability version (CUDA 12 baseline: Pascal..Hopper).")
            SET(GPU_COMPUTE_VER 60;61;70;75;80;90)

            # Blackwell SASS (sm_100 datacenter, sm_120 consumer) requires CUDA >= 12.8.
            if(CUDA_VERSION VERSION_GREATER_EQUAL "12.8")
                MESSAGE(STATUS "CUDA >= 12.8 detected, adding Blackwell compute capability (10.0, 12.0).")
                SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};100;120")
            endif()
        else()
            MESSAGE(STATUS "Building RELEASE compute capability version (legacy CUDA baseline).")
            SET(GPU_COMPUTE_VER 35;37;50;52;53;60;61;62)

            if(CUDA_VERSION_MAJOR GREATER_EQUAL 9)
                MESSAGE(STATUS "CUDA >= 9.0 detected, adding Volta compute capability (7.0).")
                SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};70")
            endif()
            if(CUDA_VERSION_MAJOR GREATER_EQUAL 10)
                MESSAGE(STATUS "CUDA >= 10.0 detected, adding Turing compute capability (7.5).")
                SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};75")
            endif()
        endif()

        if(DEV_SYNC)
            MESSAGE(STATUS "Building with device sync.")
            add_compile_definitions(SYNC)
        endif()

        # cublasLt is now linked dynamically (with the other math libs above).
        # The static-cusolver dependencies libcusolver_lapack_static.a and
        # libcusolver_metis_static.a are only required when cusolver is linked
        # statically, so they are no longer needed.

        SET(GENCODE_FLAGS "")
        FORMAT_GENCODE_FLAGS("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
        MESSAGE(STATUS "cuda architecture flags: ${GENCODE_FLAGS}")

        # Debug symbols (-g host) and device line tables (-lineinfo) bloat the
        # shipped .so with .debug_* sections; keep them only for DEV_BUILD.
        if(DEV_BUILD)
            SET(DEBUG_NVCC_FLAGS "-g;-lineinfo;")
        else()
            SET(DEBUG_NVCC_FLAGS "")
        endif()
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -std=c++17;--expt-extended-lambda;--expt-relaxed-constexpr;${GENCODE_FLAGS};${DEBUG_NVCC_FLAGS} -w;")

        FILE(GLOB_RECURSE GPU_SOURCES
                src/*.cu
                src/*.cuh
                src/common/*.cpp
                src/common/*.h
                )

        # DAI-minimal CUDA 12 build: the matrix-factorization / ALS solver uses
        # cuSPARSE level-3 routines (csrmm2) that were removed in CUDA 12. DAI
        # only consumes TruncatedSVD/KMeans from the GPU lib, so exclude this
        # solver rather than porting it to the generic cusparseSpMM API.
        # TODO(blackwell): port factorization to the generic cuSPARSE API to
        # restore the matrix-factorization solver.
        list(FILTER GPU_SOURCES EXCLUDE REGEX ".*/factorization/.*\\.cu$")

        CUDA_ADD_LIBRARY(gpuh2o4gpu ${GPU_SOURCES} $<TARGET_OBJECTS:commonh2o4gpu> STATIC)
        SET_TARGET_PROPERTIES(gpuh2o4gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        if($ENV{USENVTX})
            MESSAGE(STATUS "Building with NVTX support on.")
            SET(NVTX_LIBRARY nvToolsExt)
        endif()



        TARGET_LINK_LIBRARIES(gpuh2o4gpu
                ${STATIC_LIBS}
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
        SWIG_LINK_LIBRARIES(ch2o4gpu_gpu gpuh2o4gpu
                ${PYTHON_LIBRARIES}
                ${STATIC_LIBS})

        SET_TARGET_PROPERTIES(${SWIG_MODULE_ch2o4gpu_gpu_REAL_NAME} PROPERTIES
                LINK_FLAGS ${OpenMP_CXX_FLAGS})
        #============= GPU SWIG


        #============= GTest
        ADD_SUBDIRECTORY (tests/googletest)
        enable_testing()
        INCLUDE(GoogleTest)
        include_directories(tests/googletest/include tests/googletest)

        #============= Unit Tests
        # file(GLOB_RECURSE TESTS tests/cpp/gpu/*.cu)
        FILE(GLOB_RECURSE TEST_SOURCES
        tests/cpp/gpu/*.cu
        tests/cpp/gpu/*.cuh
        tests/cpp/gpu/*.h
        )

        # Exclude the ALS/factorization test: that solver is not built on CUDA 12
        # (uses cuSPARSE csrmm2, removed in CUDA 12). See GPU_SOURCES filter above.
        list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/factorization/.*")

        CUDA_ADD_EXECUTABLE(h2o4gpu_test tests/cpp/test_main.cu ${TEST_SOURCES})
        target_link_libraries(h2o4gpu_test
        gtest
        commonh2o4gpu
        gpuh2o4gpu
        ${STATIC_LIBS}
        ${BLAS_LIBRARIES}
        ${NVTX_LIBRARY}
        ${NVML_LIBRARY})
endif()
