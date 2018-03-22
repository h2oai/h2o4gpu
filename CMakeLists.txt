cmake_minimum_required(VERSION 3.2)
project(h2o4gpu)
include(cmake/Utils.cmake)

find_package(OpenMP)
find_package(BLAS REQUIRED)

# Options
option(USE_CUDA  "Build with GPU acceleration" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_COMPUTE_VER 35;50;52;60;61 CACHE STRING
        "Space separated list of compute versions to be built against")

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# TODO probably to be removed
add_definitions(
        -D_GITHASH_=0
        -DH2O4GPU_DOUBLE
        -DH2O4GPU_SINGLE
)

if(USE_CUDA)
        find_package(CUDA 8.0 REQUIRED)

        add_definitions(
                -DCUDA_MAJOR=${CUDA_VERSION_MAJOR}
                -DHAVECUDA
        )
endif()

file(GLOB_RECURSE SOURCES
        src/common/*.cpp
        src/common/*.h
        )

include_directories (
        src/include
        src/gpu/include
)

add_library(objh2o4gpu OBJECT ${SOURCES})

add_library(cpuh2o4gpu SHARED $<TARGET_OBJECTS:objh2o4gpu>)

set_target_properties(cpuh2o4gpu PROPERTIES
        OUTPUT_NAME ch2o4gpu_cpu
        LIBRARY_OUTPUT_DIRECTORY src/interface_c
        )

if(USE_CUDA)
        set(GENCODE_FLAGS "")
        format_gencode_flags("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -std=c++11;--expt-extended-lambda;--expt-relaxed-constexpr;${GENCODE_FLAGS};-lineinfo;")

        if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
                message("CUDA 9.0 detected, adding Volta compute capability (7.0).")
                set(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};70")
        endif()

        file(GLOB_RECURSE CUDA_SOURCES
                src/*.cu
                src/*.cuh
                )

        cuda_add_library(cuda_static ${CUDA_SOURCES} STATIC)

        target_link_libraries(cuda_static ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})

        add_library(gpuh2o4gpu SHARED $<TARGET_OBJECTS:objh2o4gpu>)

        target_link_libraries(gpuh2o4gpu ${BLAS_LIBRARIES} cuda_static)

        set_target_properties(gpuh2o4gpu PROPERTIES
                OUTPUT_NAME ch2o4gpu_gpu
                LIBRARY_OUTPUT_DIRECTORY src/interface_c
                )
endif()