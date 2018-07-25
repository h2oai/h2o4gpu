CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(h2o4gpu)
INCLUDE(cmake/Utils.cmake)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

SET_DEFAULT_CONFIGURATION_RELEASE()

FIND_PACKAGE(OpenMP)
FIND_PACKAGE(BLAS REQUIRED)

#============= OPTIONS & SETTINGS
OPTION(USE_CUDA  "Build with GPU acceleration" ON)
OPTION(USE_SWIG "Use swig to generate language bindings." ON)
OPTION(BUILD_TESTS "Build tests." ON)
OPTION(USE_SYSTEM_GTEST "Use system google tests." ON)
OPTION(DEV_BUILD  "Dev build" OFF)

# Compiler flags
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
SET(GPU_COMPUTE_VER "" CACHE STRING
  "Semicolon separated list of compute versions to be built against,  e.g. -DGPU_COMPUTE_VER='35;61'")

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#============= OPTIONS & SETTINGS

# TODO probably to be removed after POGS is out in favor of XGboost GLM
ADD_DEFINITIONS(
  -D_GITHASH_=0
  -DH2O4GPU_DOUBLE
  -DH2O4GPU_SINGLE)

#============= BUILD COMMON CPU/GPU CODE
FILE(GLOB_RECURSE COMMON_SOURCES
  src/common/*.cpp
  src/common/*.h
  src/interface_c/*.cpp
  src/interface_c/*.h)

INCLUDE_DIRECTORIES(
  src/include
  src/cpu/include
  # Here and not in target_include_directories b/c cmake < 3.7 which we use in Dockerfiles does not support it
  src/gpu/include)

ADD_LIBRARY(commonh2o4gpu OBJECT ${COMMON_SOURCES})
#============= BUILD COMMON CPU/GPU CODE

#============= BUILD CPU LIBRARY
FILE(GLOB_RECURSE CPU_SOURCES
  src/cpu/*.cpp
  src/cpu/*.h)

ADD_LIBRARY(cpuh2o4gpu STATIC ${CPU_SOURCES} $<TARGET_OBJECTS:commonh2o4gpu>)
TARGET_LINK_LIBRARIES(cpuh2o4gpu ${BLAS_LIBRARIES})
#============= BUILD CPU LIBRARY

if(USE_CUDA)
  FIND_PACKAGE(CUDA 8.0 REQUIRED)
  FIND_PACKAGE(NVML REQUIRED)

  #============= BUILD GPU LIBRARY
  ADD_DEFINITIONS(
    -DCUDA_MAJOR=${CUDA_VERSION_MAJOR}
    -DHAVECUDA
    )

  if(DEV_BUILD AND NOT GPU_COMPUTE_VER)
    MESSAGE(STATUS "Building DEVELOPER compute capability version.")
    SET(GPU_COMPUTE_VER 61)
  elseif(NOT GPU_COMPUTE_VER)
    MESSAGE(STATUS "Building RELEASE compute capability version.")
    SET(GPU_COMPUTE_VER 35;50;52;60;61)
  endif()

  if(((CUDA_VERSION_MAJOR EQUAL 9)
	OR (CUDA_VERSION_MAJOR GREATER 9))
      AND NOT DEV_BUILD
      AND NOT GPU_COMPUTE_VER)
    MESSAGE(STATUS "CUDA 9.0 detected, adding Volta compute capability (7.0).")
    SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};70")
  endif()

  SET(GENCODE_FLAGS "")
  FORMAT_GENCODE_FLAGS("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
  MESSAGE("CUDA architecture flags ${GENCODE_FLAGS}")

  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -std=c++11;--expt-extended-lambda;--expt-relaxed-constexpr;${GENCODE_FLAGS};-lineinfo; -Wall;")

  FILE(GLOB_RECURSE GPU_SOURCES
    src/*.cu
    src/*.cuh
    src/gpu/kmeans/KmMatrix/*.cpp
    src/gpu/kmeans/KmMatrix/*.hpp
    src/common/*.cpp
    src/common/*.h)

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
endif()

if(USE_SWIG)
  ADD_SUBDIRECTORY(${CMAKE_CURRENT_LIST_DIR}/src/swig)
endif(USE_SWIG)

#============= Tests
if(BUILD_TESTS)
  ENABLE_TESTING()
  if (USE_SYSTEM_GTEST)
    FIND_PACKAGE(GTest REQUIRED)
  else ()
    ADD_SUBDIRECTORY(${CMAKE_CURRENT_LIST_DIR}/tests/googletest)
    SET(GTEST_LIBRARIES gtest gtest_main)
  endif (USE_SYSTEM_GTEST)
  if (USE_CUDA)
    FILE(GLOB_RECURSE CUDA_TEST_SOURCES "tests/cpp/*.cu")
    CUDA_COMPILE(CUDA_TEST_OBJS ${CUDA_TEST_SOURCES})
  endif(USE_CUDA)

  ADD_EXECUTABLE(test-h2o4gpu ${CUDA_TEST_OBJS})
  TARGET_LINK_LIBRARIES(test-h2o4gpu gpuh2o4gpu)
  TARGET_LINK_LIBRARIES(test-h2o4gpu ${GTEST_LIBRARIES})

  ADD_TEST(TestH2O4GPU test-h2o4gpu)
endif(BUILD_TESTS)
#============= Tests
