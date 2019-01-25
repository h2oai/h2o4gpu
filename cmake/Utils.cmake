# Generate nvcc compiler flags given a list of architectures
# Also generates PTX for the most recent architecture for forwards compatibility
 function(format_gencode_flags flags out)
  # Generate SASS
   foreach(ver ${flags})
     set(${out} "${${out}}-gencode=arch=compute_${ver},code=sm_${ver};")
   endforeach()
  # Generate PTX for last architecture
  list(GET flags -1 ver)
  set(${out} "${${out}}-gencode=arch=compute_${ver},code=compute_${ver};")

  set(${out} "${${out}}" PARENT_SCOPE)
endfunction(format_gencode_flags flags)

# Set a default build type to release if none was specified
function(set_default_configuration_release)
    if($ENV{CMAKE_BUILD_TYPE} MATCHES "Debug|Release|MinSizeRel|RelWithDebInfo")
        message(STATUS "Setting build type to $ENV{CMAKE_BUILD_TYPE}.")
        set(CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE )
    elseif(CMAKE_CONFIGURATION_TYPES STREQUAL "Debug;Release;MinSizeRel;RelWithDebInfo") # multiconfig generator?
        set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
    elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to 'Release' as none was specified.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
    endif()
endfunction(set_default_configuration_release)
