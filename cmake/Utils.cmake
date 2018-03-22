function(format_gencode_flags flags out)
    foreach(ver ${flags})
        set(${out} "${${out}}-gencode arch=compute_${ver},code=sm_${ver};")
    endforeach()
    set(${out} "${${out}}" PARENT_SCOPE)
endfunction(format_gencode_flags flags)

# Set a default build type to release if none was specified
function(set_default_configuration_release)
    if(CMAKE_CONFIGURATION_TYPES STREQUAL "Debug;Release;MinSizeRel;RelWithDebInfo") # multiconfig generator?
        set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
    elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to 'Release' as none was specified.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
    endif()
endfunction(set_default_configuration_release)
