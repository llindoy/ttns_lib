cmake_minimum_required (VERSION 3.0)
project(ttns_lib LANGUAGES CXX)

unset(COMPILER_SET CACHE)
unset(LIBS CACHE)

#pull the git submodules if they aren't present
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/linalg/CMakeLists.txt" OR NOT EXISTS  "${PROJECT_SOURCE_DIR}/utils/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

add_subdirectory(linalg)
add_subdirectory(utils)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

option(USE_CUDA "Compile with cuda" OFF)
option(USE_OPENMP "Compile with openmp" ON)
option(USE_GEMM3M "Compile with openmp" OFF)
if(USE_GEMM3M)
    add_compile_definitions(USE_GEMM3M)
endif()
option(JUST_INSTALL_TTNS_LIB "Don't do anything besides installing the Library" On)

add_library(ttns_lib INTERFACE)
target_include_directories(ttns_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
 
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
endif()

