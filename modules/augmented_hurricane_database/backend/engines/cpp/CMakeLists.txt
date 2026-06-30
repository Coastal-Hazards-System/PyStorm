# CMakeLists.txt - builds the _gpm pybind11 extension (GP power-exponential kernel).
# Installs into backend/python/augmented_hurricane_database/gp_metamodel/
# (CyHAN v2.2 16.2 / 16.5). Kernel: gpm_kernel.hpp (header-only) + bindings.cpp.

cmake_minimum_required(VERSION 3.18)
project(gpm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  # Fall back to the pip-installed pybind11 headers/cmake config.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
  find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(_gpm bindings.cpp)
target_include_directories(_gpm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# OpenMP-parallel + fast-math: the kernel is the performance-critical inner loop
# of GP prediction and the per-evaluation R build. The GP releases the GIL, so
# threads scale freely. OpenMP is enabled when available (Apple clang lacks it
# out of the box, so on stock macOS the kernel builds and runs serial).
if(MSVC)
  target_compile_options(_gpm PRIVATE /O2 /EHsc /fp:fast)
else()
  target_compile_options(_gpm PRIVATE -O3 -ffast-math)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_gpm PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found - building _gpm with parallel kernel")
else()
  message(STATUS "OpenMP not found - building _gpm with serial kernel")
endif()

# Install the built extension into the GP-metamodel package.
install(TARGETS _gpm
        LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../../python/augmented_hurricane_database/gp_metamodel"
        RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../../python/augmented_hurricane_database/gp_metamodel")
