# CMakeLists.txt - builds the _jdm pybind11 extension (JDM bootstrap kernel).
# Installs into backend/python/joint_distribution_model/ (CyHAN v2.2 16.2 / 16.5).
# Kernel: JDMBootstrap.hpp (header-only) + jdm_bindings.cpp.

cmake_minimum_required(VERSION 3.18)
project(jdm_engine LANGUAGES CXX)

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

pybind11_add_module(_jdm jdm_bindings.cpp)
target_include_directories(_jdm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
  target_compile_options(_jdm PRIVATE /O2 /EHsc)
else()
  target_compile_options(_jdm PRIVATE -O3)
endif()

# Install the built extension into the orchestration package.
install(TARGETS _jdm
        LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../../python/joint_distribution_model"
        RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../../python/joint_distribution_model")
