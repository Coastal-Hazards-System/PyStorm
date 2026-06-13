cmake_minimum_required(VERSION 3.15)
project(pot_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -O3)
endif()
if(MSVC)
    add_compile_options(/O2 /W3)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(_pot pot_bindings.cpp)

target_include_directories(_pot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CyHAN v2.2 §16.2 / §16.5: install into the expanded orchestration package
# so callers write `from peaks_over_threshold import _pot`.
install(TARGETS _pot
        LIBRARY DESTINATION
        "${CMAKE_CURRENT_SOURCE_DIR}/../../python/peaks_over_threshold")
