cmake_minimum_required(VERSION 3.22)

project(AOC LANGUAGES CXX)

# ============================================================================
# Global Project Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_package(TBB QUIET)
endif()

add_subdirectory(day3)
add_subdirectory(day5)

