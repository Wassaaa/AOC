project(day3 LANGUAGES CXX)

add_executable(day3 main.cpp)
# ============================================================================
# Compiler Options
# ============================================================================
target_compile_options(day3 PRIVATE
    $<$<CONFIG:Release>:-O3 -mavx2 -march=native -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g -mavx2 -march=native>
)
