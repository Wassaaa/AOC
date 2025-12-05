project(day5 LANGUAGES CXX)

add_executable(day5 main.cpp)

# ============================================================================
# Compiler Options
# ============================================================================
target_compile_options(day5 PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Release>:-O3 -g -mavx2 -DNDEBUG -fopt-info-vec-optimized>
    $<$<CONFIG:Debug>:-O0 -mavx2 -g -fopt-info-vec-optimized>
)

if(TARGET TBB::tbb)
    target_link_libraries(day5 PRIVATE TBB::tbb)
endif()
