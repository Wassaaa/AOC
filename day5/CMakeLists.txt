project(day5 LANGUAGES CXX)

add_executable(day5 main.cpp)

# ============================================================================
# Compiler Options
# ============================================================================
target_compile_options(day5 PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
)

if(TARGET TBB::tbb)
    target_link_libraries(day5 PRIVATE TBB::tbb)
endif()
