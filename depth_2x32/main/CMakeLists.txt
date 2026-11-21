idf_component_register(
    SRCS "main.cpp" "methods.cpp"
    INCLUDE_DIRS "."
                 "${CMAKE_CURRENT_LIST_DIR}/../../shared"
    REQUIRES freertos esp_system esp_hw_support esp_timer heap esp_app_format esp_common
)

# The same flags apply to all methods (compiled header, generated baselines,
# and the runtime-style baseline).  They are printed by the firmware
# (INFO,cxx_flags) and reported in the paper.
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O2
    -ffast-math
    -funroll-loops
    -Wno-unused-variable
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    "BENCH_CXX_FLAGS=\"-O2 -ffast-math -funroll-loops\""
)

set_target_properties(${COMPONENT_LIB} PROPERTIES CXX_STANDARD 17)
