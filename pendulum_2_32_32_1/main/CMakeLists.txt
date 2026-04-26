idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
                 "${CMAKE_CURRENT_LIST_DIR}/../../shared"
    REQUIRES freertos esp_system esp_hw_support esp_timer
)

# Use -O2 (not -O3) so codegen is comparable across runs and across
# compilers. -ffast-math allows reassociation but does not change the
# arithmetic count, so it remains fair between dual and reverse-mode.
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O2
    -ffast-math
    -funroll-loops
    -Wno-unused-variable
)

set_target_properties(${COMPONENT_LIB} PROPERTIES CXX_STANDARD 17)
