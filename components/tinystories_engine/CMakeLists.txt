idf_component_register(
    SRCS "ts_model.c" "ts_tokenizer.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_common
    EMBED_FILES "generated/vocab.bin"
)

# ts_model.c contains the hot matvec/attention inner loops (per-token
# latency critical); force -O3 regardless of the project-wide optimization
# level (perf here already, but be explicit and resilient to config
# changes) plus loop unrolling.
target_compile_options(${COMPONENT_LIB} PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3;-funroll-loops>
)
