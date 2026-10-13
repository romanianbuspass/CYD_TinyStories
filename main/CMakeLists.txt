idf_component_register(
    SRCS "app_main.c" "ui_story.c" "inference_task.c" "flash_model.c"
    INCLUDE_DIRS "."
    REQUIRES lvgl driver spi_flash esp_partition tinystories_engine st77922 board
)

# Register the model.bin partition data at build/flash time
set(MODEL_BIN "${CMAKE_SOURCE_DIR}/model_partition/model.bin")
idf_build_get_property(build_dir BUILD_DIR)
partition_table_get_partition_info(model_offset "--partition-name model" "offset")
esptool_py_flash_target_image(flash "model" "${model_offset}" "${MODEL_BIN}")
