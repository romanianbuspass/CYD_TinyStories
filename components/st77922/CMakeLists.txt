idf_component_register(
    SRCS "st77922.c" "st77922_touch.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_timer freertos board
)
