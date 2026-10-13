#include "esp_partition.h"
#include "esp_log.h"
#include "tinystories_engine.h"

static const char *TAG = "flash_model";
static esp_partition_mmap_handle_t s_mmap_handle;

bool flash_model_mmap(ts_model_t *model_out) {
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40 /* custom subtype from partitions.csv */, "model");
    if (!part) {
        ESP_LOGE(TAG, "'model' partition not found");
        return false;
    }
    const void *mapped_ptr = NULL;
    esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                        ESP_PARTITION_MMAP_DATA, &mapped_ptr, &s_mmap_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err));
        return false;
    }
    ESP_LOGI(TAG, "model partition mmap'd: %u bytes at flash offset 0x%06x",
             (unsigned)part->size, (unsigned)part->address);
    return ts_model_load(model_out, mapped_ptr, part->size);
}
