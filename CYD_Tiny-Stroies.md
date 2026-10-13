# Hermes Agent Task: CYD TinyStories Inference Setup & Interactive UI

## Environment & Hardware Profile
- **Workspace**: `/home/sysop/.hermes/Workspace/CYD_TinyStories`
- **Target Board**: Freenove ESP32-S3 WROOM (CYD)
- **Memory**: 8MB Octal PSRAM, 16MB Flash
- **Display**: Integrated TFT LCD with touch controller (SPI)
- **UI Framework**: LVGL v8 / v9
- **Inference Engine**: `llama2.c` / `esp32-ai-tinystories` (28.9M quantized weights)

## Objective
Initialize an ESP-IDF C/C++ project running a quantized TinyStories language model on the ESP32-S3. The system streams generated tokens straight into an auto-scrolling LVGL text display on Core 0, while inference executes on Core 1. The interface must include a persistent bottom button matrix containing selectable seed prompts so users can trigger new stories without a keyboard.

---

## Execution Steps

### 1. Project Initialization & Partitioning
- Scaffold a standard ESP-IDF project structure within `/home/sysop/.hermes/Workspace/CYD_TinyStories`.
- Create `partitions.csv` allocating maximum available flash space for the model weights:
  - Custom data partition (type `data`, subtype `spiffs` or raw `rodata`) sized to fit the ~14.9MB quantized `.bin`.
  - Lean factory app partition (~1MB).
- Configure `sdkconfig`:
  - `CONFIG_ESP32S3_DEFAULT_CPU_FREQ_240=y`
  - `CONFIG_SPIRAM_MODE_OCT=y` (Octal PSRAM enabled)
  - Enable ESP-DSP / SIMD optimizations for matrix multiplication routines.
  - Set FreeRTOS tick rate to 1000Hz (`CONFIG_FREERTOS_HZ=1000`).

### 2. Dependency Sourcing & Engine Integration
- Pull the LVGL component and the display/touch driver components matching the Freenove CYD pinout.
- Vendor the core C inference engine files into `components/tinystories_engine/`.

### 3. Display Interface & Seed Selector Implementation
Build a modular UI with two visual regions:
- **Upper 75%**: Read-only `lv_textarea` configured for word-wrapping and programmatic scrolling.
- **Lower 25%**: `lv_btnmatrix` presenting tactile prompt seeds.

Implement the interface module (e.g., `main/ui_story.c`):

```c
#include "lvgl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

static lv_obj_t *story_text_area;
static lv_obj_t *seed_matrix;

extern QueueHandle_t inference_prompt_queue;
extern QueueHandle_t token_stream_queue;

static const char *seed_map[] = {
    "Once Upon", "Dog Spot", "Dark Forest", "\n",
    "Shiny Ball", "Tiny Robot", "Random", ""
};

static const char *full_prompts[] = {
    "Once upon a time, ",
    "Tim and his dog Spot found a ",
    "In a dark, quiet forest, ",
    "Lily had a shiny red ball, but ",
    "A tiny robot woke up and ",
    NULL
};

static void seed_btn_event_cb(lv_event_t *e) {
    lv_event_code_t code = lv_event_get_code(e);
    lv_obj_t *obj = lv_event_get_target(e);

    if (code == LV_EVENT_VALUE_CHANGED) {
        uint32_t id = lv_btnmatrix_get_selected_btn(obj);
        const char *prompt = NULL;

        if (id < 4) {
            prompt = full_prompts[id];
        } else if (id == 4) {
            prompt = full_prompts[4];
        } else if (id == 5) {
            prompt = "A little cat wanted to ";
        }

        if (prompt != NULL && inference_prompt_queue != NULL) {
            // Clear current story and dispatch prompt
            lv_textarea_set_text(story_text_area, "");
            xQueueOverwrite(inference_prompt_queue, &prompt);
        }
    }
}

void init_story_ui(lv_obj_t *parent) {
    // 1. Text display region
    story_text_area = lv_textarea_create(parent);
    lv_obj_set_size(story_text_area, LV_PCT(100), LV_PCT(75));
    lv_obj_align(story_text_area, LV_ALIGN_TOP_MID, 0, 0);
    lv_textarea_set_text(story_text_area, "Tap a seed below to begin a story...\n\n");
    lv_textarea_set_cursor_click_pos(story_text_area, false);
    lv_obj_clear_flag(story_text_area, LV_OBJ_FLAG_CLICK_FOCUSABLE);

    // 2. Tactile seed buttons
    seed_matrix = lv_btnmatrix_create(parent);
    lv_btnmatrix_set_map(seed_matrix, seed_map);
    lv_obj_set_size(seed_matrix, LV_PCT(100), LV_PCT(25));
    lv_obj_align(seed_matrix, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_add_event_cb(seed_matrix, seed_btn_event_cb, LV_EVENT_VALUE_CHANGED, NULL);
}

void ui_append_token(const char *token) {
    if (story_text_area != NULL && token != NULL) {
        lv_textarea_add_text(story_text_area, token);
    }
}
4. Multithreading & Queue Synchronization
Core Allocation:

Pin the LVGL ticker and display refresh task to Core 0.

Pin the TinyStories inference task to Core 1 to isolate computation from UI rendering.

Queues:

inference_prompt_queue (QueueHandle_t, depth 1): Delivers the seed string pointer to the model sampler loop.

token_stream_queue (QueueHandle_t, depth 64): Receives newly generated token fragments from the sampler and feeds them into the UI task for safe rendering via ui_append_token().

Interrupt Handling: When a new seed is tapped while inference is running, trigger an abort flag inside the sampler loop so generation halts and restarts instantly with the new prompt.

5. Build and Validation
Write main/CMakeLists.txt to register components, bundle drivers, and expose the partition binary target.

Run idf.py build to confirm compilation cleanly passes without symbol collisions or stack overflows.


<ElicitationsGroup message="Next implementation files Hermes can generate:">
  <Elicitation label="Draft CMakeLists.txt and partitions.csv" query="Generate the exact CMakeLists.txt and partitions.csv matching this Hermes task."/>
  <Elicitation label="Write Core 1 inference loop code" query="Write the C code for the Core 1 FreeRTOS inference task that reads from the prompt queue and handles abort signals."/>
</ElicitationsGroup>