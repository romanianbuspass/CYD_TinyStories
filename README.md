# CYD TinyStories

On-device TinyStories language-model inference on an ESP32-S3 "Cheap Yellow Display" (CYD) board. A quantized ~28.9M-parameter transformer (PLE architecture, int4 weights) runs entirely on the ESP32-S3 — no network, no cloud — and streams generated story tokens into a touch-driven LVGL UI. Tapping a seed-prompt button starts a new story; tapping another seed aborts generation and restarts instantly.

## Technology Stack

- **Framework:** ESP-IDF v5.3 (CMake build system)
- **UI:** LVGL 8.3.11 (fetched via the ESP Component Registry; see `dependencies.lock`)
- **Language:** C (FreeRTOS dual-core task model)
- **Inference engine:** Custom C engine in `components/tinystories_engine/` (derived from the llama2.c-style minimal inference approach), with:
  - int4-quantized weights (group size 128), memory-mapped directly from flash
  - Per-position int8 weight staging into PSRAM for the core/head matvecs
  - Float scratch in SRAM, KV cache + logits in PSRAM
- **Host verification:** `host_verify/` builds the engine against a stub (`stub_esp.h` / `shim/`) so the model can be sanity-checked on a PC before flashing

## Hardware Dependencies

- **Board:** Freenove ESP32-S3 WROOM ("CYD" form factor)
  - 16 MB QIO flash (80 MHz), 8 MB Octal PSRAM (80 MHz), 240 MHz CPU
- **Display:** ST77922 QSPI TFT panel (16-bit color, high-byte-first pixel order)
- **Touch:** ST77922 TDDI capacitive touch over I2C (addr `0x55`, SCL=GPIO39, SDA=GPIO38)
- Display/touch drivers live in `components/st77922/`, `components/cyd_display/`, board pinout in `components/board/`

## Repository Layout

| Path | Contents |
|---|---|
| `main/` | App entry (`app_main.c`), LVGL story UI (`ui_story.c`), Core-1 inference task (`inference_task.c`), flash partition mmap (`flash_model.c`) |
| `components/tinystories_engine/` | Model loader/forward/sampler (`ts_model.c`), BPE tokenizer (`ts_tokenizer.c`), embedded vocab (`generated/vocab.bin`) |
| `components/st77922/`, `components/board/`, `components/cyd_display/` | Display, touch, and board-support drivers |
| `model_src/` | `metadata.json` (architecture, quantization, measured perf, licenses) and `tokenizer.json` (MIT-licensed BPE tokenizer) |
| `model_partition/` | Staging location for `model.bin` (weights are **not** committed — see below) |
| `host_verify/` | PC-side build harness for the engine |
| `partitions.csv` | Custom partition table (see below) |
| `CYD_Tiny-Stroies.md` | Original design/task brief |

### Partition table (`partitions.csv`)

| Name | Type | Offset | Size | Purpose |
|---|---|---|---|---|
| `nvs` | data | 0x9000 | 24 KB | NVS |
| `phy_init` | data | 0xf000 | 4 KB | PHY cal |
| `factory` | app | 0x10000 | 1 MB | Firmware |
| `model` | data (subtype 0x40) | 0x110000 | ~14.9 MB | Model weights, memory-mapped at runtime |

## Model Weights (not included)

`model.bin` (14,912,348 bytes, int4-quantized, MIT-licensed weights) is **excluded from this repository** due to size. To build/flash you must supply it yourself:

- Place the weights at `model_partition/model.bin` (this exact path is registered in `main/CMakeLists.txt` via `esptool_py_flash_target_image`, so `idf.py flash` writes them to the `model` partition automatically).
- Expected SHA-256: `1d8326c05c383ccfa615f5455575802817cb453dbc7ab28875d41a9dbb45477e` (see `model_src/metadata.json`).
- The tokenizer (`model_src/tokenizer.json`) and embedded firmware vocab (`components/tinystories_engine/generated/vocab.bin`) **are** committed.

Measured performance on-device (from `metadata.json`): ~95 ms/token (~9.9 tokens/s), -O3 build, 240 MHz.

## Setup & Installation

1. Install [ESP-IDF v5.3](https://docs.espressif.com/projects/esp-idf/en/v5.3/esp32s3/get-started/) and export its environment (`. $IDF_PATH/export.sh`).
2. Clone this repo and drop `model.bin` into `model_partition/` (see above).
3. Configure and build:
   ```sh
   idf.py set-target esp32s3   # defaults come from sdkconfig.defaults
   idf.py build
   ```
   The LVGL dependency is fetched automatically into `managed_components/` by the component manager (pinned in `dependencies.lock`).
4. Flash and monitor:
   ```sh
   idf.py -p /dev/ttyUSB0 flash monitor
   ```
   The flash step writes both the factory app and the model partition.

Key `sdkconfig.defaults` settings: Octal PSRAM, 240 MHz CPU, 1000 Hz FreeRTOS tick, QIO 80 MHz flash, performance compiler optimization.

## Usage

- On boot the screen shows a read-only story text area (top ~75%) and a button matrix of seed prompts (bottom ~25%): *Once Upon*, *Dog Spot*, *Dark Forest*, *Shiny Ball*, *Tiny Robot*, *Random*.
- Tap a seed to start generation; tokens stream into the text area as they are produced.
- Tap another seed mid-generation to abort and restart with the new prompt.
- Serial log (`idf.py monitor`) shows per-token timing (`forward_us`) and generation stats.

### Architecture notes

- **Core 0:** LVGL tick, display refresh, and the token-stream consumer task (drains `token_stream_queue` into the textarea).
- **Core 1:** Inference task — encodes the prompt, runs the sampler loop, pushes token fragments to the queue; honors an abort flag when a new seed arrives.
- Queues: `inference_prompt_queue` (depth 1, overwritten on new seed) and `token_stream_queue` (depth 64 of 32-byte fragments).

## Data Formats

- **Model file (`model.bin`):** `PLE\0` magic, version 1, 56-byte header; int4 weights (group size 128); PLE table + tied token embedding memory-mapped from flash. Full spec in `model_src/metadata.json`.
- **Tokenizer:** BPE, 32,768 input vocab / 25,353 active entries; token id `0` = `<|endoftext|>`.
- **Training data provenance:** `roneneldan/TinyStories` (CDLA-Sharing-1.0); weights and tokenizer MIT-licensed.

## Licenses

Model weights and tokenizer: MIT. Training data: CDLA-Sharing-1.0. LVGL: MIT (see `managed_components/lvgl__lvgl/` after fetch).
