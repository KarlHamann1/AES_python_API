# AES_PYTHON_API — Side-Channel Capture & Analysis (Arduino + Raspberry Pi Bare-Metal)

Turn‑key repository for collecting, storing, and analyzing side‑channel traces (power/optical) from:
- **Arduino (AVR)** running an AES demo, and
- **Raspberry Pi (AArch64) bare‑metal AES kernel** with a **hybrid console** (ASCII shell **and** binary UART protocol).

Acquisition uses a **PicoScope 5000A** (Python SDK) with hardware triggers (MCU pin / **GPIO16** on Pi) and saves traces as **`.npy`** or **`.npz`** with metadata. Evaluation scripts include CPA/DPA, TVLA capture, and filtering utilities.

---

## Repository layout

```
AES_PYTHON_API/
├─ arduino/
│  ├─ scripts/                 # capture & utilities for Arduino target
│  └─ data/                    # many trace folders (created by scripts)
├─ pi/
│  ├─ scripts/                 # Pi bare-metal API + capture & processing
│  └─ data/                    # many trace folders (created by scripts)
├─ scope/                      # PicoScope drivers/wrappers (Python side)
├─ evaluation/                 # CPA/DPA/TVLA and filtering/visualization
├─ .venv/ .vscode/ __pycache__/ .gitignore
└─ README.md
```

> **Tip:** run scripts **from the repo root** so relative imports like `from scope.picoscope_acquisition import DataAcquisition` work.  
> Example: `python arduino/scripts/batch_aquisition_better_100x.py`

---

## Hardware at a glance

- **Scope:** PicoScope 5000A series (A‑API via `picosdk`) with **EXT trigger**.
- **Sensor:** photodiode + TIA (or shunt/power probe) → **Channel A**.
- **Trigger:**
  - **Arduino:** MCU GPIO toggled around AES (wire to scope **EXT**).
  - **Raspberry Pi:** bare‑metal kernel toggles **GPIO16** around AES; wire to **EXT**.
- **UART:** host PC ↔ target (Arduino / Pi bare‑metal) for plaintext/ciphertext and control.

---

## Raspberry Pi bare‑metal kernel (AArch64)

- Boots as **`kernel8.img`** (Pi 3/4).  
- **ASCII shell** (banner + prompt) for manual commands:  
  `help`, `aes <num>`, `dummy <num>`, `mult <num>`, `rsa <num>`, `plaintext <32-hex>`.
- **Binary UART protocol (auto‑detected):**
  - `0x01` **SET_STATE**  (+16 bytes)
  - `0x02` **GET_STATE**  (→ 16 bytes)
  - `0x03` **ENCRYPT**    (one AES; scope trigger on GPIO16 inside AES)
  - `0x04` **ENCRYPT_N**  (+uint16 big‑endian; run N AES)
- Uses **GPIO16** as trigger; **mini‑UART** (AUX) on GPIO14/15 @ 115200 Baud.  
  For stable baud, pin the core clock in `config.txt` (see Raspberry Pi docs).

### SD boot files
Pi 3: `bootcode.bin`, `start.elf`, `fixup.dat`, `config.txt`, `kernel8.img`  
Pi 4: `start4.elf`, `fixup4.dat`, `config.txt`, `kernel8.img`

---

## Software prerequisites

- **Python 3.8+** (3.10+ recommended)
- Python packages (minimal):
  ```bash
  pip install numpy matplotlib scipy tqdm pyserial pycryptodome picosdk
  ```
- **PicoSDK C libraries** + **`picosdk` Python wrappers** (install vendor SDK; then `pip install picosdk`)
- (Pi users) A way to prepare the FAT32 boot partition (`kernel8.img`, firmware files)

---

## Key Python modules in this repo

- `scope/picoscope_acquisition.py`  
  Block capture helper around the **ps5000a** API: resolution, timebase, channel setup, (AC/DC coupling), optional **20 MHz BW limit**, analogue offset, EXT trigger, `prepare_block_mode()` + `capture_block()` convenience.

- `scope/picoscope_segmented_acquisition.py`  
  Segmented memory captures (multiple triggers in one run), then per‑segment retrieval.

- `arduino/scripts/…`  
  - `batch_aquisition_better_100x.py` — averaged capture: set PT, trigger AES, capture `n_avg`, save **mean** to `.npz` with PT/CT/metadata + CSV index.  
  - `batch_aquisition.py` — single‑trace capture per encryption.  
  - `batch_no_encryption_better.py` — **noise‑only** (auto‑trigger, no AES), baseline set.  
  - `TVLA_batch_aquisition.py` — interleaved RANDOM/FIXED capture for first‑order TVLA.  
  - `test_connectivity.py` — manual UART poke/spy utility.

- `pi/scripts/…`  
  - `pi_bare_metal_api.py` — host‑side UART API for the Pi kernel (binary protocol + ASCII helpers).  
  - `capture_pi_averaged.py` — averaged capture against the Pi kernel (like Arduino version).  
  - `process_pi_data.py`, `subtract_dummy_average.py` — post‑processing helpers.

- `evaluation/…`  
  CPA/DPA scripts for `.npz` **and** `.npy`, ROI by samples/time (if `dt_ns` present), key check, quick plots, FFT tools, band/high/low‑pass filtering, and a flexible **NPZ/NPY visualizer**.

---

## Data formats

- **`.npy`** — 1D float32 waveform.
- **`.npz`** — keys may include:
  - `trace` **or** `trace_mean` (float32),  
  - `plaintext`, `ciphertext` (uint8[16]),  
  - `sr_nominal` (Hz), `dt_ns` (ns), `timebase` (int), `n_avg` (int).  
- Many capture scripts also write a **CSV index** alongside traces.

---

## Quick start

### 1) Create a venv and install deps
```bash
cd AES_PYTHON_API
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install numpy matplotlib scipy tqdm pyserial pycryptodome picosdk
```

### 2) PicoSDK
Install the vendor **PicoSDK** for your OS (match Python bitness), then ensure the shared library is on your PATH/LD_LIBRARY_PATH.

### 3) Wire the trigger
- **Sensor → Channel A**, **Trigger → EXT**.  
- **Arduino:** toggle a digital pin during AES.  
- **Pi:** connect **GPIO16** to EXT.

### 4) Run a capture (examples)
```bash
# Arduino averaged capture (100x):
python arduino/scripts/batch_aquisition_better_100x.py --help
python arduino/scripts/batch_aquisition_better_100x.py

# Arduino single‑trace capture:
python arduino/scripts/batch_aquisition.py

# Pi averaged capture:
python pi/scripts/capture_pi_averaged.py
```

### 5) Analyze
```bash
# CPA on averaged NPZ (ROI optional):
python evaluation/AES_analysis_CPA_npz.py

# CPA on NPY/NPZ (CSV index of files):
python evaluation/cpa_aes_encryption.py

# Visualize NPZ/NPY quickly:
python evaluation/visualize_npz.py
```

---

## Troubleshooting

- **Serial port missing / access denied:** correct COM port (`COMx` / `/dev/ttyUSB*`), permissions (`dialout` on Linux).  
- **Pico not found:** install PicoSDK, use matching 32/64‑bit, check drivers.  
- **Pi mini‑UART baud drift:** pin the core clock / use PL011; check `enable_uart` and `core_freq` in `config.txt`.  
- **Empty captures:** verify trigger polarity/threshold, wiring to **EXT**, and that the firmware actually toggles the trigger around AES.

---

## References

- NIST. *FIPS 197: Advanced Encryption Standard (AES).*  
  https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf

- Pico Technology Ltd. *PicoScope 5000 Series (A API) Programmer’s Guide.*  
  https://www.picotech.com/download/manuals/picoscope-5000-series-a-api-programmers-guide.pdf

- Raspberry Pi. *config.txt documentation* (UART / clock notes).  
  https://www.raspberrypi.com/documentation/computers/config_txt.html

- pySerial Documentation.  
  https://pyserial.readthedocs.io/
