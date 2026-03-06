# 🤖 Coqui XTTS-v2 Flask Web UI & API

Real-time voice cloning (TTS) generator using the **Coqui XTTS-v2** model. This implementation features a premium real-time dashboard and a robust automation API, surgically optimized for next-generation hardware.

---

## 🚀 1. Quick Start (Simplified for Everyone)

This project is designed to be as simple as possible. You don't need technical knowledge to run it.

1.  **Download/Clone** this repository to your computer.
2.  **Double-click the `run.bat` file**.
3.  The system will **automatically**:
    *   Create an isolated environment (venv).
    *   Install all necessary technical components.
    *   Initialize the AI engine (Optimized for **Blackwell/RTX 50**).
    *   **Open the Web Dashboard** in your default browser.

---

## ⚠️ 2. Technical Requirements & Environment

This project is resource-intensive due to the high-performance XTTS-v2 model.

### 2.1. 🐍 Python Interpreter
- **Required Version:** **Python 3.10** (Essential for compatibility with Torch/Coqui dependencies).

### 2.2. 💻 Essential Hardware Specifications

| Requirement      | Detail                                         | Importance                                                                       |
| :--------------- | :--------------------------------------------- | :------------------------------------------------------------------------------- |
| **NVIDIA GPU**   | **Mandatory** for real-time generation.        | Critical. CPU generation is _extremely_ slow.                                    |
| **VRAM Memory**  | Minimum **8 GB VRAM** recommended.              | Necessary for loading the XTTS-v2 model in half precision (FP16).                |
| **CUDA**         | Version **12.8** (Blackwell / RTX 50 Series)   | Optimized for next-gen GPUs (RTX 5080+) and CUDA 12.x drivers.                   |

---

## 🎙️ 3. Voice Cloning & Directory Management

The engine uses **Reference Samples** to clone voices. For the system to function correctly and avoid errors, follow these rules:

### 3.1. 🎤 Speaker Samples (Reference)
*   **Where to put them**: All reference files **MUST** be placed in the `samples/` directory.
*   **Recommended Format**: Use `.wav` files (mono or stereo).
*   **Optimal Sample**: A 6 to 20-second clip of clear speech without background noise, music, or echoes.
*   **AI Tips**: High-quality samples result in surprisingly realistic clones. Avoid samples with multiple people speaking.

### 3.2. 📂 Output Folders (Critical)
The project is configured with a robust directory check. The following folders are part of the core structure:
*   `samples/`: Input directory for your reference voices.
*   `generated/`: Output directory where all synthesized `.wav` files are stored.
> **🛡️ Safety Note:** The system is prepared to handle these directories automatically. They are ignored by Git to protect your privacy, but the structure is preserved so the engine never "breaks" due to missing paths.

---

## 📚 4. Advanced Features & Interface

### 4.1. Premium Web Dashboard
Access `http://127.0.0.1:5000` to control the engine manually:
*   **Real-Time Monitoring**: Live activity log for both manual and API requests.
*   **Instant Sampling**: Switch between reference voices on the fly.
*   **Save Modes**: Use the "Overwrite" switch to keep a fixed filename or generate unique files with timestamps.

### 4.2. 🤖 Automation API (Developer & AI Agent Guide)
The system includes a production-ready API endpoint for scripts, tools, and AI agents.

**Endpoint:** `POST http://127.0.0.1:5000/api/generate`

#### 📊 Complete Parameter Technical Manual

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `text` | string | **YES** | - | The text to be synthesized into speech. |
| `sample` | string | NO | *Last UI* | Reference voice filename in `samples/` (e.g., `andres.wav`). |
| `language` | string | NO | `es` | Synthesis language (`en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, etc.). |
| `play_audio` | bool | NO | `false` | If `true`, the server plays the audio via system speakers upon generation. |
| `overwrite` | bool | NO | `false` | If `true`, saves as `[prefix].wav`. If `false`, adds a unique timestamp. |
| `speed` | float | NO | `1.05` | Speech speed factor (0.5 to 2.0). |
| `temperature`| float | NO | `0.05` | Variance factor (lower = more stable and faithful clone). |
| `repetition_penalty`| float | NO | `7.0` | Repetition penalty (values > 1.0 reduce stuttering/looping). |
| `top_p` | float | NO | `0.75` | Nucleus sampling factor for token diversity. |
| `top_k` | int | NO | `40` | Limits token selection to top K candidates. |
| `prefix` | string | NO | `api_audio` | Custom filename prefix for the generated file. |

---

## 🛡️ License & Ownership

**Lead Developer:** [Marbycore](https://github.com/marbycore)

This implementation represents an advanced integration of the Coqui XTTS-v2 engine, surgically optimized by **Marbycore** for NVIDIA Blackwell architectures and industrial automation.

*   **XTTS-v2** is a model by Coqui AI.
*   **Optimization, UI & Automation Layers** developed by **Marbycore**.

---
© 2026 Marbycore - Professional AI Implementation for Blackwell
