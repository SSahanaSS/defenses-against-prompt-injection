# 🛡️ Prompt Injection Detector API

A **FastAPI-based multimodal prompt-security pipeline** that extracts text from text/image/audio inputs, filters threats using a heuristic layer and DistilBERT, and forwards only safe prompts to Gemini for a response. 🤖🔐

## 🏗️ Architecture

```text
                     ┌──────────────────────┐
                     │        👤 Client     │
                     │ Text / Image / Audio │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │       ⚡ FastAPI      │
                     │     POST /ingest     │
                     └──────────┬───────────┘
                                │
                                ▼
              ┌──────────────────────────────────┐
              │      📥 Input Processing          │
              │                                  │
              │ 📝 Text  → Direct                │
              │ 🖼️ Image → Tesseract OCR         │
              │ 🎙️ Audio → Whisper Transcription │
              └────────────────┬─────────────────┘
                               │
                               ▼
                     ┌──────────────────────┐
                     │ 🧹 Text Preprocessing │
                     │     Emoji Removal    │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ 🛡️ Layer 1: Heuristic│
                     │       Security       │
                     └──────────┬───────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
                    🚨 Unsafe           ✅ Safe
                       │                 │
                       ▼                 ▼
                    🚫 BLOCK       ┌────────────────┐
                                   │ 🧠 Layer 2:    │
                                   │   DistilBERT   │
                                   └───────┬────────┘
                                           │
                                  ┌────────┴────────┐
                                  │                 │
                               🚨 Unsafe           ✅ Safe
                                  │                 │
                                  ▼                 ▼
                               🚫 BLOCK       ┌────────────┐
                                             │ 🤖 Gemini  │
                                             │    2.5     │
                                             └─────┬──────┘
                                                   │
                                                   ▼
                                            💬 Final Response
```

## 🔧 Components

### 1. ⚡ FastAPI API Layer

FastAPI provides the backend API and exposes the `/ingest` endpoint.

It accepts:

* 📝 Text input
* 🖼️ Image uploads
* 🎙️ Audio uploads
* 📋 Optional metadata

---

### 2. 🌐 Multimodal Input Processing

The system converts different input types into text before security analysis.

| 📥 Input    | ⚙️ Processing                  |
| ----------- | ------------------------------ |
| 📝 Text     | Used directly                  |
| 🖼️ Image   | Tesseract OCR                  |
| 🎙️ Audio   | OpenAI Whisper                 |
| 📋 Metadata | Passed through to the response |

This allows the same security pipeline to analyze prompts regardless of their original format.

---

### 3. 🧹 Text Preprocessing

The extracted text is cleaned before classification.

Currently, the pipeline performs **emoji removal** using a regular-expression based preprocessing function.

```text
 Raw Input
     ↓
 Extracted Text
     ↓
 Emoji Removal
     ↓
 Cleaned Text
```

---

### 4. 🛡️ Layer 1 — Heuristic Security Filter

The first security layer uses predefined **regular-expression patterns and rules** to quickly identify potentially unsafe prompts. 

It checks for:

* Suspicious financial/payment requests
* OTP and account-verification requests
* URLs and suspicious links
* Urgency-based language
* Prompt-injection markers
* Obfuscation patterns
* Whitelisted/structured content

The filter calculates a risk score:

```text
 0 score       → LOW
 1 score       → MEDIUM
 2+ score      → HIGH
```

Only prompts classified as **LOW risk** are passed to the second layer.

This provides a fast first-level security check before running the ML model. ⚡

---

### 5. 🧠 Layer 2 — DistilBERT Classification

Prompts that pass the heuristic layer are analyzed by a locally stored **DistilBERT sequence-classification model**.

The model:

1. Tokenizes the cleaned prompt.
2. Processes it using DistilBERT.
3. Generates classification logits.
4. Applies Softmax to obtain probabilities.
5. Selects the class with the highest confidence.

The current labels are:

```text
0 → Safe
1 → Unsafe
```

The API returns both the prediction and confidence score.

---

### 6. 🤖 Gemini Response Layer

Only prompts classified as **Safe by both security layers** are sent to Gemini.

```text
🛡️ Heuristic
      ↓
   ✅ Safe
      ↓
🧠 DistilBERT
      ↓
   ✅ Safe
      ↓
🤖 Gemini 2.5 Flash
      ↓
💬 Generate Response
```

If either security layer detects an unsafe prompt, the request is blocked and Gemini is **not called**. 🚫

---

### 7. ⏱️ Performance Monitoring

The API measures execution time for different stages of the pipeline:

*  OCR processing time
*  Audio transcription time
*  Heuristic filtering time
*  DistilBERT inference time
*  Gemini response time
*  Total request processing time

This makes it possible to evaluate the performance and latency of each component.

---

## 🔄 End-to-End Flow

```text
📥 Input
   │
   ├── 📝 Text ───────────────┐
   │                          │
   ├── 🖼️ Image → OCR ────────┤
   │                          ▼
   └── 🎙️ Audio → Whisper → Cleaned Text
                              │
                              ▼
                     🛡️ Heuristic Filter
                              │
                       ┌──────┴──────┐
                       │             │
                    🚨 Unsafe       ✅ Safe
                       │             │
                       ▼             ▼
                    🚫 BLOCK   🧠 DistilBERT
                                     │
                              ┌──────┴──────┐
                              │             │
                           🚨 Unsafe       ✅ Safe
                              │             │
                              ▼             ▼
                           🚫 BLOCK   🤖 Gemini 2.5 Flash
                                             │
                                             ▼
                                      💬 Final Response
```

## 📡 API Response

The `/ingest` endpoint returns information including:

* `request_id`
* `final_text`
*  Heuristic risk and score
*  DistilBERT prediction
*  Model confidence
*  OCR/ASR/model/Gemini execution times
*  `safe_to_send`
*  Gemini response
*  Original image filename
*  Original audio filename

This provides both the **security decision** and **performance information** for every request.

## 🧰 Technology Stack

*  **Python**
*  **FastAPI**
*  **PyTorch**
*  **Hugging Face Transformers**
*  **DistilBERT**
*  **Tesseract OCR**
*  **OpenAI Whisper**
*  **Google Gemini API**
*  **PIL/Pillow**
*  **Regex**
*  **python-dotenv**
*  **Uvicorn**

## 🔐 Security Pipeline Summary

The system follows a **defense-in-depth approach**:

```text
🌐 Multimodal Input
        ↓
📄 Text Extraction
        ↓
🧹 Preprocessing
        ↓
🛡️ Heuristic Security Layer
        ↓
🧠 DistilBERT ML Layer
        ↓
🤖 Gemini
```

By combining **rule-based filtering + ML classification**, the system provides multiple security checkpoints before user input reaches the generative AI model. 🔐🤖
