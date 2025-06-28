# 🔖 v1.2.0 – Query-Based Video Summarization & Captioning (with Optional Audio Summarization)

This release includes the full implementation of a **Query-Based Video Summarization and Captioning System**, extended with optional **audio-based summarization** support.

---

## 📌 Overview

In this study, **query-based video-to-video summarization** and **captioning of the summarized video segments** were performed using neural models. The system integrates vision-language models and transformer-based architectures to generate multimodal summaries and textual descriptions based on a given natural language query.

---

## 🎯 Core Components

### 🧠 Video Summarization

- The **query-based summarization model** is trained using the notebook `qubvis_transformer.ipynb` located under the `/models/` directory.
- Dataset structure (`video_id`, `frame_features`, `vsum_onehot`, etc.) can be examined in the data preprocessing section of the notebook.

### 📝 Video Captioning

- Caption generation for the summarized videos is handled by the model trained using `vision_language_model.ipynb` in the `/model/` directory.
- Model loading logic is encapsulated in:
  - `QBSumModel2.py` (for summarization)
  - `VidCapModel.py` (for captioning)

### 🧾 Query-Based Text Summarization (Audio Transcript)

- The optional transcript-based summarization model is implemented in `QMSUM_BERT.ipynb` using supervised contrastive learning on the QMSum dataset.
- This model is used to extract timestamp-aligned text segments relevant to the query from Whisper-generated transcriptions.

---

## 🔁 API Integration

The `QBMultimodal.py` file provides a complete API implementation for performing inference with all models. This API:

- Loads trained models via Google Drive paths listed in `/models/readme.txt`.
- Accepts the following input parameters:
  - A **YouTube video URL** to be summarized
  - A **Query string** describing what the summary should focus on
  - (Optional) A **checkbox parameter** to enable **audio-based summarization** using Whisper and QMSUM-BERT

> 🔊 **Audio Summarization (Optional)**: If enabled, the API will extract audio transcriptions via Whisper and perform query-based extractive summarization on transcript segments using the QMSUM-based model. These segments are timestamp-aligned and used to enhance or validate the visual summary.

---

## 🌐 Web Interface

- The `/templates/index.html` file provides a minimal user interface for testing.
- Users can enter a **YouTube URL** and a **Query**.
- The interface returns:
  - The **query-focused summary video**
  - The **generated captions** for the summary
  - (If enabled) the **audio-based summarized segments**

---

## 📁 File Overview

| File/Directory                        | Description                                                    |
|--------------------------------------|----------------------------------------------------------------|
| `/models/qubvis_transformer.ipynb`   | Video summarization training                                   |
| `/model/vision_language_model.ipynb` | Video captioning training                                      |
| `/models/QMSUM_BERT.ipynb`           | Query-based text summarization on Whisper transcriptions       |
| `QBSumModel2.py`                     | Summarization model loader                                     |
| `VidCapModel.py`                     | Captioning model loader                                        |
| `QBMultimodal.py`                    | Flask API for multimodal summarization and captioning          |
| `/templates/index.html`              | Web interface for input & results                              |

---
