# 🧠 Vision Question Answering using Qwen2-VL 2B

This repository contains an implementation of a **Vision-Language Question Answering** system powered by the **Qwen2-VL 2B** model, loaded in **8-bit precision** for efficient inference, and deployed via a **Gradio** interface.

---

## 📽️ Demo

Experience the model in action:

[![Watch on YouTube](https://img.youtube.com/vi/0rbg9O6M0Sk/hqdefault.jpg)](https://www.youtube.com/watch?v=0rbg9O6M0Sk)

---

## 📚 Table of Contents

- [Project Overview]
- [Model Details] 
- [System Architecture]
- [Installation]
- [Running the App] 

---

## 🚀 Project Overview

This project demonstrates a modern Vision Question Answering (VQA) pipeline. Users upload an image and ask a natural language question; the system retrieves and processes the image, then returns a concise answer.

---

## 🧠 Model Details

- **Model**: Qwen2-VL-2B (2 billion parameters)  
- **Quantization**: 8-bit via `bitsandbytes`  
- **Tokenizer**: QwenTokenizer  
- **Image Processor**: QwenImageProcessor  
- **Interface**: Gradio for image+text input and answer display  

---

## 🛠️ System Architecture

```mermaid
graph LR
    A["User Uploads Image & Asks Question"] --> B["Gradio UI"]
    B --> C["Preprocessing: Tokenizer and Image Processor"]
    C --> D["Qwen2-VL-2B (8-bit) Model"]
    D --> E["Model Output"]
    E --> F["Gradio Display"]

```
---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/allmight05/Visionary_QA.git
cd Visionary_QA

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```
---

## ▶️ Running the App

```bash
# 1. Start the application
python app.py

# 2. Access the Gradio interface
#    Open your browser and go to:
#    http://localhost:7860
```
---
