# 🧠 Vision Question Answering using Qwen2-VL 2B

This repository contains an implementation of a **Vision-Language Question Answering** system using the **Qwen2-VL 2B** model, a state-of-the-art open-source multimodal model capable of understanding both images and text.

The model is efficiently loaded in **8-bit precision** to reduce memory usage and boost inference speed. A **Gradio-powered interface** enables easy interaction with the model via your browser.

---

## 📽️ Demo

Experience the model in action:

[![Watch on YouTube](https://img.youtube.com/vi/0rbg9O6M0Sk/hqdefault.jpg)](https://www.youtube.com/watch?v=0rbg9O6M0Sk)

---

## 📚 Table of Contents

- [Project Overview](#-project-overview)
- [Model Details](#-model-details)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Example Use Cases](#-example-use-cases)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Project Overview

This project is a demonstration of Vision Question Answering (VQA) using a modern vision-language model. Given an image and a natural language question, the model attempts to understand the context of the image and provide a precise and relevant response.

---

## 🧠 Model Details

- **Model Name:** Qwen2-VL-2B  
- **Multimodal:** Vision + Text  
- **Loaded with:** `AutoModelForVision2Seq` (from Hugging Face)  
- **Quantization:** 8-bit using `bitsandbytes`  
- **Tokenizer:** QwenTokenizer  
- **Image Processor:** QwenImageProcessor

---

## 🛠️ System Architecture

```mermaid
graph LR
    A[User Uploads Image & Asks Question] --> B[Gradio UI]
    B --> C[Preprocessing (Tokenizer + ImageProcessor)]
    C --> D[Qwen2-VL-2B (8bit) Model]
    D --> E[Model Output]
    E --> F[Gradio Display]
