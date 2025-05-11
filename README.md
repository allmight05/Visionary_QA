# 🧠 Vision Question Answering using Qwen2-VL 2B

This repository contains an implementation of a **Vision-Language Question Answering** system powered by the **Qwen2-VL 2B** model, loaded in **8-bit precision** for efficient inference, and deployed via a **Gradio** interface.

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


🧪 Installation

    Clone the repository

git clone https://github.com/your-username/vision-qa-bot.git
cd vision-qa-bot

Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\activate

Install dependencies

    pip install -r requirements.txt

    Note: Make sure you have bitsandbytes, transformers, torch, and gradio installed.

▶️ Running the App

python app.py

Then open your browser to http://localhost:7860.
🖼️ Example Use Cases
Image Type	Sample Question	Expected Behavior
Street Scene	“How many people are walking?”	Count visible pedestrians
Kitchen Interior	“What color is the countertop?”	Identify countertop color
Document Page	“What is the title of this paper?”	Extract header text
⚙️ Advanced Usage

    Switch Precision: Toggle between 8-bit and full precision in the load_model() function.

    Batch Inference: Extend the Gradio interface to accept multiple images.

    Custom Models: Replace Qwen2-VL with any Hugging Face–compatible multimodal model.

🤝 Contributing

Contributions are welcome! Please fork the repository and open a pull request. For major changes, open an issue first to discuss what you’d like to change.
