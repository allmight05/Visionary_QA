# Vision Q&A Bot

A simple Visual Question Answering application powered by the state-of-the-art Qwen2-VL 2 B-parameter model quantized to 8 bits, with a Gradio web interface for easy image + text interaction.

<kbd>
  <a href="https://www.youtube.com/watch?v=0rbg9O6M0Sk" target="_blank">
    <img src="https://img.youtube.com/vi/0rbg9O6M0Sk/maxresdefault.jpg" alt="YouTube Demo" width="640">
  </a>
</kbd>

## Features

- **Qwen2-VL-2B** loaded in 8-bit mode for efficient inference on consumer GPUs.  
- **Gradio** frontend for uploading images, entering questions, and viewing answers in real time.  
- Simple, modular code for easy extension to other models or UIs.

## Getting Started

### Prerequisites

- Python 3.8+  
- CUDA 11.7+ (optional, for GPU)  
- Git

### Installation

1. Clone this repository (without forking if desired):  
   ```bash
   git clone https://github.com/YOUR_USERNAME/vision-qa-bot.git
   cd vision-qa-bot
