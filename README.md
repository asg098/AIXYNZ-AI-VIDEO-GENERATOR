# AIXYNZ-AI-VIDEO-GENERATOR(use at ypur own risk)
A open sourced model used to make text to video 
# 🎬 AIXYNZ AI Video Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Generate videos from text using AI. Free and open source.**

A holiday project exploring text-to-video generation with Stable Diffusion.

## ✨ Features

- 🎨 Text-to-video generation
- ⚡ GPU & CPU support
- 🎞️ Adjustable duration, FPS, quality
- 📊 Real-time progress tracking
- 🖼️ Video gallery
- 🎯 Easy-to-use interface

## 🚀 Quick Start

### Google Colab (Recommended - Free GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `main.py`
3. Run the cell
4. Click the Gradio link
5. Start creating videos!

### Local Installation
```bash
git clone https://github.com/asg098/AIXYNZ-AI-VIDEO-GENERATOR.git
cd AIXYNZ-AI-VIDEO-GENERATOR
pip install -r requirements.txt
python main.py
```

## 📋 Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional (faster generation)
- ~5GB space for model (first run only)

## 💻 Usage

1. **Write a prompt**: "Sunset over ocean with golden waves"
2. **Set parameters**: Duration, FPS, quality
3. **Generate**: Click the button
4. **Download**: Save your video

### Example Prompts

- "Northern lights dancing over snowy mountains"
- "Dragon flying through storm clouds"
- "Colorful paint drops in water slow motion"

## 🎯 Generation Speed

| Settings | GPU (T4) | CPU |
|----------|----------|-----|
| 5s, 15fps, Fast | ~3 min | ~15 min |
| 10s, 15fps, Best | ~12 min | ~45 min |

## 🛠️ Tech Stack

- Stable Diffusion v1.5
- PyTorch + Diffusers
- Gradio UI
- OpenCV + ImageIO

## 📁 Project Structure
