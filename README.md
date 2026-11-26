# Car Model Classification System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

A full-stack computer vision application for car model classification built on **ResNet18** in PyTorch, served via a **FastAPI** backend, and delivered through an interactive **Streamlit** web interface with **Docker Compose** profiles for both CPU-only and NVIDIA GPU execution.

---

## 📌 Overview

This project provides an end-to-end machine learning workflow for classifying car models:
- **Core ML:** **ResNet18** model in PyTorch (`cv/src/train.py`) supporting automatic device selection (`cuda` when available, falling back to `cpu`).
- **Backend API:** **FastAPI** service (`cv/src/api.py`, with `backend` and `backend-gpu` Compose services) exposing validated inference endpoints.
- **Frontend:** **Streamlit** web application (`app.py`) for uploading images, viewing predictions, and interacting with the backend API.

---

## 🛠️ Tech Stack & Architecture

- **Machine Learning & Deep Learning:** PyTorch (`torch==2.6.0+cu124`), Torchvision (`ResNet18`), Pillow (`PIL`)
- **Backend / API:** FastAPI, Uvicorn, Python-Multipart
- **Frontend:** Streamlit
- **Containerization:** Docker, Docker Compose (`profiles: ["gpu"]`)

---

## 📂 Project Architecture

```text
Car-Model-Classification-CNN/
│
├── cv/                     # Core computer vision codebase and artifacts
│   ├── model/              # Saved model weights (e.g., best_model.pth)
│   └── src/
│       ├── train.py        # ResNet18 training script
│       └── api.py          # FastAPI inference server
│
├── docker/                 # Docker infrastructure and dependencies
│   ├── Dockerfile          # Container image definition
│   ├── docker-compose.yml  # CPU (default) and GPU (--profile gpu) services
│   └── requirements.txt    # Python dependencies
│
├── app.py                  # Streamlit frontend application
└── README.md
```

---

## 🚀 Getting Started & Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Chumus2/Car-Model-Classification-CNN.git
cd Car-Model-Classification-CNN
```

2. **Prepare the Dataset:**
Download and Prepare the Dataset: https://www.kaggle.com/datasets/lightangle/car-model-image-classification
Downloaded dataset name ***data*** and put into cv directory.


### Option 1: Quick Start with Docker Compose (Recommended)
The easiest way to run the entire application in isolated environments is using Docker Compose.

3. **Run with Docker Compose (CPU — Default, No GPU Required):**
```bash
cd docker
docker compose up --build
```

4. **Run with Docker Compose (CPU — Default, No GPU Required):**
```bash
cd docker
docker compose --profile gpu up --build
```

### **Option 2: Local Setup (Without Docker)**
If you prefer to run the application directly on your host machine without containers, follow these steps:

3. **Set Up a Python Virtual Environment:**
```bash
python -m venv venv
.\venv\Scripts\Activate
```

4. **Install Dependencies:**
```bash
pip install --upgrade pip
pip install -r docker/requirements.txt
```

5. **Train the Model:**
Ensure the output directory in ***cv*** exists and run ***train.py*** to generate model weights:
```bash
python cv/src/train.py
```

6. **Run the Backend and Frontend Services:**
Open two separate terminal windows (both with your virtual environment activated) to run the microservices concurrently:

- *Start the FastAPI Backend:*
```bash
uvicorn cv.src.api:app --host 0.0.0.0 --port 8000
```

- *Start the Streamlit Frontend:*
```bash
streamlit run app.py
```

**Access the Application:**
Website: http://localhost:8501
FastAPI documentation: http://localhost:8000/docs