# Realization of a Model-Agnostic Explainable AI Method for Audio Classification

## Overview

This repository contains the codebase and experimental framework for evaluating model-agnostic Explainable AI (XAI) methods—specifically RISE, LIME, and Grad-CAM—applied to audio classification tasks.

## Prerequisites

* Python 3.9+
* Virtual environment manager (e.g., `venv` or `conda`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shivam1423/Audio-XAI.git
cd Audio-XAI
```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `data/`: Directory for raw, interim, and processed audio datasets.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and rapid prototyping.
- `src/`: Core Python modules for data loading, preprocessing, model architecture, and XAI implementations.
- `experiments/`: YAML configuration files defining parameters for different experimental runs.
- `outputs/`: Model checkpoints, weights, and training logs.
- `results/`: Generated evaluation tables, metrics, and XAI visual outputs (plots/figures).
- `thesis_manuscript/`: LaTeX source files and bibliography for the final document.

## Usage

### 1. Data Preparation
Place the raw audio datasets in `data/raw/` and execute the preprocessing pipeline to extract features (e.g., spectrograms):
```bash
python src/preprocessing.py
```

### 2. Model Training

Train the baseline audio classification models using a specified experiment configuration:

```bash
python src/train.py --config experiments/exp_01_baseline.yaml
```

### 3. Evaluation and XAI Extraction

Evaluate the trained models and generate interpretations using the selected XAI methods:

```bash
python src/evaluate.py --model_path outputs/models/best_model.pth --xai_method RISE
```

## Results

| Model (Type) | Method | Ins(%) ⬆ | Del(%) ⬇ | OA(%) ⬆ |
| :--- | :--- | :--- | :--- | :--- |
| **ACDNet (1D)** | LIME | 47.31 | 15.67 | 31.64 |
| | RISE (Original) | 51.80 | 27.91 | 23.89 |
| | RISE-WAVE | 60.03 | 32.28 | 27.75 |
| | RISE-AUDIO | **77.39** | **6.99** | **70.40** |
| **Wav2Vec2 (1D)** | LIME | 71.34 | **24.62** | 46.72 |
| | RISE (Original) | 70.35 | 33.53 | 36.82 |
| | RISE-WAVE | 78.54 | 40.45 | 38.09 |
| | RISE-AUDIO | **92.53** | 24.96 | **67.57** |
| **ResNet50 (2D)** | LIME | 78.90 | 29.50 | 49.40 |
| | Grad-CAM | 76.25 | 30.69 | 45.56 |
| | RISE (Original) | 80.63 | 31.13 | 49.50 |
| | RISE-SPEC | **82.02** | 22.26 | **59.76** |
| | RISE-AUDIO | 75.88 | **20.58** | 55.30 |
| **HTS-AT (2D)** | LIME | 63.28 | 26.45 | 36.83 |
| | Grad-CAM | -- | -- | -- |
| | RISE (Original) | 56.61 | 24.05 | 32.56 |
| | RISE-SPEC | 57.26 | **19.91** | 37.35 |
| | RISE-AUDIO | **74.29** | 36.72 | **37.57** |
