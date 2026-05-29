# Realization of a Model-Agnostic Explainable AI Method for Audio Classification

## Overview

This repository contains the codebase and experimental framework for evaluating model-agnostic Explainable AI (XAI) methods—specifically RISE, LIME, and Grad-CAM—applied to audio classification tasks.

## Prerequisites

* Python 3.9+
* Virtual environment manager (e.g., `venv` or `conda`)

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd research-thesis-codebase
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

| Metric | Model | LIME | Grad-CAM | RISE (Original) | RISE-WAVE | RISE-SPEC | RISE-AUDIO |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ins(%) ⬆** | **ACDNet (1D)** | 47.31 | -- | 51.80 | 60.03 | -- | **77.39** |
| | **Wav2Vec2 (1D)** | 71.34 | -- | 70.35 | 78.54 | -- | **92.53** |
| | **ResNet50 (2D)** | 78.90 | 76.25 | 80.63 | -- | **82.02** | 75.88 |
| | **HTS-AT (2D)** | 63.28 | -- | 56.61 | -- | 57.26 | **74.29** |
| **Del(%) ⬇** | **ACDNet (1D)** | 15.67 | -- | 27.91 | 32.28 | -- | **6.99** |
| | **Wav2Vec2 (1D)** | **24.62** | -- | 33.53 | 40.45 | -- | 24.96 |
| | **ResNet50 (2D)** | 29.50 | 30.69 | 31.13 | -- | 22.26 | **20.58** |
| | **HTS-AT (2D)** | 26.45 | -- | 24.05 | -- | **19.91** | 36.72 |
| **OA(%) ⬆** | **ACDNet (1D)** | 31.64 | -- | 23.89 | 27.75 | -- | **70.40** |
| | **Wav2Vec2 (1D)** | 46.72 | -- | 36.82 | 38.09 | -- | **67.57** |
| | **ResNet50 (2D)** | 49.40 | 45.56 | 49.50 | -- | **59.76** | 55.30 |
| | **HTS-AT (2D)** | 36.83 | -- | 32.56 | -- | 37.35 | **37.57** |
