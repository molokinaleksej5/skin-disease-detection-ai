# Skin Disease Detection AI

End-to-end deep learning system for skin lesion classification using ResNet50, PyTorch, Grad-CAM explainability and Android deployment.

> Research prototype. Not intended for real medical diagnosis.

## Project Overview

Skin cancer is one of the most common oncological diseases worldwide. Early detection is especially important for melanoma because delayed diagnosis can significantly reduce treatment effectiveness.

This project implements an AI-based system for dermoscopic image classification. The system analyzes a skin lesion image, predicts one of eight diagnostic classes and provides probability scores for each class.

The project covers the full ML pipeline:

- dataset analysis and preprocessing;
- class imbalance handling;
- ResNet-based model training;
- model comparison and evaluation;
- Grad-CAM visual explanation;
- Android application integration.

## Problem

Skin lesion classification is challenging because different diseases may have visually similar patterns. Benign nevi and melanoma can share similar color, border and texture characteristics, especially at early stages.

The main goal of this project is to build a model that can classify dermoscopic images and provide interpretable predictions for medical image analysis research.

## Dataset

The project uses the ISIC 2019 dataset for skin lesion classification.

The task is formulated as multi-class classification with 8 classes:

| Code | Class |
|---|---|
| AK | Actinic Keratosis |
| BCC | Basal Cell Carcinoma |
| BKL | Benign Keratosis-like Lesions |
| DF | Dermatofibroma |
| MEL | Melanoma |
| NV | Melanocytic Nevus |
| SCC | Squamous Cell Carcinoma |
| VASC | Vascular Lesions |

## Key Dataset Challenges

The dataset has a strong class imbalance. The NV class is significantly larger than rare classes such as DF and VASC.

To handle this problem, the project uses:

- stratified train / validation / test split;
- image resizing to 224×224;
- RGB normalization;
- data augmentation;
- class weights;
- Focal Loss.

## Model Architecture

The final model is based on ResNet50 with transfer learning.

ResNet50 was selected after comparison with ResNet18 and ResNet34. The deeper architecture showed better ability to extract complex visual features such as lesion shape, border irregularity, color heterogeneity and texture changes.

```text
Input Image
    ↓
Preprocessing
    ↓
ResNet50 Feature Extractor
    ↓
Modified MLP Classification Head
    ↓
Softmax Probabilities
    ↓
Predicted Skin Lesion Class
```

## Training Strategy

The training pipeline includes:

- transfer learning;
- modified classification head;
- AdamW optimizer;
- Focal Loss;
- class weights;
- learning rate scheduling;
- validation monitoring;
- evaluation on a separate test set.

## Results

The final model achieved the following test results:

| Metric | Value |
|---|---:|
| Accuracy | 84.41% |
| Weighted F1-score | 0.8401 |
| Macro F1-score | 0.7440 |
| MEL Precision | 0.8238 |
| MEL Recall | 0.6819 |
| MEL F1-score | 0.7462 |
| MEL ROC AUC | 0.9358 |

The model showed strong overall classification quality and good separation ability for melanoma according to ROC AUC.

## Explainability

Grad-CAM was used to visualize which image regions influenced the model prediction.

This is important for medical image analysis because the final prediction should not be treated as a black-box result. Visual explanation helps understand whether the model focuses on the lesion area or irrelevant background artifacts.

## Android Application

The trained model was integrated into an Android application prototype.

The application allows the user to:

- select a skin lesion image;
- run model inference;
- view the predicted class;
- view probability scores for all classes.

## Repository Structure

```text
skin-disease-detection-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── model.py
│   ├── dataset.py
│   └── gradcam.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── roc_curve_mel.png
│   ├── pr_curve_mel.png
│   └── gradcam_examples.png
│
├── android/
│   └── screenshots/
│
├── models/
│   └── README.md
│
└── docs/
    └── thesis_summary.md
```

## Installation

```bash
git clone https://github.com/molokinaleksej5/skin-disease-detection-ai.git
cd skin-disease-detection-ai

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Inference Example

```bash
python src/predict.py --image path/to/image.jpg
```

Example output:

```json
{
  "predicted_class": "MEL",
  "confidence": 0.5629,
  "probabilities": {
    "AK": 0.0001,
    "BCC": 0.0000,
    "BKL": 0.0259,
    "DF": 0.0000,
    "MEL": 0.5629,
    "NV": 0.4110,
    "SCC": 0.0001,
    "VASC": 0.0000
  }
}
```

## Tech Stack

- Python
- PyTorch
- TorchVision
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Grad-CAM
- Android
- Git

## Limitations

This project is a research and educational prototype. It is not a certified medical device and must not be used for real diagnosis.

Current limitations:

- model performance depends on image quality;
- rare classes remain harder to classify;
- visually similar classes may produce close probabilities;
- clinical validation was not performed;
- the system should be used only as a decision-support prototype.

## Future Improvements

- add FastAPI inference service;
- add Docker support;
- add MLflow experiment tracking;
- improve rare class performance;
- add live demo;
- expand Android application functionality;
- test additional architectures such as EfficientNet and ConvNeXt.

## Author

Alexey Molokin  
ML Engineer / Python Developer
