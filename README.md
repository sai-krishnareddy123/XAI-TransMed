# XAI-TransMed

## Explainable AI Transformer-Based Medical Model for Alzheimer's Disease Detection and Progression Prediction

XAI-TransMed is an Explainable AI-based decision-support and research prototype designed to analyze MRI images along with clinical and demographic information for Alzheimer's disease assessment.

The system combines a Vision Transformer (ViT) for dementia-stage classification with machine-learning models for progression-related prediction. Explainable AI techniques such as attention maps, attention rollout, and Grad-CAM are used to make the model's predictions more interpretable.

## Problem Statement

Alzheimer's disease is a progressive neurological disorder that affects memory, thinking, and daily activities. Analyzing MRI scans and clinical information can be complex, while AI models may provide predictions without clearly explaining the factors influencing their decisions.

XAI-TransMed aims to support more interpretable AI-based assessment by combining MRI analysis, clinical information, progression prediction, and visual explanations.

## Key Features

- MRI-based dementia-stage classification
- Vision Transformer (ViT) for feature extraction and classification
- Multimodal integration of MRI and clinical/demographic information
- Four-stage dementia classification:
  - Non-Demented
  - Very Mild
  - Mild
  - Moderate
- Progression-related prediction using XGBoost and Cox regression
- Explainability using:
  - Attention Maps
  - Attention Rollout
  - Grad-CAM
- Class probability visualization
- Progression-related prediction output
- Prototype interface for model interaction

## System Workflow

MRI Image + Clinical Data
        ↓
Preprocessing
        ↓
Vision Transformer (ViT)
        ↓
Feature Extraction
        ↓
Dementia Stage Classification
        ↓
Progression Prediction
        ↓
Explainable AI
        ↓
Prediction + Probabilities + Visual Explanation

## Technologies Used

- Python
- PyTorch / TensorFlow
- Vision Transformer (ViT)
- XGBoost
- Cox Regression
- Grad-CAM
- Attention Maps / Attention Rollout
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- FastAPI
- HTML
- CSS
- JavaScript

## Dataset

The project uses the OASIS dataset for MRI-based Alzheimer's disease analysis.

## Results

The research prototype evaluates dementia-stage classification and progression-related prediction using standard machine-learning evaluation metrics.

The project presentation reports an overall classification accuracy of 98.6% on the evaluated OASIS test dataset.

## Explainable AI

A key component of XAI-TransMed is interpretability. Attention-based visualizations and Grad-CAM are used to highlight regions of the MRI that contribute to model predictions.

This helps users better understand the model output rather than relying only on a predicted class.

## Responsible AI

XAI-TransMed is intended as a decision-support and research prototype and is not intended to replace healthcare professionals.

Important considerations include:

- Human oversight
- Transparency and interpretability
- Patient-data privacy
- Responsible handling of medical information
- Awareness of dataset limitations and potential bias
- Further validation before real-world clinical use

## Future Scope

Future development may include:

- Larger and more diverse datasets
- Additional clinical variables
- Improved longitudinal progression prediction
- More extensive model validation
- Enhanced explainability techniques
- Further testing in realistic healthcare workflows

## Project Alignment

This project aligns with:

**SDG 3 – Good Health and Well-being**

The project explores how responsible and explainable AI can support healthcare professionals in understanding Alzheimer's disease assessment and progression-related predictions.

## Disclaimer

XAI-TransMed is an academic/research prototype. Its outputs should not be considered a medical diagnosis or a substitute for professional medical advice.

## Authors

**Voladri Sai Krishnareddy**

CSE, Vardhaman College of Engineering


---

## 📂 Project Architecture

```text
XAI_CODE/
│
├── backend/                    # FastAPI Application server & routing pipelines
│   ├── main.py                 
│   ├── model.py                
│   ├── models.py               
│   ├── explain.py              
│   ├── requirements.txt        
│   └── *.pth / *.pkl           
│
├── fronted/                    # Frontend UI application directory
│   ├── index.html              
│   ├── script.js               
│   └── style.css               
│
└── README.md                   # Project documentation index
