# XAI-TransMed: Explainable AI for Alzheimer's Detection and Progression Rate

XAI-TransMed is an advanced decision-support platform that combines deep learning, sequence modeling, and traditional machine learning to classify dementia stages from brain MRI scans and estimate disease progression rates over time.

---

## 🚀 Key Features

* **Dementia Stage Classification:** Classifies single MRI scans into 4 cognitive categories using a Vision Transformer (`vit_small_patch16_224`).
* **Explainable AI (XAI) Visualizations:** Dynamically overlays **Attention Rollout** or **Grad-CAM** heatmaps onto structural scans to highlight diagnostic regions of interest.
* **Clinical Progression Tracking:** Computes a statistical Hazard Ratio representing long-term progression risks utilizing Cox-proportional-based clinical indexing metrics.
* **Longitudinal MRI Modeling (LSTM):** Evaluates sequences of historical scans chronologically to forecast future Clinical Dementia Rating (CDR) baseline shifts.
* **Hybrid Next-Visit MMSE Prediction:** Passes extracted ViT latent tokens along with standard demographic or structural indicators through a trained **XGBoost Regressor**.

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
