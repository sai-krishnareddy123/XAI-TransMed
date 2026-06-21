# # # =============================================================================
# # # main.py – xAI-transmed API with region scores and Grad-CAM
# # # =============================================================================

# # from fastapi import FastAPI, File, UploadFile, Query
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse
# # from PIL import Image
# # import io
# # import base64
# # import logging

# # from model import ViTClassifier
# # from explain import (
# #     attention_rollout,
# #     apply_heatmap,
# #     compute_region_attention,
# #     create_demo_regions,
# #     ViTGradCAM,
# # )

# # # Setup logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # app = FastAPI(title="xAI-transmed API")

# # # Allow frontend to access the API
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Load the model once at startup
# # classifier = ViTClassifier(model_path="best_vit_model.pth", device="cuda")

# # # Initialize Grad-CAM (targeting the last transformer block)
# # gradcam = ViTGradCAM(classifier.model, target_layer_name='blocks.11.norm1')

# # # Store region masks as an app attribute to compute only once
# # app.state.region_masks = None

# # @app.post("/predict")
# # async def predict(
# #     file: UploadFile = File(...),
# #     method: str = Query("rollout", regex="^(rollout|gradcam)$")  # choose explanation method
# # ):
# #     """
# #     Receive an MRI image and return:
# #     - Prediction (class, confidence, probabilities)
# #     - Attention heatmap (as base64 PNG)
# #     - Region‑wise attention scores
# #     """
# #     try:
# #         # Read and convert image
# #         contents = await file.read()
# #         image = Image.open(io.BytesIO(contents)).convert('RGB')
# #         logger.info(f"Received image: size {image.size}, format {image.format}")

# #         # Get prediction
# #         pred = classifier.predict(image)
# #         logger.info(f"Prediction: {pred}")

# #         # Prepare tensor for explanation
# #         img_tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)

# #         # Generate explanation (heatmap)
# #         if method == "gradcam":
# #             # Get class index from prediction (map class name to index)
# #             class_idx = classifier.class_names.index(pred['predicted_class'])
# #             mask = gradcam.generate_heatmap(img_tensor, class_idx=class_idx)
# #         else:  # rollout
# #             mask = attention_rollout(classifier.model, img_tensor)

# #         # Apply heatmap overlay
# #         heatmap_img = apply_heatmap(image, mask)

# #         # ---- Region attention scores ----
# #         # Get image size (height, width)
# #         img_size = (image.height, image.width)

# #         # Initialize region masks on first request
# #         if app.state.region_masks is None:
# #             # For demo, use simple rectangular masks; replace with actual segmentation if available
# #             app.state.region_masks = create_demo_regions(img_size)
# #             logger.info("Demo region masks created.")

# #         # Compute region scores
# #         region_scores = compute_region_attention(mask, img_size, app.state.region_masks)
# #         logger.info(f"Region scores: {region_scores}")

# #         # Convert images to base64 for frontend
# #         buffered_orig = io.BytesIO()
# #         image.save(buffered_orig, format="PNG")
# #         orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode()

# #         buffered_heat = io.BytesIO()
# #         heatmap_img.save(buffered_heat, format="PNG")
# #         heatmap_base64 = base64.b64encode(buffered_heat.getvalue()).decode()

# #         return JSONResponse({
# #             "success": True,
# #             "prediction": pred,
# #             "original_image": orig_base64,
# #             "heatmap": heatmap_base64,
# #             "region_scores": region_scores,
# #             "method_used": method
# #         })

# #     except Exception as e:
# #         logger.exception("Prediction error")
# #         return JSONResponse({"success": False, "error": str(e)})

# # @app.get("/health")
# # async def health():
# #     return {"status": "ok", "model_loaded": True}

# # =============================================================================
# # main.py – xAI-transmed API (Classification + Progression)
# # =============================================================================

# # =============================================================================
# # main.py – xAI-transmed API (Classification + Progression)
# # =============================================================================

# # =============================================================================
# # main.py – xAI-transmed API (Classification + Progression)
# # =============================================================================



# # =============================================================================
# # main.py – xAI-transmed API (Classification + Progression) with LSTM Diagnostics
# # =============================================================================

# # =============================================================================
# # main.py – xAI-transmed API (Classification + Progression)
# # With fix for LSTM loading via __main__ attachment
# # =============================================================================
# # import joblib
# # import xgboost as xgb
# # from fastapi import Form
# # import os
# # import io
# # import base64
# # import pickle
# # import logging
# # from typing import List, Optional

# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import pandas as pd
# # from PIL import Image
# # from fastapi import FastAPI, File, UploadFile, Query
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse
# # from pydantic import BaseModel

# # # Local modules (must exist in the same directory)
# # from model import ViTClassifier
# # from explain import (
# #     attention_rollout,
# #     apply_heatmap,
# #     compute_region_attention,
# #     create_demo_regions,
# #     ViTGradCAM,
# # )

# # # -------------------- Setup logging --------------------
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # -------------------- FastAPI app --------------------
# # app = FastAPI(title="xAI-transmed API")

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # -------------------- Device --------------------
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # logger.info(f"Using device: {device}")

# # # -------------------- Load classification model --------------------
# # try:
# #     classifier = ViTClassifier(model_path="best_vit_model.pth", device=device)


# #     #changed
# #     #     # ---- TEMPORARY: Print layer names for Grad-CAM ----
# #     # print("\n=== Available layer names for Grad-CAM ===")
# #     # for name, _ in classifier.model.named_modules():
# #     #     if 'norm' in name or 'attn' in name:
# #     #         print(name)
# #     # print("==========================================\n")

# #     logger.info("✅ ViT classifier loaded.")
# # except Exception as e:
# #     logger.error(f"Failed to load ViT model: {e}")
# #     classifier = None

# # # -------------------- Grad-CAM (for explainability) --------------------
# # gradcam = ViTGradCAM(classifier.model, target_layer_name='blocks.11.norm1') if classifier else None

# # # -------------------- Load Cox progression model (clinical) --------------------
# # cox_model = None
# # try:
# #     with open('cox_progression_model.pkl', 'rb') as f:
# #         cox_model = pickle.load(f)
# #     logger.info("✅ Cox progression model loaded.")
# # except FileNotFoundError:
# #     logger.warning("cox_progression_model.pkl not found. Cox endpoint disabled.")
# # except Exception as e:
# #     logger.error(f"Failed to load Cox model: {e}")

# # # -------------------- LSTM progression model (imaging) --------------------
# # lstm_model = None
# # lstm_metadata = {}

# # # Define the LSTM class (must match the training architecture)
# # class ProgressionLSTM(nn.Module):
# #     def __init__(self, input_size=384, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
# #         super().__init__()
# #         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
# #         self.fc = nn.Sequential(
# #             nn.Linear(hidden_size, 64),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(64, output_size)
# #         )

# #     def forward(self, x):
# #         out, (hidden, cell) = self.lstm(x)
# #         last_hidden = hidden[-1]  # take last layer's hidden state
# #         return self.fc(last_hidden)
# # # Load XGBoost progression model
# # xgb_model = None
# # xgb_imputer = None
# # try:
# #     xgb_model = joblib.load('xgb_mmse_model.pkl')
# #     xgb_imputer = joblib.load('imputer.pkl')
# #     logger.info("✅ XGBoost progression model loaded.")
# # except FileNotFoundError:
# #     logger.warning("XGBoost model files not found. XGB endpoint disabled.")
# # # --- Crucial fix: attach the class to __main__ for pickle loading ---
# # import __main__
# # __main__.ProgressionLSTM = ProgressionLSTM

# # # --- Diagnostics and loading ---
# # lstm_file = 'lstm_progression_full.pth'
# # meta_file = 'lstm_metadata.pkl'

# # print("\n=== LSTM Model Loading Diagnostics ===")
# # if os.path.exists(lstm_file):
# #     print(f"✅ Found {lstm_file}")
# # else:
# #     print(f"❌ {lstm_file} NOT FOUND")
# # if os.path.exists(meta_file):
# #     print(f"✅ Found {meta_file}")
# # else:
# #     print(f"❌ {meta_file} NOT FOUND")

# # try:
# #     print("Attempting torch.load with weights_only=False (trusted file)...")
# #     lstm_model = torch.load(lstm_file, map_location=device, weights_only=False)
# #     lstm_model.eval()
# #     print("✅ Model loaded successfully.")

# #     with open(meta_file, 'rb') as f:
# #         lstm_metadata = pickle.load(f)
# #     print(f"✅ Metadata loaded: {lstm_metadata}")

# #     logger.info("✅ LSTM progression model loaded (with weights_only=False).")
# # except Exception as e:
# #     print(f"❌ Error: {type(e).__name__}: {e}")
# #     logger.error(f"Failed to load LSTM model: {e}")
# # print("=== End of Diagnostics ===\n")

# # # -------------------- Median values for Cox model (computed from OASIS‑2) --------------------
# # MEDIANS = {
# #     'Age': 77.0,
# #     'EDUC': 15.0,
# #     'SES': 2.0,
# #     'MMSE': 29.0,
# #     'nWBV': 0.729,
# #     'eTIV': 1470.0,
# #     'ASF': 1.194
# # }

# # # -------------------- Pydantic schema for Cox input --------------------
# # class ProgressionInput(BaseModel):
# #     Age: float
# #     EDUC: float
# #     SES: Optional[float] = None
# #     MMSE: float
# #     nWBV: float
# #     eTIV: float
# #     ASF: float

# # # -------------------- Region masks cache --------------------
# # app.state.region_masks = None

# # # ==================== ENDPOINTS ====================

# # # -------------------- 1. Classification + Explainability --------------------
# # @app.post("/predict")
# # async def predict(
# #     file: UploadFile = File(...),
# #     method: str = Query("rollout", pattern="^(rollout|gradcam)$")
# # ):
# #     """
# #     Upload a single MRI image. Returns:
# #     - Predicted class & probabilities
# #     - Attention heatmap (base64 PNG)
# #     - Region attention scores
# #     """
# #     if classifier is None:
# #         return JSONResponse({"success": False, "error": "Classifier not available."})

# #     try:
# #         contents = await file.read()
# #         image = Image.open(io.BytesIO(contents)).convert('RGB')
# #         logger.info(f"Received image: size {image.size}")

# #         # Prediction
# #         pred = classifier.predict(image)
# #         logger.info(f"Prediction: {pred}")

# #         # Prepare tensor for explanation
# #         img_tensor = classifier.transform(image).unsqueeze(0).to(device)

# #         # Generate heatmap
# #         if method == "gradcam":
# #             class_idx = classifier.class_names.index(pred['predicted_class'])
# #             mask = gradcam.generate_heatmap(img_tensor, class_idx=class_idx)
# #         else:
# #             mask = attention_rollout(classifier.model, img_tensor)

# #         heatmap_img = apply_heatmap(image, mask)

# #         # Region scores (demo masks)
# #         img_size = (image.height, image.width)
# #         if app.state.region_masks is None:
# #             app.state.region_masks = create_demo_regions(img_size)
# #         region_scores = compute_region_attention(mask, img_size, app.state.region_masks)

# #         # Convert images to base64
# #         buffered_orig = io.BytesIO()
# #         image.save(buffered_orig, format="PNG")
# #         orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode()

# #         buffered_heat = io.BytesIO()
# #         heatmap_img.save(buffered_heat, format="PNG")
# #         heatmap_base64 = base64.b64encode(buffered_heat.getvalue()).decode()

# #         return JSONResponse({
# #             "success": True,
# #             "prediction": pred,
# #             "original_image": orig_base64,
# #             "heatmap": heatmap_base64,
# #             "region_scores": region_scores,
# #             "method_used": method
# #         })

# #     except Exception as e:
# #         logger.exception("Prediction error")
# #         return JSONResponse({"success": False, "error": str(e)})


# # # -------------------- 2. Progression (Cox model, clinical) --------------------
# # @app.post("/predict_progression")
# # async def predict_progression(data: ProgressionInput):
# #     """
# #     Accept clinical features, return hazard ratio from Cox model.
# #     """
# #     if cox_model is None:
# #         return JSONResponse({"success": False, "error": "Cox model not available."})

# #     try:
# #         # Impute missing SES
# #         input_dict = data.dict()
# #         for key in MEDIANS:
# #             if input_dict.get(key) is None:
# #                 input_dict[key] = MEDIANS[key]

# #         # Create DataFrame with correct column order
# #         df_input = pd.DataFrame([input_dict])
# #         required_cols = cox_model.params_.index.tolist()
# #         df_input = df_input[required_cols]

# #         # Predict hazard ratio
# #         hazard_ratio = cox_model.predict_partial_hazard(df_input).iloc[0]

# #         return JSONResponse({
# #             "success": True,
# #             "hazard_ratio": float(hazard_ratio),
# #             "interpretation": "Values >1 indicate higher risk."
# #         })

# #     except Exception as e:
# #         logger.exception("Cox prediction error")
# #         return JSONResponse({"success": False, "error": str(e)})


# # # -------------------- 3. Progression (LSTM, imaging) --------------------
# # @app.post("/predict_progression_lstm")
# # async def predict_progression_lstm(files: List[UploadFile] = File(...)):
# #     """
# #     Upload multiple MRI images (chronological order, oldest first).
# #     Returns predicted CDR for the next visit.
# #     """
# #     if lstm_model is None:
# #         return JSONResponse({"success": False, "error": "LSTM model not available."})

# #     try:
# #         # Extract features from each image using the ViT feature extractor
# #         features = []
# #         for file in files:
# #             contents = await file.read()
# #             image = Image.open(io.BytesIO(contents)).convert('RGB')
# #             img_tensor = classifier.transform(image).unsqueeze(0).to(device)
# #             with torch.no_grad():
# #                 feat = classifier.model.forward_features(img_tensor).mean(dim=1).cpu().numpy().flatten()
# #             features.append(feat)

# #         # Pad or truncate to MAX_LEN
# #         max_len = lstm_metadata.get('max_len', 5)  # fallback to 5 if not found
# #         if len(features) > max_len:
# #             features = features[-max_len:]  # keep most recent
# #         pad_len = max_len - len(features)
# #         if pad_len > 0:
# #             # Pad at the beginning (older missing visits) with zeros
# #             pad = [np.zeros(384)] * pad_len
# #             features = pad + features

# #         # Convert to tensor
# #         input_tensor = torch.tensor([features], dtype=torch.float32).to(device)  # (1, max_len, 384)

# #         # Predict
# #         with torch.no_grad():
# #             pred_cdr = lstm_model(input_tensor).item()

# #         return JSONResponse({
# #             "success": True,
# #             "predicted_cdr": round(pred_cdr, 3),
# #             "num_visits_used": len(files),
# #             "max_len": max_len
# #         })

# #     except Exception as e:
# #         logger.exception("LSTM prediction error")
# #         return JSONResponse({"success": False, "error": str(e)})

# # @app.post("/predict_progression_xgb")
# # async def predict_progression_xgb(
# #     file: UploadFile = File(...),
# #     Age: float = Form(...),
# #     EDUC: float = Form(...),
# #     SES: Optional[float] = Form(None),
# #     MMSE: float = Form(...),
# #     nWBV: float = Form(...),
# #     eTIV: float = Form(...),
# #     ASF: float = Form(...)
# # ):
# #     """
# #     Upload an MRI image and provide clinical data.
# #     Returns predicted MMSE for the next visit.
# #     """
# #     if xgb_model is None or classifier is None:
# #         return JSONResponse({"success": False, "error": "Model not available."})

# #     try:
# #         # Read and preprocess image
# #         contents = await file.read()
# #         image = Image.open(io.BytesIO(contents)).convert('RGB')
# #         img_tensor = classifier.transform(image).unsqueeze(0).to(device)
# #         with torch.no_grad():
# #             vit_features = classifier.model.forward_features(img_tensor).mean(dim=1).cpu().numpy().flatten()

# #         # Prepare clinical data array (order must match training: Age, EDUC, SES, nWBV, eTIV, ASF, MR Delay)
# #         # MR Delay is unknown, set to 0.
# #         clinical = np.array([Age, EDUC, SES if SES is not None else np.nan, nWBV, eTIV, ASF, 0], dtype=np.float32)
# #         clinical_reshaped = clinical.reshape(1, -1)
# #         clinical_imputed = xgb_imputer.transform(clinical_reshaped).flatten()

# #         # Combine features
# #         combined = np.concatenate([vit_features, clinical_imputed])
# #         combined = combined.reshape(1, -1)

# #         # Predict
# #         pred_mmse = xgb_model.predict(combined)[0]

# #         return JSONResponse({
# #             "success": True,
# #             "predicted_next_mmse": round(pred_mmse, 2),
# #             "interpretation": "Predicted MMSE score for the next visit (lower indicates more severe impairment)."
# #         })
# #     except Exception as e:
# #         logger.exception("XGBoost prediction error")
# #         return JSONResponse({"success": False, "error": str(e)})

# # # -------------------- 4. Health check --------------------
# # @app.get("/health")
# # async def health():
# #     status = {
# #         "status": "ok",
# #         "classifier": classifier is not None,
# #         "cox_model": cox_model is not None,
# #         "lstm_model": lstm_model is not None,
# #         "device": str(device)
# #     }
# #     return JSONResponse(status)  

# # =============================================================================
# # main.py – xAI-transmed API (with XGBoost MMSE progression)
# # =============================================================================

# import os
# import io
# import base64
# import pickle
# import logging
# from typing import List, Optional

# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, Query, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import joblib
# import xgboost as xgb

# # Local modules
# from model import ViTClassifier
# from explain import (
#     attention_rollout,
#     apply_heatmap,
#     compute_region_attention,
#     create_demo_regions,
#     ViTGradCAM,
# )

# # -------------------- Setup logging --------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # -------------------- FastAPI app --------------------
# app = FastAPI(title="xAI-transmed API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------- Device --------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"Using device: {device}")

# # -------------------- Load classification model --------------------
# try:
#     classifier = ViTClassifier(model_path="best_vit_model.pth", device=device)
#     logger.info("✅ ViT classifier loaded.")
# except Exception as e:
#     logger.error(f"Failed to load ViT model: {e}")
#     classifier = None

# # -------------------- Grad-CAM (for explainability) --------------------
# gradcam = ViTGradCAM(classifier.model, target_layer_name='blocks.11.norm1') if classifier else None

# # -------------------- Load Cox progression model (clinical) --------------------
# cox_model = None
# try:
#     with open('cox_progression_model.pkl', 'rb') as f:
#         cox_model = pickle.load(f)
#     logger.info("✅ Cox progression model loaded.")
# except FileNotFoundError:
#     logger.warning("cox_progression_model.pkl not found. Cox endpoint disabled.")
# except Exception as e:
#     logger.error(f"Failed to load Cox model: {e}")

# # -------------------- LSTM progression model (imaging) --------------------
# lstm_model = None
# lstm_metadata = {}

# class ProgressionLSTM(nn.Module):
#     def __init__(self, input_size=384, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, output_size)
#         )

#     def forward(self, x):
#         out, (hidden, cell) = self.lstm(x)
#         last_hidden = hidden[-1]
#         return self.fc(last_hidden)

# try:
#     torch.serialization.add_safe_globals([ProgressionLSTM])
#     lstm_model = torch.load('lstm_progression_full.pth', map_location=device, weights_only=False)
#     lstm_model.eval()
#     with open('lstm_metadata.pkl', 'rb') as f:
#         lstm_metadata = pickle.load(f)
#     logger.info("✅ LSTM progression model loaded.")
# except FileNotFoundError:
#     logger.warning("LSTM model files not found. LSTM endpoint disabled.")
# except Exception as e:
#     logger.error(f"Failed to load LSTM model: {e}")

# # -------------------- Load XGBoost progression model --------------------
# xgb_model = None
# try:
#     xgb_model = joblib.load('xgb_mmse_model.pkl')
#     logger.info("✅ XGBoost progression model loaded.")
# except FileNotFoundError:
#     logger.warning("xgb_mmse_model.pkl not found. XGB endpoint disabled.")
# except Exception as e:
#     logger.error(f"Failed to load XGBoost model: {e}")

# # -------------------- Median values for Cox model --------------------
# MEDIANS = {
#     'Age': 77.0,
#     'EDUC': 15.0,
#     'SES': 2.0,
#     'MMSE': 29.0,
#     'nWBV': 0.729,
#     'eTIV': 1470.0,
#     'ASF': 1.194
# }

# # -------------------- Pydantic schema for Cox input --------------------
# class ProgressionInput(BaseModel):
#     Age: float
#     EDUC: float
#     SES: Optional[float] = None
#     MMSE: float
#     nWBV: float
#     eTIV: float
#     ASF: float

# # -------------------- Region masks cache --------------------
# app.state.region_masks = None

# # ==================== ENDPOINTS ====================

# # -------------------- 1. Classification + Explainability --------------------
# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...),
#     method: str = Query("rollout", pattern="^(rollout|gradcam)$")
# ):
#     if classifier is None:
#         return JSONResponse({"success": False, "error": "Classifier not available."})
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert('RGB')
#         logger.info(f"Received image: size {image.size}")

#         pred = classifier.predict(image)
#         logger.info(f"Prediction: {pred}")

#         img_tensor = classifier.transform(image).unsqueeze(0).to(device)

#         if method == "gradcam":
#             class_idx = classifier.class_names.index(pred['predicted_class'])
#             mask = gradcam.generate_heatmap(img_tensor, class_idx=class_idx)
#         else:
#             mask = attention_rollout(classifier.model, img_tensor)

#         heatmap_img = apply_heatmap(image, mask)

#         img_size = (image.height, image.width)
#         if app.state.region_masks is None:
#             app.state.region_masks = create_demo_regions(img_size)
#         region_scores = compute_region_attention(mask, img_size, app.state.region_masks)

#         buffered_orig = io.BytesIO()
#         image.save(buffered_orig, format="PNG")
#         orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode()

#         buffered_heat = io.BytesIO()
#         heatmap_img.save(buffered_heat, format="PNG")
#         heatmap_base64 = base64.b64encode(buffered_heat.getvalue()).decode()

#         return JSONResponse({
#             "success": True,
#             "prediction": pred,
#             "original_image": orig_base64,
#             "heatmap": heatmap_base64,
#             "region_scores": region_scores,
#             "method_used": method
#         })
#     except Exception as e:
#         logger.exception("Prediction error")
#         return JSONResponse({"success": False, "error": str(e)})

# # -------------------- 2. Progression (Cox model, clinical) --------------------
# @app.post("/predict_progression")
# async def predict_progression(data: ProgressionInput):
#     if cox_model is None:
#         return JSONResponse({"success": False, "error": "Cox model not available."})
#     try:
#         input_dict = data.dict()
#         for key in MEDIANS:
#             if input_dict.get(key) is None:
#                 input_dict[key] = MEDIANS[key]

#         df_input = pd.DataFrame([input_dict])
#         required_cols = cox_model.params_.index.tolist()
#         df_input = df_input[required_cols]

#         hazard_ratio = cox_model.predict_partial_hazard(df_input).iloc[0]

#         return JSONResponse({
#             "success": True,
#             "hazard_ratio": float(hazard_ratio),
#             "interpretation": "Values >1 indicate higher risk."
#         })
#     except Exception as e:
#         logger.exception("Cox prediction error")
#         return JSONResponse({"success": False, "error": str(e)})

# # -------------------- 3. Progression (LSTM, imaging) --------------------
# @app.post("/predict_progression_lstm")
# async def predict_progression_lstm(files: List[UploadFile] = File(...)):
#     if lstm_model is None:
#         return JSONResponse({"success": False, "error": "LSTM model not available."})
#     try:
#         features = []
#         for file in files:
#             contents = await file.read()
#             image = Image.open(io.BytesIO(contents)).convert('RGB')
#             img_tensor = classifier.transform(image).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 feat = classifier.model.forward_features(img_tensor).mean(dim=1).cpu().numpy().flatten()
#             features.append(feat)

#         max_len = lstm_metadata.get('max_len', 5)
#         if len(features) > max_len:
#             features = features[-max_len:]
#         pad_len = max_len - len(features)
#         if pad_len > 0:
#             pad = [np.zeros(384)] * pad_len
#             features = pad + features

#         input_tensor = torch.tensor([features], dtype=torch.float32).to(device)
#         with torch.no_grad():
#             pred_cdr = lstm_model(input_tensor).item()

#         return JSONResponse({
#             "success": True,
#             "predicted_cdr": round(pred_cdr, 3),
#             "num_visits_used": len(files),
#             "max_len": max_len
#         })
#     except Exception as e:
#         logger.exception("LSTM prediction error")
#         return JSONResponse({"success": False, "error": str(e)})

# # -------------------- 4. Progression (XGBoost, single MRI + clinical) --------------------
# @app.post("/predict_progression_xgb")
# async def predict_progression_xgb(
#     file: UploadFile = File(...),
#     Age: float = Form(...),
#     EDUC: float = Form(...),
#     SES: Optional[float] = Form(None),
#     MMSE: float = Form(...),
#     nWBV: float = Form(...),
#     eTIV: float = Form(...),
#     ASF: float = Form(...)
# ):
#     if xgb_model is None or classifier is None:
#         return JSONResponse({"success": False, "error": "Model not available."})

#     try:
#         # Read and preprocess image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert('RGB')
#         img_tensor = classifier.transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             vit_features = classifier.model.forward_features(img_tensor).mean(dim=1).cpu().numpy().flatten()

#         # Clinical array (order: Age, EDUC, SES, nWBV, eTIV, ASF, MR Delay)
#         # Use median SES=2.0 if not provided
#         ses_value = 2.0 if SES is None else SES
#         clinical = np.array([Age, EDUC, ses_value, nWBV, eTIV, ASF, 0], dtype=np.float32)

#         # Combine
#         combined = np.concatenate([vit_features, clinical])
#         combined = combined.reshape(1, -1)

#         # Predict
#         pred_mmse = xgb_model.predict(combined)[0]

#         return JSONResponse({
#             "success": True,
#             "predicted_next_mmse": round(pred_mmse, 2),
#             "interpretation": "Predicted MMSE score for the next visit (lower indicates more severe impairment)."
#         })
#     except Exception as e:
#         logger.exception("XGBoost prediction error")
#         return JSONResponse({"success": False, "error": str(e)})

# # -------------------- 5. Health check --------------------
# @app.get("/health")
# async def health():
#     status = {
#         "status": "ok",
#         "classifier": classifier is not None,
#         "cox_model": cox_model is not None,
#         "lstm_model": lstm_model is not None,
#         "xgb_model": xgb_model is not None,
#         "device": str(device)
#     }
#     return JSONResponse(status)
# =============================================================================
# main.py – xAI-transmed API (with XGBoost MMSE progression)
# =============================================================================
from models import ProgressionLSTM

import os
import io
import base64
import pickle
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import xgboost as xgb


# import os
# print("Current working directory:", os.getcwd())
# print("Files in this directory:")
# for f in os.listdir('.'):
#     print(f"  {f}")



# Local modules
from model import ViTClassifier
from explain import (
    attention_rollout,
    apply_heatmap,
    compute_region_attention,
    create_demo_regions,
    ViTGradCAM,
)

# -------------------- Setup logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- FastAPI app --------------------
app = FastAPI(title="xAI-transmed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -------------------- Load classification model --------------------
try:
    classifier = ViTClassifier(model_path="best_vit_model.pth", device=device)
    logger.info("✅ ViT classifier loaded.")
except Exception as e:
    logger.error(f"Failed to load ViT model: {e}")
    classifier = None

# -------------------- Grad-CAM (for explainability) --------------------
gradcam = ViTGradCAM(classifier.model, target_layer_name='blocks.11.norm1') if classifier else None

# -------------------- Load Cox progression model (clinical) --------------------
cox_model = None
try:
    with open('cox_progression_model.pkl', 'rb') as f:
        cox_model = pickle.load(f)
    logger.info("✅ Cox progression model loaded.")
except FileNotFoundError:
    logger.warning("cox_progression_model.pkl not found. Cox endpoint disabled.")
except Exception as e:
    logger.error(f"Failed to load Cox model: {e}")

# -------------------- LSTM progression model (imaging) --------------------
lstm_model = None
lstm_metadata = {}

# class ProgressionLSTM(nn.Module):
#     def __init__(self, input_size=384, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, output_size)
#         )

#     def forward(self, x):
#         out, (hidden, cell) = self.lstm(x)
#         last_hidden = hidden[-1]
#         return self.fc(last_hidden)
import __main__
__main__.ProgressionLSTM = ProgressionLSTM


try:
    torch.serialization.add_safe_globals([ProgressionLSTM])
    lstm_model = torch.load('lstm_progression_full.pth', map_location=device, weights_only=False)
    lstm_model.eval()
    with open('lstm_metadata.pkl', 'rb') as f:
        lstm_metadata = pickle.load(f)
    logger.info("✅ LSTM progression model loaded.")
except FileNotFoundError:
    logger.warning("LSTM model files not found. LSTM endpoint disabled.")
except Exception as e:
    logger.error(f"Failed to load LSTM model: {e}")

# -------------------- Load XGBoost progression model --------------------
xgb_model = None
try:
    xgb_model = joblib.load('xgb_mmse_model.pkl')
    logger.info("✅ XGBoost progression model loaded.")
except FileNotFoundError:
    logger.warning("xgb_mmse_model.pkl not found. XGB endpoint disabled.")
except Exception as e:
    logger.error(f"Failed to load XGBoost model: {e}")

# -------------------- Median values for Cox model --------------------
MEDIANS = {
    'Age': 77.0,
    'EDUC': 15.0,
    'SES': 2.0,
    'MMSE': 29.0,
    'nWBV': 0.729,
    'eTIV': 1470.0,
    'ASF': 1.194
}

# -------------------- Pydantic schema for Cox input --------------------
class ProgressionInput(BaseModel):
    Age: float
    EDUC: float
    SES: Optional[float] = None
    MMSE: float
    nWBV: float
    eTIV: float
    ASF: float

# -------------------- Region masks cache --------------------
app.state.region_masks = None

# ==================== ENDPOINTS ====================

# -------------------- 1. Classification + Explainability --------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    method: str = Query("rollout", pattern="^(rollout|gradcam)$")
):
    if classifier is None:
        return JSONResponse({"success": False, "error": "Classifier not available."})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        logger.info(f"Received image: size {image.size}")

        pred = classifier.predict(image)
        logger.info(f"Prediction: {pred}")

        img_tensor = classifier.transform(image).unsqueeze(0).to(device)

        if method == "gradcam":
            class_idx = classifier.class_names.index(pred['predicted_class'])
            mask = gradcam.generate_heatmap(img_tensor, class_idx=class_idx)
        else:
            mask = attention_rollout(classifier.model, img_tensor)

        heatmap_img = apply_heatmap(image, mask)

        img_size = (image.height, image.width)
        if app.state.region_masks is None:
            app.state.region_masks = create_demo_regions(img_size)
        region_scores = compute_region_attention(mask, img_size, app.state.region_masks)

        buffered_orig = io.BytesIO()
        image.save(buffered_orig, format="PNG")
        orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode()

        buffered_heat = io.BytesIO()
        heatmap_img.save(buffered_heat, format="PNG")
        heatmap_base64 = base64.b64encode(buffered_heat.getvalue()).decode()

        return JSONResponse({
            "success": True,
            "prediction": pred,
            "original_image": orig_base64,
            "heatmap": heatmap_base64,
            "region_scores": region_scores,
            "method_used": method
        })
    except Exception as e:
        logger.exception("Prediction error")
        return JSONResponse({"success": False, "error": str(e)})

# -------------------- 2. Progression (Cox model, clinical) --------------------
@app.post("/predict_progression")
async def predict_progression(data: ProgressionInput):
    if cox_model is None:
        return JSONResponse({"success": False, "error": "Cox model not available."})
    try:
        input_dict = data.dict()
        for key in MEDIANS:
            if input_dict.get(key) is None:
                input_dict[key] = MEDIANS[key]

        df_input = pd.DataFrame([input_dict])
        required_cols = cox_model.params_.index.tolist()
        df_input = df_input[required_cols]

        hazard_ratio = cox_model.predict_partial_hazard(df_input).iloc[0]

        return JSONResponse({
            "success": True,
            "hazard_ratio": float(hazard_ratio),
            "interpretation": "Values >1 indicate higher risk."
        })
    except Exception as e:
        logger.exception("Cox prediction error")
        return JSONResponse({"success": False, "error": str(e)})

# -------------------- 3. Progression (LSTM, imaging) --------------------
@app.post("/predict_progression_lstm")
async def predict_progression_lstm(files: List[UploadFile] = File(...)):
    if lstm_model is None:
        return JSONResponse({"success": False, "error": "LSTM model not available."})
    try:
        features = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            img_tensor = classifier.transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = classifier.model.forward_features(img_tensor).mean(dim=1).cpu().numpy().flatten()
            features.append(feat)

        max_len = lstm_metadata.get('max_len', 5)
        if len(features) > max_len:
            features = features[-max_len:]
        pad_len = max_len - len(features)
        if pad_len > 0:
            pad = [np.zeros(384)] * pad_len
            features = pad + features

        input_tensor = torch.tensor([features], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_cdr = lstm_model(input_tensor).item()

        return JSONResponse({
            "success": True,
            "predicted_cdr": round(pred_cdr, 3),
            "num_visits_used": len(files),
            "max_len": max_len
        })
    except Exception as e:
        logger.exception("LSTM prediction error")
        return JSONResponse({"success": False, "error": str(e)})

# -------------------- 4. Progression (XGBoost, single MRI + clinical) --------------------
@app.post("/predict_progression_xgb")
async def predict_progression_xgb(
    file: UploadFile = File(...),
    Age: float = Form(...),
    EDUC: float = Form(...),
    SES: Optional[float] = Form(None),
    MMSE: float = Form(...),
    nWBV: float = Form(...),
    eTIV: float = Form(...),
    ASF: float = Form(...)
):
    if xgb_model is None or classifier is None:
        return JSONResponse({"success": False, "error": "Model not available."})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = classifier.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            vit_features = classifier.model.forward_features(img_tensor).mean(dim=1).cpu().numpy().flatten()

        ses_value = 2.0 if SES is None else SES
        clinical = np.array([Age, EDUC, ses_value, nWBV, eTIV, ASF, 0], dtype=np.float32)

        combined = np.concatenate([vit_features, clinical])
        combined = combined.reshape(1, -1)

        pred_mmse = float(xgb_model.predict(combined)[0])
        mmse_change = pred_mmse - MMSE

        # Categorize progression rate
        if mmse_change < -5:
            rate = "🔴 Rapid decline"
        elif mmse_change < -3:
            rate = "🟡 Moderate decline"
        elif mmse_change < -1:
            rate = "🟢 Slow decline"
        else:
            rate = "⚪ Stable / No decline"

        return JSONResponse({
            "success": True,
            "predicted_next_mmse": round(pred_mmse, 2),
            "mmse_change": round(mmse_change, 2),
            "progression_rate": rate,
            "interpretation": f"MMSE change: {mmse_change:+.2f} points. {rate}"
        })
    except Exception as e:
        logger.exception("XGBoost prediction error")
        return JSONResponse({"success": False, "error": str(e)})

# -------------------- 5. Health check --------------------
@app.get("/health")
async def health():
    status = {
        "status": "ok",
        "classifier": classifier is not None,
        "cox_model": cox_model is not None,
        "lstm_model": lstm_model is not None,
        "xgb_model": xgb_model is not None,
        "device": str(device)
    }
    return JSONResponse(status)