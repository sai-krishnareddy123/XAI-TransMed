# # explain.py – Complete version with Grad‑CAM and region attention (fixed)

# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import cv2
# import logging
# from typing import Dict, Tuple, Optional

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ==================== Attention Rollout ====================

# class AttentionWrapper(nn.Module):
#     """Wrap an attention module to store attention weights."""
#     def __init__(self, attn_module):
#         super().__init__()
#         self.attn_module = attn_module
#         self.attention_maps = None

#     def forward(self, x, *args, **kwargs):
#         B, N, C = x.shape
#         qkv = self.attn_module.qkv(x).reshape(B, N, 3, self.attn_module.num_heads, C // self.attn_module.num_heads)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
#         q, k, v = qkv.unbind(0)

#         q = q * self.attn_module.scale
#         attn = (q @ k.transpose(-2, -1))
#         attn = attn.softmax(dim=-1)
#         self.attention_maps = attn.detach()

#         attn = self.attn_module.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.attn_module.proj(x)
#         x = self.attn_module.proj_drop(x)
#         return x

# def attention_rollout(model, image_tensor, head_fusion='mean', discard_ratio=0.9):
#     """Compute attention rollout heatmap."""
#     original_attns = []
#     wrappers = []

#     for name, module in model.named_modules():
#         if 'attn' in name and hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
#             original_attns.append((name, module))
#             wrapper = AttentionWrapper(module)
#             parent = model
#             for part in name.split('.')[:-1]:
#                 parent = getattr(parent, part)
#             setattr(parent, name.split('.')[-1], wrapper)
#             wrappers.append(wrapper)
#             logger.info(f"Wrapped: {name}")

#     if not wrappers:
#         raise RuntimeError("No attention modules found.")

#     model.eval()
#     with torch.no_grad():
#         _ = model(image_tensor)

#     all_attentions = []
#     for wrapper in wrappers:
#         if wrapper.attention_maps is not None:
#             all_attentions.append(wrapper.attention_maps)

#     for (name, original), wrapper in zip(original_attns, wrappers):
#         parent = model
#         for part in name.split('.')[:-1]:
#             parent = getattr(parent, part)
#         setattr(parent, name.split('.')[-1], original)

#     if not all_attentions:
#         raise RuntimeError("No attention maps captured.")

#     attention_stack = torch.stack(all_attentions, dim=0).squeeze(1)  # (L, H, N, N)
#     num_tokens = attention_stack.shape[-1]

#     if head_fusion == 'mean':
#         attention = attention_stack.mean(dim=1)
#     elif head_fusion == 'max':
#         attention = attention_stack.max(dim=1)[0]
#     elif head_fusion == 'min':
#         attention = attention_stack.min(dim=1)[0]
#     else:
#         raise ValueError("head_fusion must be 'mean', 'max', or 'min'")

#     result = torch.eye(num_tokens, device=attention.device)
#     for layer_attn in attention:
#         layer_attn = layer_attn + torch.eye(num_tokens, device=attention.device)
#         layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True)
#         result = torch.matmul(layer_attn, result)

#     cls_attn = result[0, 1:]
#     cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
#     return cls_attn.cpu().numpy()

# # ==================== Heatmap Overlay ====================

# # def apply_heatmap(image: Image.Image, mask: np.ndarray):
# #     """Resize mask to image size and overlay as heatmap."""
# #     img = np.array(image.convert('RGB'))
# #     h, w, _ = img.shape
# #     patch_size = 16
# #     grid_h = h // patch_size
# #     grid_w = w // patch_size

# #     # Determine the grid size of the attention mask (assume square)
# #     mask_len = len(mask)
# #     mask_grid = int(np.sqrt(mask_len))  # e.g., 14 for 224x224 input
# #     attn_grid = mask.reshape((mask_grid, mask_grid))

# #     # Upsample to original image size
# #     attn_full = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_LINEAR)

# #     # Apply colormap
# #     heatmap = cv2.applyColorMap(np.uint8(255 * attn_full), cv2.COLORMAP_JET)
# #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
# #     overlayed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
# #     return Image.fromarray(overlayed)


# def apply_heatmap(image: Image.Image, mask: np.ndarray):
#     """
#     Resize mask to image size and overlay as heatmap.
#     mask: 1D numpy array of length N (should be a perfect square, e.g., 196).
#     """
#     img = np.array(image.convert('RGB'))
#     h, w = img.shape[:2]

#     # Determine grid size from mask length (must be square)
#     mask_len = mask.size
#     grid_size = int(np.sqrt(mask_len))
#     if grid_size * grid_size != mask_len:
#         # Fallback (should not happen)
#         grid_size = int(np.sqrt(mask_len))
#     attn_grid = mask.reshape((grid_size, grid_size))

#     # Resize to original image dimensions
#     attn_full = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_LINEAR)

#     # Apply colormap (mask should be in [0,1] already)
#     heatmap = cv2.applyColorMap(np.uint8(255 * attn_full), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     overlayed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
#     return Image.fromarray(overlayed)

# # ==================== Region Attention Scores ====================

# # def compute_region_attention(
# #     attention_mask: np.ndarray,          # 1D array from attention_rollout (length = num_patches)
# #     image_size: Tuple[int, int],         # (height, width) of original image
# #     regions: Dict[str, np.ndarray]       # dict of region_name -> binary mask (2D, same size as image)
# # ) -> Dict[str, float]:
# #     """Compute average attention score for each brain region."""
# #     h, w = image_size
# #     patch_size = 16

# #     # Determine grid size of the attention mask (assume square)
# #     mask_len = len(attention_mask)
# #     mask_grid = int(np.sqrt(mask_len))  # e.g., 14 for 224x224 input
# #     attn_grid = attention_mask.reshape((mask_grid, mask_grid))

# #     # Upsample to original image size
# #     attn_full = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_LINEAR)

# #     region_scores = {}
# #     for name, mask in regions.items():
# #         if mask.shape != (h, w):
# #             mask = cv2.resize(mask.astype(np.uint8), (w, h))
# #             mask = mask > 0.5
# #         if np.sum(mask) > 0:
# #             score = np.mean(attn_full[mask])
# #         else:
# #             score = 0.0
# #         region_scores[name] = float(score)
# #     return region_scores

# def compute_region_attention(
#     attention_mask: np.ndarray,  # 1D array from attention_rollout (length = num_patches)
#     image_size: Tuple[int, int], # (height, width) of original image
#     regions: Dict[str, np.ndarray]  # dict of region_name -> binary mask (2D, same size as image)
# ) -> Dict[str, float]:
#     """Compute average attention score for each brain region."""
#     h, w = image_size

#     # Determine grid size from attention_mask itself (should be square)
#     mask_len = len(attention_mask)
#     mask_grid = int(np.sqrt(mask_len))
#     if mask_grid * mask_grid != mask_len:
#         # Fallback (should not happen with proper masks)
#         mask_grid = int(np.sqrt(mask_len))
#     attn_grid = attention_mask.reshape((mask_grid, mask_grid))

#     # Upsample to original image size
#     attn_full = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_LINEAR)

#     region_scores = {}
#     for name, mask in regions.items():
#         if mask.shape != (h, w):
#             mask = cv2.resize(mask.astype(np.uint8), (w, h))
#             mask = mask > 0.5
#         if np.sum(mask) > 0:
#             score = np.mean(attn_full[mask])
#         else:
#             score = 0.0
#         region_scores[name] = float(score)
#     return region_scores

# def create_demo_regions(image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
#     """Create simple rectangular masks for demonstration."""
#     h, w = image_size
#     regions = {}
#     # Left hippocampus
#     left_mask = np.zeros((h, w), dtype=bool)
#     left_mask[int(h*0.4):int(h*0.6), int(w*0.2):int(w*0.4)] = True
#     regions['left_hippocampus'] = left_mask
#     # Right hippocampus
#     right_mask = np.zeros((h, w), dtype=bool)
#     right_mask[int(h*0.4):int(h*0.6), int(w*0.6):int(w*0.8)] = True
#     regions['right_hippocampus'] = right_mask
#     # Cortex (outer ring approximation)
#     cortex_mask = np.ones((h, w), dtype=bool)
#     cortex_mask[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = False
#     regions['cortex'] = cortex_mask
#     return regions

# # ==================== Grad‑CAM for ViT ====================

# # class ViTGradCAM:
# #     """Grad‑CAM for Vision Transformer using the last transformer block."""
# #     def __init__(self, model, target_layer_name: str = 'blocks.11.norm1'):
# #         self.model = model
# #         self.target_layer_name = target_layer_name
# #         self.activations = None
# #         self.gradients = None
# #         self._register_hooks()

# #     def _register_hooks(self):
# #         target_module = dict(self.model.named_modules())[self.target_layer_name]
# #         target_module.register_forward_hook(self._save_activation)
# #         target_module.register_full_backward_hook(self._save_gradient)

# #     def _save_activation(self, module, input, output):
# #         self.activations = output.detach()

# #     def _save_gradient(self, module, grad_input, grad_output):
# #         self.gradients = grad_output[0].detach()

# #     def generate_heatmap(self, input_tensor, class_idx=None):
# #         self.model.zero_grad()
# #         output = self.model(input_tensor)
# #         if class_idx is None:
# #             class_idx = output.argmax(dim=1).item()

# #         one_hot = torch.zeros_like(output)
# #         one_hot[0, class_idx] = 1
# #         output.backward(gradient=one_hot, retain_graph=True)

# #         gradients = self.gradients  # (1, num_patches+1, hidden_dim)
# #         activations = self.activations

# #         # Exclude CLS token
# #         gradients = gradients[:, 1:, :]
# #         activations = activations[:, 1:, :]

# #         # Global average pooling of gradients per channel
# #         weights = gradients.mean(dim=(0, 1))  # (hidden_dim,)

# #         # Weighted combination
# #         cam = (weights * activations).sum(dim=-1)  # (1, num_patches)
# #         cam = cam.squeeze(0).cpu().numpy()

# #         cam = np.maximum(cam, 0)  # ReLU
# #         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# #         # Reshape to grid
# #         num_patches = cam.shape[0]
# #         grid_h = grid_w = int(np.sqrt(num_patches))
# #         cam = cam.reshape((grid_h, grid_w))
# #         return cam



# class ViTGradCAM:
#     def __init__(self, model, target_layer_name='blocks.11.norm1'):
#         self.model = model
#         self.target_layer_name = target_layer_name
#         self.activations = None
#         self.gradients = None
#         self._register_hooks()

#     def _register_hooks(self):
#         target_module = dict(self.model.named_modules())[self.target_layer_name]
#         print(f"Hooking into: {self.target_layer_name}")
#         target_module.register_forward_hook(self._save_activation)
#         target_module.register_full_backward_hook(self._save_gradient)

#     def _save_activation(self, module, input, output):
#         self.activations = output.detach()
#         print(f"✅ Activation captured: shape {self.activations.shape}")

#     def _save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()
#         print(f"✅ Gradient captured: shape {self.gradients.shape}")

#     def generate_heatmap(self, input_tensor, class_idx=None):
#         self.model.zero_grad()
#         output = self.model(input_tensor)
#         print(f"Model output shape: {output.shape}")
#         if class_idx is None:
#             class_idx = output.argmax(dim=1).item()
#             print(f"Using predicted class: {class_idx}")

#         one_hot = torch.zeros_like(output)
#         one_hot[0, class_idx] = 1
#         output.backward(gradient=one_hot, retain_graph=True)
#         print("✅ Backward pass completed")

#         gradients = self.gradients
#         activations = self.activations

#         if gradients is None or activations is None:
#             print("❌ ERROR: Gradients or activations not captured!")
#             return None

#         # Exclude CLS token
#         gradients = gradients[:, 1:, :]
#         activations = activations[:, 1:, :]
#         print(f"Gradients after CLS removal: {gradients.shape}")
#         print(f"Activations after CLS removal: {activations.shape}")

#         weights = gradients.mean(dim=(0, 1))
#         cam = (weights * activations).sum(dim=-1)
#         cam = cam.squeeze(0).cpu().numpy()
#         print(f"Raw CAM shape: {cam.shape}")

#         cam = np.maximum(cam, 0)
#         if cam.max() > 0:
#             cam = cam / cam.max()
#         else:
#             cam = np.zeros_like(cam)

#         num_patches = cam.shape[0]
#         grid_h = grid_w = int(np.sqrt(num_patches))
#         cam = cam.reshape((grid_h, grid_w))
#         print(f"✅ Final CAM shape: {cam.shape}")
#         return cam



# explain.py – Complete version with robust reshaping

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Attention Rollout ====================

class AttentionWrapper(nn.Module):
    """Wrap an attention module to store attention weights."""
    def __init__(self, attn_module):
        super().__init__()
        self.attn_module = attn_module
        self.attention_maps = None

    def forward(self, x, *args, **kwargs):
        B, N, C = x.shape
        qkv = self.attn_module.qkv(x).reshape(B, N, 3, self.attn_module.num_heads, C // self.attn_module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)

        q = q * self.attn_module.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        self.attention_maps = attn.detach()

        attn = self.attn_module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attn_module.proj(x)
        x = self.attn_module.proj_drop(x)
        return x

def attention_rollout(model, image_tensor, head_fusion='mean', discard_ratio=0.9):
    """Compute attention rollout heatmap."""
    original_attns = []
    wrappers = []

    for name, module in model.named_modules():
        if 'attn' in name and hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
            original_attns.append((name, module))
            wrapper = AttentionWrapper(module)
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], wrapper)
            wrappers.append(wrapper)
            logger.info(f"Wrapped: {name}")

    if not wrappers:
        raise RuntimeError("No attention modules found.")

    model.eval()
    with torch.no_grad():
        _ = model(image_tensor)

    all_attentions = []
    for wrapper in wrappers:
        if wrapper.attention_maps is not None:
            all_attentions.append(wrapper.attention_maps)

    for (name, original), wrapper in zip(original_attns, wrappers):
        parent = model
        for part in name.split('.')[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name.split('.')[-1], original)

    if not all_attentions:
        raise RuntimeError("No attention maps captured.")

    attention_stack = torch.stack(all_attentions, dim=0).squeeze(1)  # (L, H, N, N)
    num_tokens = attention_stack.shape[-1]

    if head_fusion == 'mean':
        attention = attention_stack.mean(dim=1)
    elif head_fusion == 'max':
        attention = attention_stack.max(dim=1)[0]
    elif head_fusion == 'min':
        attention = attention_stack.min(dim=1)[0]
    else:
        raise ValueError("head_fusion must be 'mean', 'max', or 'min'")

    result = torch.eye(num_tokens, device=attention.device)
    for layer_attn in attention:
        layer_attn = layer_attn + torch.eye(num_tokens, device=attention.device)
        layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True)
        result = torch.matmul(layer_attn, result)

    cls_attn = result[0, 1:]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn.cpu().numpy()

# ==================== Heatmap Overlay ====================

def apply_heatmap(image: Image.Image, mask: np.ndarray):
    """Resize mask to image size and overlay as heatmap (robust version)."""
    img = np.array(image.convert('RGB'))
    h, w, _ = img.shape

    # The mask is from the model's 224x224 input, so its grid size is sqrt(len(mask)).
    mask_len = mask.size
    grid_size = int(np.sqrt(mask_len))
    print(f"apply_heatmap: mask_len={mask_len}, grid_size={grid_size}")  # debug
    attn_grid = mask.reshape((grid_size, grid_size))

    # Upsample to original image size
    attn_full = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_full), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlayed)

# ==================== Region Attention Scores ====================

def compute_region_attention(
    attention_mask: np.ndarray,
    image_size: Tuple[int, int],
    regions: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Compute average attention score for each brain region."""
    h, w = image_size

    # The attention_mask is from the model's 224x224 input, so its grid size is sqrt(len(mask)).
    mask_len = attention_mask.size
    mask_grid = int(np.sqrt(mask_len))
    print(f"compute_region_attention: mask_len={mask_len}, mask_grid={mask_grid}")  # debug
    attn_grid = attention_mask.reshape((mask_grid, mask_grid))

    # Upsample to original image size
    attn_full = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    region_scores = {}
    for name, mask in regions.items():
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h))
            mask = mask > 0.5
        if np.sum(mask) > 0:
            score = np.mean(attn_full[mask])
        else:
            score = 0.0
        region_scores[name] = float(score)
    return region_scores

def create_demo_regions(image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create simple rectangular masks for demonstration."""
    h, w = image_size
    regions = {}
    # Left hippocampus
    left_mask = np.zeros((h, w), dtype=bool)
    left_mask[int(h*0.4):int(h*0.6), int(w*0.2):int(w*0.4)] = True
    regions['left_hippocampus'] = left_mask
    # Right hippocampus
    right_mask = np.zeros((h, w), dtype=bool)
    right_mask[int(h*0.4):int(h*0.6), int(w*0.6):int(w*0.8)] = True
    regions['right_hippocampus'] = right_mask
    # Cortex (outer ring approximation)
    cortex_mask = np.ones((h, w), dtype=bool)
    cortex_mask[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = False
    regions['cortex'] = cortex_mask
    return regions

# ==================== Grad‑CAM for ViT ====================

class ViTGradCAM:
    def __init__(self, model, target_layer_name='blocks.11.norm1'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        target_module = dict(self.model.named_modules())[self.target_layer_name]
        print(f"Hooking into: {self.target_layer_name}")
        target_module.register_forward_hook(self._save_activation)
        target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        print(f"✅ Activation captured: shape {self.activations.shape}")

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        print(f"✅ Gradient captured: shape {self.gradients.shape}")

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        print(f"Model output shape: {output.shape}")
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            print(f"Using predicted class: {class_idx}")

        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        print("✅ Backward pass completed")

        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            print("❌ ERROR: Gradients or activations not captured!")
            return None

        # Exclude CLS token
        gradients = gradients[:, 1:, :]
        activations = activations[:, 1:, :]
        print(f"Gradients after CLS removal: {gradients.shape}")
        print(f"Activations after CLS removal: {activations.shape}")

        # Global average pooling of gradients per channel
        weights = gradients.mean(dim=(0, 1))  # (hidden_dim,)

        # Weighted combination
        cam = (weights * activations).sum(dim=-1)  # (1, num_patches)
        cam = cam.squeeze(0).cpu().numpy()
        print(f"Raw CAM shape: {cam.shape}")

        cam = np.maximum(cam, 0)  # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        # Reshape to grid
        num_patches = cam.shape[0]
        grid_h = grid_w = int(np.sqrt(num_patches))
        cam = cam.reshape((grid_h, grid_w))
        print(f"✅ Final CAM shape: {cam.shape}")
        return cam