import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

class ViTClassifier:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def predict(self, image: Image.Image):
        """Return top-1 class and probabilities."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = self.class_names[probs.argmax()]
        confidence = float(probs.max())
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': {self.class_names[i]: float(probs[i]) for i in range(4)}
        }