# backend/app/model_stub.py
import json
import csv
import io
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import base64

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
import matplotlib.cm as cm


class SkinDiseaseModel:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = project_root / "saved_model" / "efficientnet_b0_finetuned.pt"
        self.mapping_path = project_root / "saved_model" / "class_mapping.json"
        self.log_path = project_root / "saved_model" / "prediction_logs.csv"

        # Load class mapping
        with open(self.mapping_path, "r") as f:
            self.idx_to_class: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}

        # Human friendly descriptions (short). Edit as needed.
        self.class_descriptions = {
            "MEL": "Melanoma — an aggressive skin cancer. Needs prompt clinical attention.",
            "NV": "Melanocytic nevus (mole) — usually benign.",
            "BCC": "Basal cell carcinoma — common form of skin cancer, often localised.",
            "AKIEC": "Actinic keratoses/ Intraepithelial carcinoma — sun-damage lesions.",
            "BKL": "Benign keratosis — benign, often scaly skin lesion.",
            "DF": "Dermatofibroma — benign fibrous lesion.",
            "VASC": "Vascular lesion — blood-vessel related lesion.",
        }

        # Build model architecture
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=weights)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, len(self.idx_to_class))

        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Transforms - match training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # For Grad-CAM hooks
        self._features = None
        self._grads = None
        # hook into the last feature block (features[-1])
        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    # hooks
    def _forward_hook(self, module, input, output):
        # output shape: (B, C, H, W)
        self._features = output.detach().cpu().numpy()

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is tuple, grad_out[0] is gradients wrt output
        self._grads = grad_out[0].detach().cpu().numpy()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)  # 1,C,H,W

    def _generate_gradcam(self, input_tensor: torch.Tensor, class_idx: int, orig_size: Tuple[int, int]) -> str:
        """
        Returns a base64 PNG of the heatmap (resized to orig_size)
        """
        # ensure grads/features from previous forward are cleared
        self._features = None
        self._grads = None

        x = input_tensor.to(self.device)
        x.requires_grad = True

        # forward
        outputs = self.model(x)  # (1, num_classes)
        score = outputs[0, class_idx]
        # backward to get gradients
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # features and grads should be set by hooks (numpy, CPU)
        if self._features is None or self._grads is None:
            # fallback: return empty transparent image
            return self._transparent_png_base64(orig_size)

        # features shape (1, C, H, W), grads shape same
        features = self._features[0]  # C,H,W
        grads = self._grads[0]        # C,H,W

        # global-average-pool grads over spatial dims -> weights
        weights = np.mean(grads, axis=(1, 2))  # C

        # weighted sum of feature maps
        cam = np.zeros(features.shape[1:], dtype=np.float32)  # H,W
        for i, w in enumerate(weights):
            cam += w * features[i]

        # relu
        cam = np.maximum(cam, 0)
        # normalize
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        # resize to original image size
        cam_img = (cam * 255).astype(np.uint8)
        cam_pil = Image.fromarray(cam_img).resize(orig_size, resample=Image.BILINEAR)

        # apply colormap (matplotlib) to get RGBA
        cmap = cm.get_cmap("jet")
        cam_arr = np.array(cam_pil) / 255.0  # H,W in 0-1
        colored = cmap(cam_arr)  # H,W,4 (RGBA floats 0-1)
        colored = (colored * 255).astype(np.uint8)
        heatmap = Image.fromarray(colored).convert("RGBA")

        # overlay optionally later in frontend; here return heatmap PNG base64
        buffered = io.BytesIO()
        heatmap.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _transparent_png_base64(self, size: Tuple[int, int]) -> str:
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(self, pil_image: Image.Image, save_log: bool = True):
        """
        Returns:
            predicted_class_name: str
            prob_dict: Dict[class_name, float]
            heatmap_b64: str
            description: str
            log_info: dict (contains log fields saved)
        """
        orig_size = pil_image.size  # (W, H)
        inp = self.preprocess(pil_image)  # tensor 1,C,224,224

        with torch.no_grad():
            outputs = self.model(inp.to(self.device))
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        # map to classes
        prob_dict = {}
        for idx, p in enumerate(probs):
            class_name = self.idx_to_class.get(idx, str(idx))
            prob_dict[class_name] = float(p)

        # top1
        best_idx = int(np.argmax(probs))
        predicted_class = self.idx_to_class[best_idx]
        description = self.class_descriptions.get(predicted_class, "")

        # generate gradcam (we need gradient so call _generate_gradcam which runs backward)
        # For gradcam we require gradients; so run a full forward+backprop without torch.no_grad
        try:
            heatmap_b64 = self._generate_gradcam(self.preprocess(pil_image), best_idx, orig_size[::-1])  # PIL size (W,H); we expect (H,W) in resize so pass reversed
        except Exception as e:
            # fallback to transparent heatmap
            heatmap_b64 = self._transparent_png_base64(orig_size[::-1])

        # Logging
        log_info = {}
        if save_log:
            log_info = self._append_log(pil_image, predicted_class, prob_dict)

        return predicted_class, prob_dict, heatmap_b64, description, log_info

    def _append_log(self, pil_image: Image.Image, predicted_class: str, prob_dict: Dict[str, float]) -> Dict:
        """Append a row to CSV with timestamp, predicted_class, top1_prob, probs (json), image_hash (optional)"""
        project_root = Path(__file__).resolve().parents[2]
        log_path = project_root / "saved_model" / "prediction_logs.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # compute simple image hash (size + first bytes) to avoid storing files
        buf = io.BytesIO()
        pil_image.resize((64, 64)).convert("RGB").save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        img_hash = str(abs(hash(img_bytes)))[:12]

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_class": predicted_class,
            "top1_prob": float(prob_dict.get(predicted_class, 0.0)),
            "probs_json": json.dumps(prob_dict),
            "image_hash": img_hash,
        }

        header = ["timestamp", "predicted_class", "top1_prob", "probs_json", "image_hash"]
        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return row


# singleton
_model_instance: SkinDiseaseModel | None = None


def get_model(device: str = "cpu") -> SkinDiseaseModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = SkinDiseaseModel(device=device)
    return _model_instance
