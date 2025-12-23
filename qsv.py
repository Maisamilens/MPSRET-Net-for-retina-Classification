import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from scipy.stats import pearsonr
import torch.nn.functional as F

# =========================
# Configuration
# =========================
DATA_ROOT = "retina_disease_classification"
MODEL_PATH = "outputs/best_model_3rd.pt"
RESULTS_DIR = "results"
RESULTS_FILE = "quality_score_validation.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# Quality proxy functions
# =========================
def sharpness_laplacian(img_gray):
    return np.abs(np.diff(img_gray.astype(np.float32), axis=0)).mean()

def contrast_std(img_gray):
    return img_gray.std()

def naturalness_score(img_gray):
    mean, std = img_gray.mean(), img_gray.std() + 1e-6
    return abs(mean - 128) / std

# =========================
# Load model safely
# =========================
print("Loading model architecture and weights...")

from mpsretnet import MPSRetNet

# Minimal config class
class DummyCfg:
    num_classes = len(CLASSES)
    use_transfer_learning = False
    backbone = 'resnet50'
    # add any other fields required by MPSRetNet __init__ if needed

cfg = DummyCfg()
model = MPSRetNet(cfg)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle different checkpoint formats
if 'model' in checkpoint:
    state_dict = checkpoint['model']
elif 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint  # assume pure state dict

model.load_state_dict(state_dict)
model.eval().to(DEVICE)
print("Model loaded successfully.")

# =========================
# Inference loop
# =========================
records = []

with torch.no_grad():
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_ROOT, cls)
        for img_name in os.listdir(cls_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            # Forward pass
            class_logits, quality_score = model(img_tensor)

            probs = F.softmax(class_logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).item()
            quality_score = quality_score.item()

            # Compute image proxies
            img_np = np.array(img.resize((224, 224)))
            img_gray = np.mean(img_np, axis=2).astype(np.uint8)
            brisque_proxy = 0.6 * sharpness_laplacian(img_gray) + 0.4 * contrast_std(img_gray)
            niqe_proxy = naturalness_score(img_gray)

            records.append({
                "image": img_name,
                "class": cls,
                "quality_score": quality_score,
                "brisque_proxy": brisque_proxy,
                "niqe_proxy": niqe_proxy,
                "prediction_entropy": entropy
            })

# =========================
# Correlation analysis
# =========================
df = pd.DataFrame(records)

r_b, p_b = pearsonr(df["quality_score"], df["brisque_proxy"])
r_n, p_n = pearsonr(df["quality_score"], df["niqe_proxy"])
r_e, p_e = pearsonr(df["quality_score"], df["prediction_entropy"])

summary = pd.DataFrame({
    "Metric": ["BRISQUE-like", "NIQE-like", "Prediction Entropy"],
    "Pearson_r": [r_b, r_n, r_e],
    "p_value": [p_b, p_n, p_e]
})

# =========================
# Save results
# =========================
output_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Per_Image_Results", index=False)
    summary.to_excel(writer, sheet_name="Correlation_Summary", index=False)

print("\n✅ Quality Score Validation Completed")
print(summary)
print(f"\n📁 Results saved to: {output_path}")
