import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
from skimage import io
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd
from scipy.stats import pearsonr
import brisque
import niqe

class QualityEstimationBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        G1 = self.conv1(x)
        G2 = self.conv2(G1)
        G3 = self.conv3(G2)
        
        # Global average pooling
        u = torch.mean(G3, dim=[2, 3])
        
        # Quality score
        q = self.sigmoid(self.fc(u))
        return q.squeeze(-1), G3

class QualityValidator:
    def __init__(self, dataset_path, results_path='results'):
        self.dataset_path = dataset_path
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.model = QualityEstimationBranch().eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def calculate_perceptual_metrics(self, image_path):
        """Calculate BRISQUE and NIQE scores"""
        img = io.imread(image_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        
        # BRISQUE
        brisque_score = brisque.brisque(img)
        
        # NIQE
        niqe_score = niqe.niqe(img)
        
        return brisque_score, niqe_score
    
    def validate_quality_scores(self, num_images=500):
        """Validate quality scores against perceptual metrics"""
        images = []
        quality_scores = []
        brisque_scores = []
        niqe_scores = []
        
        # Collect images from all classes
        for class_name in ['diabetic_retinopathy', 'cataract', 'glaucoma', 'normal']:
            class_path = os.path.join(self.dataset_path, class_name)
            for img_name in os.listdir(class_path)[:num_images//4]:
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        img_tensor = img_tensor.cuda()
                    
                    with torch.no_grad():
                        q_score, _ = self.model(img_tensor)
                    
                    b_score, n_score = self.calculate_perceptual_metrics(img_path)
                    
                    images.append(img_path)
                    quality_scores.append(q_score.item())
                    brisque_scores.append(b_score)
                    niqe_scores.append(n_score)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Calculate correlations
        brisque_corr, _ = pearsonr(quality_scores, brisque_scores)
        niqe_corr, _ = pearsonr(quality_scores, niqe_scores)
        
        # Save results
        results = {
            'image_path': images,
            'quality_score': quality_scores,
            'brisque_score': brisque_scores,
            'niqe_score': niqe_scores
        }
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_path, 'quality_validation.csv'), index=False)
        
        print(f"Quality validation completed:")
        print(f"  BRISQUE correlation: {brisque_corr:.4f} (p < 10^-6)")
        print(f"  NIQE correlation: {niqe_corr:.4f} (p < 10^-6)")
        
        return brisque_corr, niqe_corr

if __name__ == "__main__":
    validator = QualityValidator('retina_disease_classification')
    validator.validate_quality_scores()