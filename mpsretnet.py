#!/usr/bin/env python3
"""
MPS-RetNet: Multi-Scale Prototype-Guided Semi-Supervised Framework
for Retinal Disease Classification - COMPLETE IMPLEMENTATION

This version includes:
1. Correct data paths for Google Colab
2. Full training pipeline with early stopping
3. External validation on Messidor-2 (1744 images)
4. Saliency maps for 100 randomly selected test images
5. Complete metrics and visualizations
6. All results saved to results folder

Author: Based on paper by Maisam Abbas, Ran-Zan Wang
"""

import os
import sys
import random
import time
import copy
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, auc, cohen_kappa_score, classification_report,
    silhouette_score, davies_bouldin_score
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from scipy import stats as scipy_stats

from tqdm.auto import tqdm

# Matplotlib settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.facecolor': 'white'
})

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=" * 60)
print(f"MPS-RetNet: Multi-Scale Prototype-Guided Semi-Supervised Framework")
print(f"=" * 60)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ==================== CONFIGURATION ====================
class Config:
    """Configuration class for all hyperparameters and paths"""
    
    # ===== DATA PATHS =====
    # Primary dataset paths (adjust these for your environment)
    # For Google Colab:
    data_root = "retina_data/retina_disease_classification"
    external_data_root = "messidor"
    
    
    # ===== CLASSES =====
    classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    num_classes = 4
    
    # ===== IMAGE SETTINGS =====
    image_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # ===== TRAINING SETTINGS =====
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 0.01
    grad_clip_norm = 2.0
    early_stopping_patience = 15
    
    # ===== SEMI-SUPERVISED SETTINGS =====
    labeled_ratio = 0.26
    ema_decay = 0.999
    consistency_weight = 1.0
    confidence_threshold = 0.95
    rampup_epochs = 5
    
    # ===== MODEL ARCHITECTURE =====
    embed_dim = 512
    num_heads = 8
    num_transformer_blocks = 2
    num_prototypes = 8
    prototype_loss_weight = 0.1
    quality_channels = [32, 64, 128]
    dilation_rates = [3, 6, 9]
    focal_gamma = 2.0
    
    # ===== EVALUATION =====
    seeds = [42, 123, 456, 789, 1024]
    n_bootstrap = 1000
    bootstrap_ci = 0.95
    calibration_bins = 10
    
    # ===== OUTPUT PATHS =====
    results_dir = "results"
    checkpoint_dir = "checkpoints"


def find_data_path(config):
    """Find the correct data path from alternatives"""
    # Check primary path
    if os.path.exists(config.data_root):
        subdirs = [d for d in os.listdir(config.data_root) 
                   if os.path.isdir(os.path.join(config.data_root, d))]
        if any(c in subdirs or c.lower() in [s.lower() for s in subdirs] 
               for c in config.classes):
            return config.data_root
    
    # Check alternatives
    for path in config.alternative_data_roots:
        if os.path.exists(path):
            subdirs = [d for d in os.listdir(path) 
                       if os.path.isdir(os.path.join(path, d))]
            if any(c in subdirs or c.lower() in [s.lower() for s in subdirs] 
                   for c in config.classes):
                return path
    
    return None


def find_external_path(config):
    """Find the correct external validation data path"""
    for path in [config.external_data_root] + config.alternative_external_roots:
        if os.path.exists(path):
            subdirs = [d for d in os.listdir(path) 
                       if os.path.isdir(os.path.join(path, d))]
            if any(c in subdirs or c.lower() in [s.lower() for s in subdirs] 
                   for c in config.classes):
                return path
    return None


config = Config()

# Create output directories
os.makedirs(config.results_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ==================== DATA AUGMENTATION ====================
class WeakAugmentation:
    """Weak augmentation for labeled data and teacher network"""
    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, x):
        return self.transform(x)


class StrongAugmentation:
    """Strong augmentation for unlabeled data student network"""
    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))
        ])
    
    def __call__(self, x):
        return self.transform(x)


class TestTransform:
    """Transform for validation and test data"""
    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, x):
        return self.transform(x)


# ==================== DATASETS ====================
class SemiSupervisedDataset(Dataset):
    """Dataset for semi-supervised learning"""
    def __init__(self, samples, labels, weak_transform, strong_transform=None, is_labeled=True):
        self.samples = samples
        self.labels = labels
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform or weak_transform
        self.is_labeled = is_labeled
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        weak_img = self.weak_transform(img)
        
        if self.is_labeled:
            return weak_img, self.labels[idx], idx
        else:
            strong_img = self.strong_transform(img)
            return weak_img, strong_img, idx


class SimpleDataset(Dataset):
    """Simple dataset for evaluation"""
    def __init__(self, samples, labels, transform):
        self.samples = samples
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        return self.transform(img), self.labels[idx], img_path


# ==================== MODEL COMPONENTS ====================
class MultiScaleSpatialPooling(nn.Module):
    """Multi-scale spatial pooling module for context aggregation"""
    def __init__(self, in_channels, out_channels, dilation_rates):
        super().__init__()
        
        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolution branches
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in dilation_rates
        ])
        
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Projection layer
        total_channels = out_channels * (2 + len(dilation_rates))
        self.projection = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Collect features from all branches
        features = [self.conv1x1(x)]
        features.extend([conv(x) for conv in self.dilated_convs])
        gap_feat = F.interpolate(self.gap(x), size, mode='bilinear', align_corners=False)
        features.append(gap_feat)
        
        # Concatenate and project
        concat = torch.cat(features, dim=1)
        return self.projection(concat)


class SpatialAttentionBlock(nn.Module):
    """Transformer block with spatial attention"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized)
        x = x + self.dropout(attended)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class PrototypeModule(nn.Module):
    """Prototype-guided representation module"""
    def __init__(self, embed_dim, num_prototypes, temperature=0.1):
        super().__init__()
        
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.num_prototypes = num_prototypes
    
    def forward(self, z):
        # Normalize embeddings and prototypes
        z_norm = F.normalize(z, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        
        # Compute similarity scores
        similarity = torch.mm(z_norm, p_norm.t()) / self.temperature.abs().clamp(min=0.01)
        
        # Soft attention weights
        attention = F.softmax(similarity, dim=-1)
        
        # Weighted prototype aggregation with residual
        aggregated = torch.mm(attention, self.prototypes)
        enhanced = z + aggregated
        
        return enhanced, attention
    
    def diversity_loss(self):
        """Compute diversity regularization loss"""
        p_norm = F.normalize(self.prototypes, dim=-1)
        similarity_matrix = torch.mm(p_norm, p_norm.t())
        identity = torch.eye(self.num_prototypes, device=self.prototypes.device)
        return torch.norm(similarity_matrix - identity, p='fro') ** 2


class QualityEstimationBranch(nn.Module):
    """Quality estimation branch for adaptive supervision"""
    def __init__(self, channels=[32, 64, 128]):
        super().__init__()
        
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], 1)
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        quality = torch.sigmoid(self.fc(pooled)).squeeze(-1)
        return quality


class MPSRetNet(nn.Module):
    """MPS-RetNet: Main model architecture"""
    def __init__(self, cfg):
        super().__init__()
        
        # ConvNeXt backbone
        convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(convnext.features.children()))
        
        # Multi-scale spatial pooling
        self.mssp = MultiScaleSpatialPooling(1024, cfg.embed_dim, cfg.dilation_rates)
        
        # Spatial attention blocks
        self.attention_blocks = nn.ModuleList([
            SpatialAttentionBlock(cfg.embed_dim, cfg.num_heads)
            for _ in range(cfg.num_transformer_blocks)
        ])
        
        # Prototype module
        self.prototype_module = PrototypeModule(cfg.embed_dim, cfg.num_prototypes)
        
        # Quality estimation branch
        self.quality_branch = QualityEstimationBranch(cfg.quality_channels)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.num_classes)
        )
    
    def forward(self, x, return_features=False, return_quality=False):
        # Backbone feature extraction
        backbone_features = self.backbone(x)
        
        # Multi-scale spatial pooling
        mssp_features = self.mssp(backbone_features)
        
        # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = mssp_features.shape
        tokens = mssp_features.flatten(2).transpose(1, 2)
        
        # Apply spatial attention blocks
        for block in self.attention_blocks:
            tokens = block(tokens)
        
        # Global average pooling over spatial positions
        embedding = tokens.mean(dim=1)
        
        # Prototype-guided enhancement
        enhanced, _ = self.prototype_module(embedding)
        
        # Classification
        logits = self.classifier(enhanced)
        
        output = {'logits': logits}
        
        if return_features:
            output['embedding'] = embedding
        
        if return_quality:
            output['quality_score'] = self.quality_branch(x)
        
        return output
    
    def get_diversity_loss(self):
        return self.prototype_module.diversity_loss()
    
    def get_prototypes(self):
        return self.prototype_module.prototypes.detach().cpu().numpy()


class EMAModel:
    """Exponential Moving Average model for teacher network"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    (1 - self.decay) * param.data + 
                    self.decay * self.shadow[name]
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


# ==================== LOSS FUNCTIONS ====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss


def consistency_loss(student_pred, teacher_pred, mask=None):
    """Compute consistency loss between student and teacher predictions"""
    loss = F.mse_loss(student_pred, teacher_pred, reduction='none').sum(dim=-1)
    
    if mask is not None:
        return (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss.mean()


def cosine_rampup(epoch, rampup_epochs):
    """Cosine ramp-up schedule for consistency weight"""
    if epoch >= rampup_epochs:
        return 1.0
    return (1 - np.cos(np.pi * epoch / rampup_epochs)) / 2


# ==================== DATA PREPARATION ====================
def load_images_from_directory(root_path, classes):
    """Load all images from a directory organized by class folders"""
    samples = []
    labels = []
    
    root = Path(root_path)
    if not root.exists():
        print(f"  Path does not exist: {root_path}")
        return samples, labels
    
    # Get actual subdirectories
    subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    
    for class_idx, class_name in enumerate(classes):
        # Try exact match first, then case-insensitive
        class_dir = None
        if class_name in subdirs:
            class_dir = subdirs[class_name]
        else:
            # Case-insensitive search
            for subdir_name, subdir_path in subdirs.items():
                if class_name.lower() == subdir_name.lower():
                    class_dir = subdir_path
                    break
                # Also check partial matches
                if class_name.lower().replace('_', '') in subdir_name.replace('_', ''):
                    class_dir = subdir_path
                    break
        
        if class_dir is None:
            print(f"  Warning: Class folder not found for '{class_name}'")
            continue
        
        # Load images
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        image_files = [
            p for p in class_dir.glob('*') 
            if p.suffix.lower() in valid_extensions
        ]
        
        for img_path in sorted(image_files):
            samples.append(str(img_path))
            labels.append(class_idx)
        
        print(f"  {class_name}: {len(image_files)} images")
    
    return samples, labels


def prepare_data(cfg, seed=42):
    """Prepare datasets for training, validation, and testing"""
    set_seed(seed)
    
    # Find data path
    data_root = find_data_path(cfg)
    if data_root is None:
        print("\nERROR: Could not find dataset. Please ensure data is in one of these locations:")
        for path in [cfg.data_root] + cfg.alternative_data_roots:
            print(f"  - {path}")
        print("\nExpected folder structure:")
        print("  data_root/")
        for c in cfg.classes:
            print(f"    {c}/")
        raise FileNotFoundError("Dataset not found")
    
    print(f"\nLoading dataset from: {data_root}")
    samples, labels = load_images_from_directory(data_root, cfg.classes)
    
    if len(samples) == 0:
        raise ValueError(f"No images found in {data_root}")
    
    print(f"Total images: {len(samples)}")
    
    # Split data
    # First split: 90% train+val, 10% test
    tv_samples, test_samples, tv_labels, test_labels = train_test_split(
        samples, labels, test_size=0.1, stratify=labels, random_state=seed
    )
    
    # Second split: labeled vs unlabeled from train+val
    labeled_samples, unlabeled_samples, labeled_labels, _ = train_test_split(
        tv_samples, tv_labels, 
        test_size=1 - cfg.labeled_ratio / 0.9,
        stratify=tv_labels, random_state=seed
    )
    
    # Third split: train vs validation from labeled
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        labeled_samples, labeled_labels,
        test_size=0.2, stratify=labeled_labels, random_state=seed
    )
    
    print(f"\nData splits:")
    print(f"  Training (labeled): {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    print(f"  Unlabeled: {len(unlabeled_samples)}")
    
    # Create transforms
    weak_transform = WeakAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    strong_transform = StrongAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    test_transform = TestTransform(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    
    # Create datasets and loaders
    train_dataset = SemiSupervisedDataset(
        train_samples, train_labels, weak_transform, is_labeled=True
    )
    unlabeled_dataset = SemiSupervisedDataset(
        unlabeled_samples, [0] * len(unlabeled_samples),
        weak_transform, strong_transform, is_labeled=False
    )
    val_dataset = SimpleDataset(val_samples, val_labels, test_transform)
    test_dataset = SimpleDataset(test_samples, test_labels, test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )
    
    return {
        'train_loader': train_loader,
        'unlabeled_loader': unlabeled_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'test_samples': test_samples,
        'test_labels': test_labels,
        'val_samples': val_samples,
        'val_labels': val_labels
    }


def prepare_external_data(cfg):
    """Prepare external validation dataset (Messidor-2)"""
    external_root = find_external_path(cfg)
    
    if external_root is None:
        print("\nExternal validation dataset not found. Checked locations:")
        for path in [cfg.external_data_root] + cfg.alternative_external_roots:
            print(f"  - {path}")
        return None
    
    print(f"\nLoading external dataset from: {external_root}")
    samples, labels = load_images_from_directory(external_root, cfg.classes)
    
    if len(samples) == 0:
        print("No images found in external dataset")
        return None
    
    print(f"External total images: {len(samples)}")
    
    test_transform = TestTransform(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    dataset = SimpleDataset(samples, labels, test_transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    
    return {
        'loader': loader,
        'samples': samples,
        'labels': labels
    }


# ==================== TRAINING ====================
def train_epoch(model, ema, train_loader, unlabeled_loader, optimizer, cfg, epoch, device):
    """Train for one epoch"""
    model.train()
    
    # Calculate consistency weight with ramp-up
    consistency_w = cfg.consistency_weight * cosine_rampup(epoch, cfg.rampup_epochs)
    focal_loss_fn = FocalLoss(cfg.focal_gamma)
    
    # Metrics
    total_loss = 0
    sup_loss_total = 0
    cons_loss_total = 0
    proto_loss_total = 0
    high_conf_count = 0
    high_conf_total = 0
    
    # Create iterators
    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)
    num_batches = min(len(train_loader), len(unlabeled_loader))
    
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
    
    for batch_idx in pbar:
        # Get labeled batch
        try:
            labeled_imgs, labels, _ = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(train_loader)
            labeled_imgs, labels, _ = next(labeled_iter)
        
        # Get unlabeled batch
        try:
            weak_imgs, strong_imgs, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            weak_imgs, strong_imgs, _ = next(unlabeled_iter)
        
        # Move to device
        labeled_imgs = labeled_imgs.to(device)
        labels = labels.to(device)
        weak_imgs = weak_imgs.to(device)
        strong_imgs = strong_imgs.to(device)
        
        # Forward pass for labeled data
        output = model(labeled_imgs, return_quality=True)
        quality_scores = output['quality_score']
        
        # Quality-weighted supervised loss
        focal = focal_loss_fn(output['logits'], labels)
        sup_loss = (quality_scores * focal).mean()
        
        # Generate pseudo-labels from teacher (EMA model)
        ema.apply_shadow()
        with torch.no_grad():
            teacher_output = model(weak_imgs)
            teacher_probs = F.softmax(teacher_output['logits'], dim=-1)
        ema.restore()
        
        # Student predictions on strong augmented images
        student_output = model(strong_imgs)
        student_probs = F.softmax(student_output['logits'], dim=-1)
        
        # Confidence-based masking
        max_probs, _ = teacher_probs.max(dim=-1)
        mask = (max_probs >= cfg.confidence_threshold).float()
        
        high_conf_count += mask.sum().item()
        high_conf_total += len(mask)
        
        # Consistency loss
        cons_loss = consistency_loss(student_probs, teacher_probs, mask)
        
        # Prototype diversity loss
        proto_loss = model.get_diversity_loss()
        
        # Total loss
        loss = sup_loss + consistency_w * cons_loss + cfg.prototype_loss_weight * proto_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        
        # Update EMA
        ema.update()
        
        # Update metrics
        total_loss += loss.item()
        sup_loss_total += sup_loss.item()
        cons_loss_total += cons_loss.item()
        proto_loss_total += proto_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'sup': f'{sup_loss.item():.4f}',
            'cons': f'{cons_loss.item():.4f}'
        })
    
    return {
        'total': total_loss / num_batches,
        'sup': sup_loss_total / num_batches,
        'cons': cons_loss_total / num_batches,
        'proto': proto_loss_total / num_batches,
        'hc_ratio': high_conf_count / high_conf_total if high_conf_total > 0 else 0
    }


@torch.no_grad()
def validate(model, loader, device, return_embeddings=False):
    """Validate model on a dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = []
    total_loss = 0
    
    for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        
        output = model(images, return_features=return_embeddings)
        probs = F.softmax(output['logits'], dim=-1)
        
        loss = F.cross_entropy(output['logits'], labels.to(device))
        total_loss += loss.item()
        
        preds = output['logits'].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        
        if return_embeddings:
            all_embeddings.extend(output['embedding'].cpu().numpy())
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix and specificity
    cm = confusion_matrix(all_labels, all_preds)
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    # AUC
    try:
        auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc_score = 0
    
    results = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'auc': auc_score,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'precision_per_class': precision_c,
        'recall_per_class': recall_c,
        'f1_per_class': f1_c,
        'specificity_per_class': np.array(specificity)
    }
    
    if return_embeddings:
        results['embeddings'] = np.array(all_embeddings)
    
    return results


def calibration_metrics(labels, probs, num_bins=10):
    """Compute calibration metrics (ECE and Brier score)"""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Expected Calibration Error
    ece = 0
    bin_data = []
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += np.abs(avg_acc - avg_conf) * prop_in_bin
            bin_data.append({'confidence': avg_conf, 'accuracy': avg_acc})
    
    # Brier score
    one_hot = np.zeros((len(labels), probs.shape[1]))
    one_hot[np.arange(len(labels)), labels] = 1
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    
    return {'ece': ece, 'brier': brier, 'bins': bin_data}


def bootstrap_confidence_interval(labels, predictions, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for accuracy"""
    accuracies = []
    n = len(labels)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        acc = accuracy_score(labels[indices], predictions[indices])
        accuracies.append(acc)
    
    lower = np.percentile(accuracies, (1 - ci) / 2 * 100)
    upper = np.percentile(accuracies, (1 + ci) / 2 * 100)
    
    return lower, upper


# ==================== CHECKPOINTS ====================
def save_checkpoint(model, ema, optimizer, scheduler, epoch, history, cfg, filename, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow': copy.deepcopy(ema.shadow),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history
    }
    
    path = os.path.join(cfg.checkpoint_dir, filename)
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")
    
    if is_best:
        best_path = os.path.join(cfg.checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"  Best model saved: {best_path}")


def load_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


# ==================== VISUALIZATION ====================
def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', lw=2, label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', lw=2, label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, [x * 100 for x in history['val_acc']], 'g-', lw=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 and Kappa
    axes[0, 2].plot(epochs, [x * 100 for x in history['val_f1']], 'b-', lw=2, label='F1')
    axes[0, 2].plot(epochs, [x * 100 for x in history['val_kappa']], 'r-', lw=2, label='Kappa')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score (%)')
    axes[0, 2].set_title('F1 Score and Cohen\'s Kappa')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(epochs, history['val_auc'], 'm-', lw=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('Validation ROC-AUC')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1, 1].plot(epochs, history['sup_loss'], 'b-', lw=2, label='Supervised')
    axes[1, 1].plot(epochs, history['cons_loss'], 'r-', lw=2, label='Consistency')
    axes[1, 1].plot(epochs, history['proto_loss'], 'g-', lw=2, label='Prototype')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Component Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # High confidence ratio
    axes[1, 2].plot(epochs, [x * 100 for x in history['hc_ratio']], 'c-', lw=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('High Confidence Ratio (%)')
    axes[1, 2].set_title('Pseudo-Label Utilization')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(labels, predictions, classes, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curves(labels, probs, classes, save_path):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_calibration(calibration_data, save_path):
    """Plot reliability diagram"""
    plt.figure(figsize=(8, 8))
    
    if calibration_data['bins']:
        confs = [b['confidence'] for b in calibration_data['bins']]
        accs = [b['accuracy'] for b in calibration_data['bins']]
        plt.bar(confs, accs, width=0.08, alpha=0.7, color='steelblue', edgecolor='navy')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives')
    plt.title(f"Reliability Diagram\nECE = {calibration_data['ece']:.4f}, Brier = {calibration_data['brier']:.4f}")
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne(model, loader, classes, device, save_path):
    """Generate t-SNE visualization of embeddings"""
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels, _ in tqdm(loader, desc="Extracting embeddings"):
            output = model(images.to(device), return_features=True)
            embeddings.extend(output['embedding'].cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    prototypes = model.get_prototypes()
    
    # Combine embeddings and prototypes for t-SNE
    combined = np.vstack([embeddings, prototypes])
    
    # Fit t-SNE
    perplexity = min(30, len(combined) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    combined_2d = tsne.fit_transform(combined)
    
    embeddings_2d = combined_2d[:len(embeddings)]
    prototypes_2d = combined_2d[len(embeddings):]
    
    # Compute clustering metrics
    sil_score = silhouette_score(embeddings, labels)
    db_score = davies_bouldin_score(embeddings, labels)
    
    # Plot
    plt.figure(figsize=(12, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=color, s=50, alpha=0.6, label=class_name
        )
    
    # Plot prototypes
    plt.scatter(
        prototypes_2d[:, 0], prototypes_2d[:, 1],
        c='black', marker='X', s=300, label='Prototypes', zorder=5
    )
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE Visualization\nSilhouette = {sil_score:.3f}, Davies-Bouldin = {db_score:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    
    return {'silhouette': sil_score, 'davies_bouldin': db_score}


# ==================== GRAD-CAM ====================
class GradCAM:
    """Grad-CAM for model interpretability"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        logits = output['logits']
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot)
        
        # Compute Grad-CAM
        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * self.activations[0]).sum(dim=0))
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class


def plot_gradcam_samples(model, samples, labels, classes, device, save_path):
    """Plot Grad-CAM visualizations for one sample per class"""
    model.eval()
    
    # Get target layer (last conv layer in backbone)
    target_layer = model.backbone[-1][-1]
    gradcam = GradCAM(model, target_layer)
    
    test_transform = TestTransform(config.image_size, config.imagenet_mean, config.imagenet_std)
    
    fig, axes = plt.subplots(len(classes), 3, figsize=(12, 4 * len(classes)))
    
    for class_idx, class_name in enumerate(classes):
        # Find a sample of this class
        sample_indices = [i for i, l in enumerate(labels) if l == class_idx]
        
        if not sample_indices:
            continue
        
        sample_idx = sample_indices[0]
        img = Image.open(samples[sample_idx]).convert('RGB')
        
        # Get Grad-CAM
        x = test_transform(img).unsqueeze(0).to(device)
        cam, _ = gradcam(x, class_idx)
        
        # Resize CAM to image size
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255
        
        # Display original image
        img_display = img.resize((224, 224))
        axes[class_idx, 0].imshow(img_display)
        axes[class_idx, 0].set_title(f'{class_name}')
        axes[class_idx, 0].axis('off')
        
        # Display Grad-CAM heatmap
        axes[class_idx, 1].imshow(cam_resized, cmap='jet')
        axes[class_idx, 1].set_title('Grad-CAM')
        axes[class_idx, 1].axis('off')
        
        # Display overlay
        overlay = 0.6 * np.array(img_display) / 255 + 0.4 * plt.cm.jet(cam_resized)[:, :, :3]
        axes[class_idx, 2].imshow(overlay)
        axes[class_idx, 2].set_title('Overlay')
        axes[class_idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def generate_saliency_maps_100(model, samples, labels, classes, device, save_dir, n=100):
    """Generate saliency maps for 100 randomly selected test images"""
    model.eval()
    
    # Get target layer
    target_layer = model.backbone[-1][-1]
    gradcam = GradCAM(model, target_layer)
    
    test_transform = TestTransform(config.image_size, config.imagenet_mean, config.imagenet_std)
    
    # Create output directory
    saliency_dir = os.path.join(save_dir, 'saliency_maps_100')
    os.makedirs(saliency_dir, exist_ok=True)
    
    # Select random indices
    n_samples = min(n, len(samples))
    indices = np.random.choice(len(samples), n_samples, replace=False)
    
    print(f"\nGenerating saliency maps for {n_samples} images...")
    
    saliency_data = []
    
    for i, idx in enumerate(tqdm(indices, desc="Generating saliency maps")):
        try:
            img = Image.open(samples[idx]).convert('RGB')
            x = test_transform(img).unsqueeze(0).to(device)
            
            # Get Grad-CAM
            cam, pred_class = gradcam(x, labels[idx])
            
            # Resize CAM
            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255
            
            # Save individual figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            img_display = img.resize((224, 224))
            axes[0].imshow(img_display)
            axes[0].set_title(f'True: {classes[labels[idx]]}')
            axes[0].axis('off')
            
            axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title('Grad-CAM')
            axes[1].axis('off')
            
            overlay = 0.6 * np.array(img_display) / 255 + 0.4 * plt.cm.jet(cam_resized)[:, :, :3]
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(saliency_dir, f'saliency_{i+1:03d}_{classes[labels[idx]]}.png'), dpi=150)
            plt.close()
            
            # Store data for statistics
            saliency_data.append({
                'index': idx,
                'class': labels[idx],
                'mean_activation': float(cam.mean()),
                'max_activation': float(cam.max()),
                'area_above_threshold': float((cam > 0.5).mean())
            })
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    # Generate 10x10 grid
    print("\nGenerating 10x10 grid...")
    n_grid = min(100, len(indices))
    grid_size = int(np.ceil(np.sqrt(n_grid)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices[:n_grid]):
        try:
            img = Image.open(samples[idx]).convert('RGB')
            x = test_transform(img).unsqueeze(0).to(device)
            
            cam, _ = gradcam(x, labels[idx])
            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255
            
            img_display = np.array(img.resize((224, 224)))
            overlay = 0.6 * img_display / 255 + 0.4 * plt.cm.jet(cam_resized)[:, :, :3]
            
            axes[i].imshow(overlay)
            axes[i].axis('off')
            axes[i].set_title(classes[labels[idx]][:3], fontsize=6)
        except:
            axes[i].axis('off')
    
    # Hide remaining axes
    for i in range(n_grid, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'saliency_maps_100_grid.pdf'), dpi=300)
    plt.close()
    
    print(f"Saved: {saliency_dir}/ and saliency_maps_100_grid.pdf")
    
    return saliency_data


def saliency_statistical_analysis(model, samples, labels, classes, device, save_path, n=100):
    """Compute statistical analysis of saliency patterns"""
    model.eval()
    
    target_layer = model.backbone[-1][-1]
    gradcam = GradCAM(model, target_layer)
    test_transform = TestTransform(config.image_size, config.imagenet_mean, config.imagenet_std)
    
    # Collect statistics per class
    stats_data = {i: {'mean': [], 'area': []} for i in range(len(classes))}
    
    indices = np.random.choice(len(samples), min(n, len(samples)), replace=False)
    
    print("\nComputing saliency statistics...")
    for idx in tqdm(indices, desc="Analyzing saliency"):
        try:
            img = Image.open(samples[idx]).convert('RGB')
            x = test_transform(img).unsqueeze(0).to(device)
            
            cam, _ = gradcam(x, labels[idx])
            
            stats_data[labels[idx]]['mean'].append(float(cam.mean()))
            stats_data[labels[idx]]['area'].append(float((cam > 0.5).mean()))
        except:
            continue
    
    # ANOVA test
    groups = [stats_data[i]['mean'] for i in range(len(classes)) if len(stats_data[i]['mean']) > 0]
    if len(groups) > 1 and all(len(g) > 0 for g in groups):
        f_stat, p_value = scipy_stats.f_oneway(*groups)
    else:
        f_stat, p_value = 0, 1
    
    # Pearson correlation
    all_means = [v for s in stats_data.values() for v in s['mean']]
    all_areas = [v for s in stats_data.values() for v in s['area']]
    
    if len(all_means) > 2:
        r_val, r_pval = scipy_stats.pearsonr(all_means, all_areas)
    else:
        r_val, r_pval = 0, 1
    
    print(f"\nStatistical Analysis:")
    print(f"  ANOVA: F = {f_stat:.2f}, p = {p_value:.2e}")
    print(f"  Pearson: r = {r_val:.4f}, p = {r_pval:.2e}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot of mean activations
    box_data = [stats_data[i]['mean'] for i in range(len(classes))]
    bp = axes[0].boxplot(box_data, labels=classes, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_xlabel('Disease Class')
    axes[0].set_ylabel('Mean Activation')
    axes[0].set_title(f'Saliency Intensity by Class\nANOVA: F = {f_stat:.2f}, p = {p_value:.2e}')
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot of intensity vs area
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        if stats_data[i]['mean']:
            axes[1].scatter(
                stats_data[i]['mean'], stats_data[i]['area'],
                c=color, label=class_name, alpha=0.6, s=50
            )
    axes[1].set_xlabel('Mean Activation')
    axes[1].set_ylabel('Area Above Threshold')
    axes[1].set_title(f'Activation Intensity vs. Spatial Extent\nPearson: r = {r_val:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    
    return {
        'anova_f': f_stat,
        'anova_p': p_value,
        'pearson_r': r_val,
        'pearson_p': r_pval
    }


# ==================== MAIN TRAINING FUNCTION ====================
def train_model(cfg, seed=42):
    """Main training function"""
    print("\n" + "=" * 60)
    print("MPS-RetNet Training")
    print("=" * 60)
    
    set_seed(seed)
    
    # Prepare data
    data = prepare_data(cfg, seed)
    
    # Create model
    model = MPSRetNet(cfg).to(device)
    ema = EMAModel(model, cfg.ema_decay)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_kappa': [], 'val_auc': [], 'sup_loss': [], 'cons_loss': [],
        'proto_loss': [], 'hc_ratio': [], 'lr': []
    }
    
    best_acc = 0
    patience_counter = 0
    start_time = time.time()
    
    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    
    for epoch in range(cfg.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, ema, data['train_loader'], data['unlabeled_loader'],
            optimizer, cfg, epoch, device
        )
        
        # Validate with EMA model
        ema.apply_shadow()
        val_metrics = validate(model, data['val_loader'], device)
        ema.restore()
        
        # Update scheduler
        scheduler.step(val_metrics['accuracy'])
        
        # Record history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_kappa'].append(val_metrics['kappa'])
        history['val_auc'].append(val_metrics['auc'])
        history['sup_loss'].append(train_metrics['sup'])
        history['cons_loss'].append(train_metrics['cons'])
        history['proto_loss'].append(train_metrics['proto'])
        history['hc_ratio'].append(train_metrics['hc_ratio'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}:")
        print(f"  Train Loss: {train_metrics['total']:.4f} (Sup: {train_metrics['sup']:.4f}, Cons: {train_metrics['cons']:.4f})")
        print(f"  Val Acc: {val_metrics['accuracy'] * 100:.2f}%, F1: {val_metrics['f1'] * 100:.2f}%, Kappa: {val_metrics['kappa'] * 100:.2f}%")
        print(f"  Val AUC: {val_metrics['auc']:.4f}, HC Ratio: {train_metrics['hc_ratio'] * 100:.1f}%")
        
        # Check for improvement
        is_best = val_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = val_metrics['accuracy']
            patience_counter = 0
            save_checkpoint(model, ema, optimizer, scheduler, epoch, history, cfg, f'checkpoint_epoch_{epoch + 1}.pt', is_best=True)
        else:
            patience_counter += 1
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, ema, optimizer, scheduler, epoch, history, cfg, f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc * 100:.2f}%")
    
    # Load best model
    best_checkpoint_path = os.path.join(cfg.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_checkpoint_path):
        load_checkpoint(model, best_checkpoint_path, device)
        print(f"Loaded best model from {best_checkpoint_path}")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = validate(model, data['test_loader'], device, return_embeddings=True)
    
    # Calibration
    calibration = calibration_metrics(test_metrics['labels'], test_metrics['probabilities'])
    
    # Confidence intervals
    ci_lower, ci_upper = bootstrap_confidence_interval(
        test_metrics['labels'], test_metrics['predictions']
    )
    
    test_metrics.update({
        'ece': calibration['ece'],
        'brier': calibration['brier'],
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    
    # Save final checkpoint
    save_checkpoint(model, ema, optimizer, scheduler, epoch, history, cfg, 'final_model.pt')
    
    return model, history, test_metrics, data, calibration, train_time


def generate_all_results(model, history, test_metrics, data, calibration, cfg):
    """Generate all results and visualizations"""
    print("\n" + "=" * 60)
    print("Generating Results")
    print("=" * 60)
    
    # Training history
    plot_training_history(history, f"{cfg.results_dir}/fig_training_history.pdf")
    
    # Confusion matrix
    plot_confusion_matrix(
        test_metrics['labels'], test_metrics['predictions'],
        cfg.classes, f"{cfg.results_dir}/fig_confusion_matrix.pdf"
    )
    
    # ROC curves
    plot_roc_curves(
        test_metrics['labels'], test_metrics['probabilities'],
        cfg.classes, f"{cfg.results_dir}/fig_roc.pdf"
    )
    
    # Calibration
    plot_calibration(calibration, f"{cfg.results_dir}/fig_calibration.pdf")
    
    # Grad-CAM samples
    plot_gradcam_samples(
        model, data['test_samples'], data['test_labels'],
        cfg.classes, device, f"{cfg.results_dir}/fig_gradcam.pdf"
    )
    
    # t-SNE visualization
    clustering_metrics = plot_tsne(
        model, data['test_loader'], cfg.classes, device,
        f"{cfg.results_dir}/fig_tsne.pdf"
    )
    
    # Statistical analysis of saliency
    saliency_stats = saliency_statistical_analysis(
        model, data['test_samples'], data['test_labels'],
        cfg.classes, device, f"{cfg.results_dir}/fig_statistical_analysis.pdf"
    )
    
    # Saliency maps for 100 images
    saliency_data = generate_saliency_maps_100(
        model, data['test_samples'], data['test_labels'],
        cfg.classes, device, cfg.results_dir, n=100
    )
    
    # Save CSV tables
    # Table 2: Class-wise metrics
    df_class = pd.DataFrame({
        'Class': cfg.classes + ['Macro Average'],
        'Sensitivity (%)': list(test_metrics['recall_per_class'] * 100) + [test_metrics['recall'] * 100],
        'Specificity (%)': list(test_metrics['specificity_per_class'] * 100) + [test_metrics['specificity_per_class'].mean() * 100],
        'Precision (%)': list(test_metrics['precision_per_class'] * 100) + [test_metrics['precision'] * 100],
        'F1 Score (%)': list(test_metrics['f1_per_class'] * 100) + [test_metrics['f1'] * 100]
    })
    df_class.to_csv(f"{cfg.results_dir}/table2_class_metrics.csv", index=False)
    print(f"Saved: {cfg.results_dir}/table2_class_metrics.csv")
    
    # Table 8: Overall metrics
    df_overall = pd.DataFrame({
        'Metric': ['Accuracy (%)', 'F1 Score (%)', 'Cohen\'s Kappa (%)', 'ROC-AUC',
                   'ECE', 'Brier Score', '95% CI Lower', '95% CI Upper',
                   'Silhouette Score', 'Davies-Bouldin Index', 'ANOVA F-statistic', 'Pearson r'],
        'Value': [
            f"{test_metrics['accuracy'] * 100:.2f}",
            f"{test_metrics['f1'] * 100:.2f}",
            f"{test_metrics['kappa'] * 100:.2f}",
            f"{test_metrics['auc']:.4f}",
            f"{test_metrics['ece']:.4f}",
            f"{test_metrics['brier']:.4f}",
            f"{test_metrics['ci_lower']:.4f}",
            f"{test_metrics['ci_upper']:.4f}",
            f"{clustering_metrics['silhouette']:.4f}",
            f"{clustering_metrics['davies_bouldin']:.4f}",
            f"{saliency_stats['anova_f']:.2f}",
            f"{saliency_stats['pearson_r']:.4f}"
        ]
    })
    df_overall.to_csv(f"{cfg.results_dir}/table8_overall_metrics.csv", index=False)
    print(f"Saved: {cfg.results_dir}/table8_overall_metrics.csv")
    
    # Training history CSV
    pd.DataFrame(history).to_csv(f"{cfg.results_dir}/training_history.csv", index=False)
    print(f"Saved: {cfg.results_dir}/training_history.csv")
    
    return clustering_metrics, saliency_stats


def external_validation(model, cfg):
    """Perform external validation on Messidor-2 dataset"""
    print("\n" + "=" * 60)
    print("External Validation (Messidor-2)")
    print("=" * 60)
    
    external_data = prepare_external_data(cfg)
    
    if external_data is None:
        print("External validation skipped - dataset not found")
        return None
    
    # Evaluate
    external_metrics = validate(model, external_data['loader'], device)
    external_calibration = calibration_metrics(
        external_metrics['labels'], external_metrics['probabilities']
    )
    
    ci_lower, ci_upper = bootstrap_confidence_interval(
        external_metrics['labels'], external_metrics['predictions']
    )
    
    print(f"\nExternal Validation Results ({len(external_data['samples'])} images):")
    print(f"  Accuracy: {external_metrics['accuracy'] * 100:.2f}%")
    print(f"  F1 Score: {external_metrics['f1'] * 100:.2f}%")
    print(f"  Cohen's Kappa: {external_metrics['kappa'] * 100:.2f}%")
    print(f"  ROC-AUC: {external_metrics['auc']:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Save external validation results
    df_external = pd.DataFrame({
        'Class': cfg.classes + ['Macro Average'],
        'Sensitivity (%)': list(external_metrics['recall_per_class'] * 100) + [external_metrics['recall'] * 100],
        'Specificity (%)': list(external_metrics['specificity_per_class'] * 100) + [external_metrics['specificity_per_class'].mean() * 100],
        'F1 Score (%)': list(external_metrics['f1_per_class'] * 100) + [external_metrics['f1'] * 100]
    })
    df_external.to_csv(f"{cfg.results_dir}/table4_external_validation.csv", index=False)
    print(f"Saved: {cfg.results_dir}/table4_external_validation.csv")
    
    # Confusion matrix for external data
    plot_confusion_matrix(
        external_metrics['labels'], external_metrics['predictions'],
        cfg.classes, f"{cfg.results_dir}/fig_confusion_matrix_external.pdf"
    )
    
    return {
        'metrics': external_metrics,
        'calibration': external_calibration,
        'ci': (ci_lower, ci_upper),
        'n_samples': len(external_data['samples'])
    }


def print_final_summary(test_metrics, calibration, clustering, saliency_stats, train_time, cfg, external=None):
    """Print final summary of all results"""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Test Accuracy':<25} {test_metrics['accuracy'] * 100:>14.2f}%")
    print(f"{'Test F1 Score':<25} {test_metrics['f1'] * 100:>14.2f}%")
    print("{:<25} {:>14.2f}%".format("Cohen's Kappa", test_metrics['kappa'] * 100))
    print(f"{'ROC-AUC':<25} {test_metrics['auc']:>15.4f}")
    print(f"{'ECE':<25} {calibration['ece']:>15.4f}")
    print(f"{'Brier Score':<25} {calibration['brier']:>15.4f}")
    print(f"{'95% CI':<25} [{test_metrics['ci_lower']:.4f}, {test_metrics['ci_upper']:.4f}]")
    print(f"{'Silhouette Score':<25} {clustering['silhouette']:>15.4f}")
    print(f"{'Davies-Bouldin Index':<25} {clustering['davies_bouldin']:>15.4f}")
    print(f"{'ANOVA F-statistic':<25} {saliency_stats['anova_f']:>15.2f}")
    print(f"{'Pearson r':<25} {saliency_stats['pearson_r']:>15.4f}")
    print(f"{'Training Time':<25} {train_time / 60:>14.2f} min")
    
    if external:
        print(f"\n{'External Validation':<25}")
        print("-" * 42)
        print(f"{'External Accuracy':<25} {external['metrics']['accuracy'] * 100:>14.2f}%")
        print(f"{'External F1 Score':<25} {external['metrics']['f1'] * 100:>14.2f}%")
        print(f"{'External Kappa':<25} {external['metrics']['kappa'] * 100:>14.2f}%")
        print(f"{'External 95% CI':<25} [{external['ci'][0]:.4f}, {external['ci'][1]:.4f}]")
        print(f"{'External N Samples':<25} {external['n_samples']:>15}")
    
    print("\n" + "-" * 42)
    print("\nClassification Report:")
    print(classification_report(
        test_metrics['labels'], test_metrics['predictions'],
        target_names=cfg.classes, digits=4
    ))


# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    print(f"\nStarted: {datetime.now()}")
    
    try:
        # Train model
        model, history, test_metrics, data, calibration, train_time = train_model(config)
        
        # Generate all results
        clustering, saliency_stats = generate_all_results(
            model, history, test_metrics, data, calibration, config
        )
        
        # External validation
        external = external_validation(model, config)
        
        # Print summary
        print_final_summary(
            test_metrics, calibration, clustering, saliency_stats,
            train_time, config, external
        )
        
        print("\n" + "=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        print(f"\nResults saved to: {config.results_dir}/")
        print(f"Checkpoints saved to: {config.checkpoint_dir}/")
        
        print("\nGenerated files:")
        for f in sorted(os.listdir(config.results_dir)):
            fpath = os.path.join(config.results_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / 1024
                print(f"  - {f} ({size:.1f} KB)")
        
        saliency_dir = os.path.join(config.results_dir, 'saliency_maps_100')
        if os.path.exists(saliency_dir):
            n_files = len([f for f in os.listdir(saliency_dir) if f.endswith('.png')])
            print(f"  - saliency_maps_100/ ({n_files} files)")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure your data is organized correctly.")
        print("Expected structure:")
        print("  data_root/")
        print("    cataract/")
        print("    diabetic_retinopathy/")
        print("    glaucoma/")
        print("    normal/")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\nFinished: {datetime.now()}")