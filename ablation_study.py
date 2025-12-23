#!/usr/bin/env python3
"""
Component Ablation Study for MPS-RetNet
Tests the impact of different architectural components
"""

import os
import sys
import random
import time
import copy
import warnings
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR

import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, auc, cohen_kappa_score, classification_report,
    silhouette_score, davies_bouldin_score
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold

from scipy import stats as scipy_stats
from tqdm.auto import tqdm

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 
    'font.size': 12, 
    'figure.dpi': 150, 
    'savefig.dpi': 300, 
    'figure.facecolor': 'white'
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'='*70}")
print(f"Component Ablation Study - MPS-RetNet")
print(f"{'='*70}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"PyTorch Version: {torch.__version__}")
print(f"{'='*70}")


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration class with all hyperparameters"""
    # Data paths
    data_root = "retina_disease_classification"
    results_dir = "results"
    checkpoint_dir = "checkpoints"
    local_save_path = "outputs"
    
    # Classes
    classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    num_classes = 4
    
    # Image preprocessing
    image_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Training hyperparameters
    batch_size = 16
    num_epochs = 100
    learning_rate = 3e-5
    weight_decay = 0.02
    grad_clip_norm = 1.0
    early_stopping_patience = 25
    min_epochs = 40
    
    # Semi-supervised learning parameters
    labeled_ratio = 0.26
    ema_decay = 0.999
    consistency_weight = 1.0
    confidence_threshold = 0.95
    rampup_epochs = 15
    
    # FixMatch parameters
    fixmatch_threshold = 0.95
    use_hard_labels = True
    
    # Contrastive learning parameters
    contrastive_weight = 0.1
    contrastive_temperature = 0.07
    contrastive_dim = 128
    
    # MixUp parameters
    mixup_alpha = 0.2
    use_mixup = True
    
    # SAM optimizer parameters
    use_sam = True
    sam_rho = 0.05
    
    # Distribution alignment
    use_distribution_alignment = True
    da_temperature = 0.5
    
    # Model architecture
    embed_dim = 512
    num_heads = 8
    num_transformer_blocks = 2
    num_prototypes = 8
    prototype_loss_weight = 0.1
    quality_channels = [32, 64, 128]
    dilation_rates = [3, 6, 9]
    focal_gamma = 2.0
    
    # Component flags for ablation
    use_transfer_learning = True
    use_spatial_attention = True
    use_multiscale_pooling = True
    use_prototype_module = True
    use_quality_branch = True
    use_ema = True
    use_confidence_filtering = True
    use_contrastive = True


config = Config()

# Create directories
os.makedirs(config.results_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.local_save_path, exist_ok=True)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ============================================================================
# SHARPNESS-AWARE MINIMIZATION (SAM) OPTIMIZER
# ============================================================================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer for better generalization."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho value: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute gradient at current point and perturb weights"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original weights and perform actual update"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
                    del self.state[p]["old_p"]
        
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ============================================================================
# DATA AUGMENTATION CLASSES
# ============================================================================
class WeakAugmentation:
    """Weak augmentation for teacher model"""
    def __init__(self, size: int, mean: List[float], std: List[float]):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, x):
        return self.transform(x)


class StrongAugmentation:
    """Strong augmentation for student model (FixMatch-style)"""
    def __init__(self, size: int, mean: List[float], std: List[float]):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))
        ])
    
    def __call__(self, x):
        return self.transform(x)


class ContrastiveAugmentation:
    """Augmentation for contrastive learning (SimCLR-style)"""
    def __init__(self, size: int, mean: List[float], std: List[float]):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class TestTransform:
    """Transform for validation/test (no augmentation)"""
    def __init__(self, size: int, mean: List[float], std: List[float]):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, x):
        return self.transform(x)


# ============================================================================
# DATASET CLASSES
# ============================================================================
class SemiSupervisedDataset(Dataset):
    """Dataset for semi-supervised learning with multiple augmentation views"""
    def __init__(self, samples: List[str], labels: List[int], 
                 weak_transform, strong_transform=None, 
                 contrastive_transform=None, is_labeled: bool = True):
        self.samples = samples
        self.labels = labels
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform or weak_transform
        self.contrastive_transform = contrastive_transform
        self.is_labeled = is_labeled
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        weak_img = self.weak_transform(img)
        
        if self.is_labeled:
            return weak_img, self.labels[idx], idx
        else:
            strong_img = self.strong_transform(img)
            if self.contrastive_transform:
                cont_img1, cont_img2 = self.contrastive_transform(img)
                return weak_img, strong_img, cont_img1, cont_img2, idx
            return weak_img, strong_img, idx


class SimpleDataset(Dataset):
    """Simple dataset for evaluation"""
    def __init__(self, samples: List[str], labels: List[int], transform):
        self.samples = samples
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        return self.transform(img), self.labels[idx], img_path


# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class MultiScaleSpatialPooling(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale feature extraction"""
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: List[int]):
        super().__init__()
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for r in dilation_rates
        ])
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        total_channels = out_channels * (2 + len(dilation_rates))
        self.projection = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        features = [self.conv1x1(x)]
        features.extend([conv(x) for conv in self.dilated_convs])
        features.append(F.interpolate(self.gap(x), size, mode='bilinear', align_corners=False))
        return self.projection(torch.cat(features, dim=1))


class SimpleGlobalPooling(nn.Module):
    """Simple global pooling for ablation study"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
    
    def forward(self, x):
        return self.conv(x)


class SpatialAttentionBlock(nn.Module):
    """Transformer-style spatial attention block"""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
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
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized)
        x = x + self.dropout(attended)
        return x + self.ffn(self.norm2(x))


class PrototypeModule(nn.Module):
    """Learnable prototype module for class representation"""
    def __init__(self, embed_dim: int, num_prototypes: int, temperature: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.num_prototypes = num_prototypes
        nn.init.xavier_uniform_(self.prototypes)
    
    def forward(self, z):
        z_norm = F.normalize(z, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        similarity = torch.mm(z_norm, p_norm.t()) / self.temperature.abs().clamp(min=0.01)
        attention = F.softmax(similarity, dim=-1)
        enhanced = z + torch.mm(attention, self.prototypes)
        return enhanced, attention
    
    def diversity_loss(self):
        """Encourage prototype diversity"""
        p_norm = F.normalize(self.prototypes, dim=-1)
        similarity = torch.mm(p_norm, p_norm.t())
        identity = torch.eye(self.num_prototypes, device=self.prototypes.device)
        return torch.norm(similarity - identity, p='fro') ** 2


class QualityEstimationBranch(nn.Module):
    """Branch for estimating image quality scores"""
    def __init__(self, channels: List[int] = [32, 64, 128]):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], 1)
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return torch.sigmoid(self.fc(pooled)).squeeze(-1)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class MPSRetNet(nn.Module):
    """
    Multi-Scale Prototype-Guided Semi-Supervised Network for Retinal Disease Classification
    """
    def __init__(self, cfg, ablation_config: Optional[Dict] = None):
        super().__init__()
        
        # Determine which components to use
        self.use_transfer = cfg.use_transfer_learning if ablation_config is None else ablation_config.get('transfer', True)
        self.use_attention = cfg.use_spatial_attention if ablation_config is None else ablation_config.get('attention', True)
        self.use_mssp = cfg.use_multiscale_pooling if ablation_config is None else ablation_config.get('mssp', True)
        self.use_proto = cfg.use_prototype_module if ablation_config is None else ablation_config.get('proto', True)
        self.use_quality = cfg.use_quality_branch if ablation_config is None else ablation_config.get('quality', True)
        self.use_contrastive = cfg.use_contrastive if ablation_config is None else ablation_config.get('contrastive', True)
        
        # Backbone
        convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if self.use_transfer else None)
        self.backbone = nn.Sequential(*list(convnext.features.children()))
        
        # Multi-scale spatial pooling
        self.mssp = (MultiScaleSpatialPooling(1024, cfg.embed_dim, cfg.dilation_rates) 
                     if self.use_mssp else SimpleGlobalPooling(1024, cfg.embed_dim))
        
        # Spatial attention blocks
        self.attention_blocks = (nn.ModuleList([
            SpatialAttentionBlock(cfg.embed_dim, cfg.num_heads) 
            for _ in range(cfg.num_transformer_blocks)
        ]) if self.use_attention else None)
        
        # Prototype module
        self.prototype_module = (PrototypeModule(cfg.embed_dim, cfg.num_prototypes) 
                                 if self.use_proto else None)
        
        # Quality estimation branch
        self.quality_branch = (QualityEstimationBranch(cfg.quality_channels) 
                               if self.use_quality else None)
        
        # Contrastive projection head
        self.projection_head = (ProjectionHead(cfg.embed_dim, cfg.embed_dim, cfg.contrastive_dim) 
                                if self.use_contrastive else None)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Dropout(0.2),
            nn.Linear(cfg.embed_dim, cfg.num_classes)
        )
    
    def forward(self, x, return_features: bool = False, return_quality: bool = False, 
                return_projection: bool = False):
        # Backbone features
        backbone_features = self.backbone(x)
        
        # Multi-scale pooling
        mssp_features = self.mssp(backbone_features)
        
        # Reshape for attention
        B, C, H, W = mssp_features.shape
        tokens = mssp_features.flatten(2).transpose(1, 2)
        
        # Apply attention blocks
        if self.attention_blocks:
            for block in self.attention_blocks:
                tokens = block(tokens)
        
        # Global average pooling
        embedding = tokens.mean(dim=1)
        
        # Prototype enhancement
        if self.prototype_module:
            enhanced, proto_attention = self.prototype_module(embedding)
        else:
            enhanced = embedding
        
        # Classification
        logits = self.classifier(enhanced)
        
        # Build output
        output = {'logits': logits}
        
        if return_features:
            output['embedding'] = embedding
        
        if return_quality and self.quality_branch:
            output['quality_score'] = self.quality_branch(x)
        elif return_quality:
            output['quality_score'] = torch.ones(x.size(0), device=x.device)
        
        if return_projection and self.projection_head:
            output['projection'] = F.normalize(self.projection_head(embedding), dim=-1)
        
        return output
    
    def get_diversity_loss(self):
        """Get prototype diversity loss"""
        if self.prototype_module:
            return self.prototype_module.diversity_loss()
        return torch.tensor(0.0, device=device)
    
    def get_prototypes(self):
        """Get learned prototypes for visualization"""
        if self.prototype_module:
            return self.prototype_module.prototypes.detach().cpu().numpy()
        return None


# ============================================================================
# EMA MODEL
# ============================================================================
class EMAModel:
    """Exponential Moving Average model for teacher network"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1 - self.decay) * param.data + self.decay * self.shadow[name]
    
    def apply_shadow(self):
        """Apply shadow weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent loss for contrastive learning"""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    
    sim = torch.mm(z, z.t()) / temperature
    sim_ij = torch.diag(sim, batch_size)
    sim_ji = torch.diag(sim, -batch_size)
    positive = torch.cat([sim_ij, sim_ji], dim=0)
    
    mask = (~torch.eye(2 * batch_size, dtype=bool, device=z.device)).float()
    nominator = torch.exp(positive)
    denominator = mask * torch.exp(sim)
    
    loss = -torch.log(nominator / (denominator.sum(dim=1) + 1e-8))
    return loss.mean()


def consistency_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor, 
                     mask: Optional[torch.Tensor] = None, use_hard_labels: bool = True) -> torch.Tensor:
    """Consistency loss between student and teacher predictions"""
    if use_hard_labels:
        pseudo_labels = teacher_pred.argmax(dim=-1)
        loss = F.cross_entropy(student_pred, pseudo_labels, reduction='none')
    else:
        loss = F.mse_loss(student_pred, teacher_pred, reduction='none').sum(dim=-1)
    
    if mask is not None:
        return (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss.mean()


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, 
                    y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """MixUp loss computation"""
    return lam * criterion(pred, y_a).mean() + (1 - lam) * criterion(pred, y_b).mean()


def cosine_rampup(epoch: int, rampup_epochs: int) -> float:
    """Cosine ramp-up for consistency weight scheduling"""
    if epoch >= rampup_epochs:
        return 1.0
    return (1 - np.cos(np.pi * epoch / rampup_epochs)) / 2


def get_class_distribution(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute class distribution from labels"""
    counts = torch.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    return counts / counts.sum()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_images_from_directory(root_path: str, classes: List[str]) -> Tuple[List[str], List[int]]:
    """Load image paths and labels from directory structure"""
    samples, labels = [], []
    root = Path(root_path)
    
    if not root.exists():
        print(f"  Warning: Path does not exist: {root_path}")
        return samples, labels
    
    subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    
    for class_idx, class_name in enumerate(classes):
        class_dir = None
        for sn, sp in subdirs.items():
            if (class_name.lower() == sn or 
                class_name.lower().replace('_', '') in sn.replace('_', '') or
                sn in class_name.lower()):
                class_dir = sp
                break
        
        if not class_dir:
            print(f"  Warning: Class folder not found for '{class_name}'")
            continue
        
        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        image_files = [p for p in class_dir.glob('*') if p.suffix.lower() in valid_ext]
        
        for img_path in sorted(image_files):
            samples.append(str(img_path))
            labels.append(class_idx)
        
        print(f"    {class_name}: {len(image_files)} images")
    
    return samples, labels


def prepare_data_for_ablation(cfg, ablation_config: Dict) -> Dict:
    """Prepare data loaders for ablation study"""
    set_seed(42)
    
    print(f"\nLoading dataset from: {cfg.data_root}")
    samples, labels = load_images_from_directory(cfg.data_root, cfg.classes)
    
    if not samples:
        raise ValueError(f"No images found in {cfg.data_root}")
    
    print(f"Total images: {len(samples)}")
    
    # Split data
    labeled_samples, unlabeled_samples, labeled_labels, _ = train_test_split(
        samples, labels, 
        test_size=1 - cfg.labeled_ratio,
        stratify=labels, 
        random_state=42
    )
    
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        labeled_samples, labeled_labels,
        test_size=0.2,
        stratify=labeled_labels,
        random_state=42
    )
    
    print(f"\nData Splits:")
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Unlabeled: {len(unlabeled_samples)}")
    
    weak_t = WeakAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    strong_t = StrongAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    contrastive_t = ContrastiveAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std) if cfg.use_contrastive else None
    test_t = TestTransform(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    
    train_dataset = SemiSupervisedDataset(train_samples, train_labels, weak_t, is_labeled=True)
    unlabeled_dataset = SemiSupervisedDataset(unlabeled_samples, [0] * len(unlabeled_samples), weak_t, strong_t, contrastive_t, is_labeled=False)
    val_dataset = SimpleDataset(val_samples, val_labels, test_t)
    
    return {
        'train_loader': DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
        'unlabeled_loader': DataLoader(unlabeled_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
        'val_loader': DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4),
        'train_labels': train_labels,
        'class_distribution': get_class_distribution(train_labels, cfg.num_classes)
    }


def prepare_data(cfg, seed: int = 42) -> Dict:
    """Prepare data loaders for standard training"""
    set_seed(seed)
    
    print(f"\nLoading dataset from: {cfg.data_root}")
    samples, labels = load_images_from_directory(cfg.data_root, cfg.classes)
    
    if not samples:
        raise ValueError(f"No images found in {cfg.data_root}")
    
    print(f"Total images: {len(samples)}")
    
    tv_samples, test_samples, tv_labels, test_labels = train_test_split(
        samples, labels, test_size=0.1, stratify=labels, random_state=seed
    )
    
    labeled_samples, unlabeled_samples, labeled_labels, _ = train_test_split(
        tv_samples, tv_labels, 
        test_size=1 - cfg.labeled_ratio / 0.9,
        stratify=tv_labels, 
        random_state=seed
    )
    
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        labeled_samples, labeled_labels,
        test_size=0.2,
        stratify=labeled_labels,
        random_state=seed
    )
    
    print(f"\nData Splits:")
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)} | Unlabeled: {len(unlabeled_samples)}")
    
    weak_t = WeakAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    strong_t = StrongAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    contrastive_t = ContrastiveAugmentation(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std) if cfg.use_contrastive else None
    test_t = TestTransform(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    
    train_dataset = SemiSupervisedDataset(train_samples, train_labels, weak_t, is_labeled=True)
    unlabeled_dataset = SemiSupervisedDataset(unlabeled_samples, [0] * len(unlabeled_samples), weak_t, strong_t, contrastive_t, is_labeled=False)
    val_dataset = SimpleDataset(val_samples, val_labels, test_t)
    test_dataset = SimpleDataset(test_samples, test_labels, test_t)
    
    return {
        'train_loader': DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
        'unlabeled_loader': DataLoader(unlabeled_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
        'val_loader': DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4),
        'test_loader': DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4),
        'test_samples': test_samples, 'test_labels': test_labels,
        'val_samples': val_samples, 'val_labels': val_labels,
        'all_samples': samples, 'all_labels': labels,
        'train_labels': train_labels,
        'class_distribution': get_class_distribution(train_labels, cfg.num_classes)
    }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model: nn.Module, ema, train_loader: DataLoader, unlabeled_loader: DataLoader,
                optimizer, cfg, epoch: int, device: torch.device,
                use_ema: bool = True, use_conf_filter: bool = True,
                use_contrastive: bool = True, use_sam: bool = True) -> Dict:
    """Enhanced semi-supervised training epoch"""
    model.train()
    
    consistency_w = cfg.consistency_weight * cosine_rampup(epoch, cfg.rampup_epochs)
    contrastive_w = cfg.contrastive_weight * cosine_rampup(epoch, cfg.rampup_epochs)
    
    focal_loss_fn = FocalLoss(cfg.focal_gamma)
    
    total_loss, sup_loss_total, cons_loss_total = 0, 0, 0
    proto_loss_total, contrastive_loss_total = 0, 0
    hc_count, hc_total = 0, 0
    
    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)
    num_batches = min(len(train_loader), len(unlabeled_loader))
    
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{cfg.num_epochs}", leave=False)
    
    for batch_idx in pbar:
        # Get labeled batch
        try:
            labeled_imgs, labels, _ = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(train_loader)
            labeled_imgs, labels, _ = next(labeled_iter)
        
        # Get unlabeled batch
        try:
            if use_contrastive and cfg.use_contrastive:
                weak_imgs, strong_imgs, cont_img1, cont_img2, _ = next(unlabeled_iter)
                cont_img1, cont_img2 = cont_img1.to(device), cont_img2.to(device)
            else:
                weak_imgs, strong_imgs, _ = next(unlabeled_iter)
                cont_img1, cont_img2 = None, None
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            if use_contrastive and cfg.use_contrastive:
                weak_imgs, strong_imgs, cont_img1, cont_img2, _ = next(unlabeled_iter)
                cont_img1, cont_img2 = cont_img1.to(device), cont_img2.to(device)
            else:
                weak_imgs, strong_imgs, _ = next(unlabeled_iter)
                cont_img1, cont_img2 = None, None
        
        labeled_imgs, labels = labeled_imgs.to(device), labels.to(device)
        weak_imgs, strong_imgs = weak_imgs.to(device), strong_imgs.to(device)
        
        def forward_and_compute_loss():
            # Supervised loss
            if cfg.use_mixup and np.random.random() > 0.5:
                mixed_imgs, y_a, y_b, lam = mixup_data(labeled_imgs, labels, cfg.mixup_alpha)
                output = model(mixed_imgs, return_quality=True)
                sup_loss = mixup_criterion(focal_loss_fn, output['logits'], y_a, y_b, lam)
            else:
                output = model(labeled_imgs, return_quality=True)
                focal = focal_loss_fn(output['logits'], labels)
                sup_loss = (output['quality_score'] * focal).mean()
            
            # Consistency loss
            if use_ema and ema:
                ema.apply_shadow()
                with torch.no_grad():
                    teacher_output = model(weak_imgs)
                    teacher_probs = F.softmax(teacher_output['logits'] / cfg.da_temperature, dim=-1)
                ema.restore()
            else:
                with torch.no_grad():
                    teacher_output = model(weak_imgs)
                    teacher_probs = F.softmax(teacher_output['logits'] / cfg.da_temperature, dim=-1)
            
            student_output = model(strong_imgs, return_projection=use_contrastive)
            student_logits = student_output['logits']
            
            max_probs, pseudo_labels = teacher_probs.max(dim=-1)
            
            if use_conf_filter:
                mask = (max_probs >= cfg.fixmatch_threshold).float()
            else:
                mask = torch.ones_like(max_probs)
            
            if cfg.use_hard_labels:
                cons_loss = (F.cross_entropy(student_logits, pseudo_labels, reduction='none') * mask).sum() / (mask.sum() + 1e-8)
            else:
                student_probs = F.softmax(student_logits, dim=-1)
                cons_loss = consistency_loss(student_probs, teacher_probs, mask, use_hard_labels=False)
            
            # Contrastive loss
            contrastive_loss = torch.tensor(0.0, device=device)
            if use_contrastive and cfg.use_contrastive and cont_img1 is not None:
                z1 = model(cont_img1, return_projection=True)['projection']
                z2 = model(cont_img2, return_projection=True)['projection']
                contrastive_loss = nt_xent_loss(z1, z2, cfg.contrastive_temperature)
            
            # Prototype diversity loss
            proto_loss = model.get_diversity_loss()
            
            # Total loss
            loss = (sup_loss + consistency_w * cons_loss + 
                    cfg.prototype_loss_weight * proto_loss + contrastive_w * contrastive_loss)
            
            return loss, sup_loss, cons_loss, proto_loss, contrastive_loss, mask
        
        # Optimization with SAM
        if use_sam and isinstance(optimizer, SAM):
            loss, sup_loss, cons_loss, proto_loss, contrastive_loss, mask = forward_and_compute_loss()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            loss2, _, _, _, _, _ = forward_and_compute_loss()
            loss2.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss, sup_loss, cons_loss, proto_loss, contrastive_loss, mask = forward_and_compute_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        
        if use_ema and ema:
            ema.update()
        
        total_loss += loss.item()
        sup_loss_total += sup_loss.item()
        cons_loss_total += cons_loss.item()
        proto_loss_total += proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss
        contrastive_loss_total += contrastive_loss.item()
        hc_count += mask.sum().item()
        hc_total += len(mask)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'hc': f'{mask.sum().item()}/{len(mask)}'})
    
    return {
        'total': total_loss / num_batches, 'sup': sup_loss_total / num_batches,
        'cons': cons_loss_total / num_batches, 'proto': proto_loss_total / num_batches,
        'contrastive': contrastive_loss_total / num_batches,
        'hc_ratio': hc_count / hc_total if hc_total > 0 else 0
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, return_embeddings: bool = False) -> Dict:
    """Validate model on given data loader"""
    model.eval()
    
    all_preds, all_labels, all_probs, all_embeddings = [], [], [], []
    total_loss = 0
    
    for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        output = model(images, return_features=return_embeddings)
        
        probs = F.softmax(output['logits'], dim=-1)
        total_loss += F.cross_entropy(output['logits'], labels.to(device)).item()
        
        all_preds.extend(output['logits'].argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        
        if return_embeddings:
            all_embeddings.extend(output['embedding'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    spec = []
    for i in range(len(cm)):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    try:
        auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc_score = 0
    
    results = {
        'loss': total_loss / len(loader), 'accuracy': accuracy_score(all_labels, all_preds),
        'precision': prec, 'recall': rec, 'f1': f1,
        'kappa': cohen_kappa_score(all_labels, all_preds), 'auc': auc_score,
        'predictions': all_preds, 'labels': all_labels, 'probabilities': all_probs,
        'confusion_matrix': cm, 'precision_per_class': prec_c, 'recall_per_class': rec_c,
        'f1_per_class': f1_c, 'specificity_per_class': np.array(spec)
    }
    
    if return_embeddings:
        results['embeddings'] = np.array(all_embeddings)
    
    return results


def train_model(cfg, seed: int = 42, ablation_config=None, fold_data=None):
    """Main training function with all enhancements"""
    print(f"\n{'='*70}\nMPS-RetNet Training (Seed: {seed})\n{'='*70}")
    
    set_seed(seed)
    data = fold_data if fold_data else prepare_data_for_ablation(cfg, ablation_config)
    model = MPSRetNet(cfg, ablation_config).to(device)
    
    use_ema = cfg.use_ema if ablation_config is None else ablation_config.get('ema', True)
    use_sam = cfg.use_sam if ablation_config is None else ablation_config.get('sam', True)
    use_contrastive = cfg.use_contrastive if ablation_config is None else ablation_config.get('contrastive', True)
    use_conf_filter = cfg.use_confidence_filtering if ablation_config is None else ablation_config.get('conf_filter', True)
    
    ema = EMAModel(model, cfg.ema_decay) if use_ema else None
    
    if use_sam:
        optimizer = SAM(model.parameters(), AdamW, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, rho=cfg.sam_rho)
        scheduler = ReduceLROnPlateau(optimizer.base_optimizer, mode='max', factor=0.5, patience=7)
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Using: EMA={use_ema}, SAM={use_sam}, Contrastive={use_contrastive}")
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_kappa': [], 
               'val_auc': [], 'sup_loss': [], 'cons_loss': [], 'proto_loss': [], 
               'contrastive_loss': [], 'hc_ratio': [], 'lr': []}
    
    best_acc, best_f1, patience_counter, start_time = 0, 0, 0, time.time()
    
    for epoch in range(cfg.num_epochs):
        train_metrics = train_epoch(model, ema, data['train_loader'], data['unlabeled_loader'],
                                   optimizer, cfg, epoch, device, use_ema, use_conf_filter, use_contrastive, use_sam)
        
        if use_ema and ema:
            ema.apply_shadow()
            val_metrics = validate(model, data['val_loader'], device)
            ema.restore()
        else:
            val_metrics = validate(model, data['val_loader'], device)
        
        current_lr = (optimizer.base_optimizer if use_sam else optimizer).param_groups[0]['lr']
        scheduler.step(val_metrics['accuracy'])
        
        for k, v in [('train_loss', train_metrics['total']), ('val_loss', val_metrics['loss']),
                     ('val_acc', val_metrics['accuracy']), ('val_f1', val_metrics['f1']),
                     ('val_kappa', val_metrics['kappa']), ('val_auc', val_metrics['auc']),
                     ('sup_loss', train_metrics['sup']), ('cons_loss', train_metrics['cons']),
                     ('proto_loss', train_metrics['proto']), ('contrastive_loss', train_metrics['contrastive']),
                     ('hc_ratio', train_metrics['hc_ratio']), ('lr', current_lr)]:
            history[k].append(v)
        
        print(f"\nEpoch {epoch+1}: Acc={val_metrics['accuracy']*100:.2f}%, F1={val_metrics['f1']*100:.2f}%")
        
        if val_metrics['f1'] > best_f1:
            best_acc, best_f1, patience_counter = val_metrics['accuracy'], val_metrics['f1'], 0
            if ablation_config is None:
                save_checkpoint(model, ema, optimizer, scheduler, epoch, history, cfg, 'best_model.pt', True)
        else:
            patience_counter += 1
        
        if epoch >= cfg.min_epochs and patience_counter >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - start_time
    
    if ablation_config is None and os.path.exists(os.path.join(cfg.checkpoint_dir, 'best_model.pt')):
        load_checkpoint(model, os.path.join(cfg.checkpoint_dir, 'best_model.pt'), device)
    
    if use_ema and ema:
        ema.apply_shadow()
        test_metrics = validate(model, data['val_loader'], device, True)
        ema.restore()
    else:
        test_metrics = validate(model, data['val_loader'], device, True)
    
    return model, history, test_metrics, data, train_time


def save_checkpoint(model: nn.Module, ema, optimizer, scheduler, epoch: int, 
                    history: Dict, cfg, filename: str, is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow': copy.deepcopy(ema.shadow) if ema else None,
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history
    }
    
    ckpt_path = os.path.join(cfg.checkpoint_dir, filename)
    torch.save(checkpoint, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")
    
    local_path = os.path.join(cfg.local_save_path, filename)
    if os.path.normpath(os.path.abspath(ckpt_path)) != os.path.normpath(os.path.abspath(local_path)):
        torch.save(checkpoint, local_path)
        print(f"  Saved checkpoint: {local_path}")
    
    if is_best:
        best_ckpt_path = os.path.join(cfg.checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_ckpt_path)
        
        best_local_path = os.path.join(cfg.local_save_path, 'best_model.pt')
        if os.path.normpath(os.path.abspath(best_ckpt_path)) != os.path.normpath(os.path.abspath(best_local_path)):
            torch.save(checkpoint, best_local_path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> Dict:
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================
def run_component_ablation():
    """Run component ablation study"""
    print(f"\n{'='*70}\nComponent Ablation Study\n{'='*70}")
    
    # Define configurations to test
    configurations = [
        {"name": "Full Model", "ablation_config": {}},
        {"name": "Without Transfer Learning", "ablation_config": {"transfer": False}},
        {"name": "Without Spatial Attention", "ablation_config": {"attention": False}},
        {"name": "Without Multi-Scale Pooling", "ablation_config": {"mssp": False}},
        {"name": "Without Prototype Module", "ablation_config": {"proto": False}},
        {"name": "Without Quality Branch", "ablation_config": {"quality": False}},
        {"name": "Without EMA (Direct Student)", "ablation_config": {"ema": False}},
        {"name": "Without Confidence Filtering", "ablation_config": {"conf_filter": False}},
        {"name": "Quality Branch: Random Weights", "ablation_config": {"quality": False, "quality_random": True}},
        {"name": "Quality Branch: Uniform Weights", "ablation_config": {"quality": False, "quality_uniform": True}},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n--- Testing: {config['name']} ---")
        try:
            _, _, test_metrics, _, _ = train_model(config=Config(), ablation_config=config['ablation_config'])
            
            results.append({
                "Configuration": config['name'],
                "Accuracy (%)": test_metrics['accuracy'] * 100,
                "F1 (%)": test_metrics['f1'] * 100,
                "kappa (%)": test_metrics['kappa'] * 100
            })
            
            print(f"  Results: Accuracy={test_metrics['accuracy']*100:.2f}%, F1={test_metrics['f1']*100:.2f}%, Kappa={test_metrics['kappa']*100:.2f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "Configuration": config['name'],
                "Accuracy (%)": 0.0,
                "F1 (%)": 0.0,
                "kappa (%)": 0.0
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{Config().results_dir}/component_ablation_results.csv", index=False)
    print(f"\nResults saved to: {Config().results_dir}/component_ablation_results.csv")
    
    # Print results
    print("\nComponent Ablation Results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_component_ablation()