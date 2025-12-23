#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis for MPS-RetNet
Tests different values for attention heads, transformer blocks, prototypes, and lambda_proto
"""

import os
import sys
import random
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Configuration
class Config:
    data_root = "retina_disease_classification"
    results_dir = "ablation_results"
    classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    num_classes = 4
    
    image_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    batch_size = 16
    num_epochs = 50
    learning_rate = 3e-5
    weight_decay = 0.02
    
    embed_dim = 512
    quality_channels = [32, 64, 128]
    dilation_rates = [3, 6, 9]
    focal_gamma = 2.0
    
    # Default hyperparameters (will be varied)
    num_heads = 8
    num_transformer_blocks = 2
    num_prototypes = 8
    prototype_loss_weight = 0.1

config = Config()
os.makedirs(config.results_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# DATA LOADING
# ============================================================================
class SimpleDataset(Dataset):
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
        return self.transform(img), self.labels[idx]

def load_images_from_directory(root_path: str, classes: List[str]):
    samples, labels = [], []
    root = Path(root_path)
    
    if not root.exists():
        print(f"Warning: Path does not exist: {root_path}")
        return samples, labels
    
    for class_idx, class_name in enumerate(classes):
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"Warning: Class folder not found for '{class_name}'")
            continue
        
        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = [p for p in class_dir.glob('*') if p.suffix.lower() in valid_ext]
        
        for img_path in sorted(image_files):
            samples.append(str(img_path))
            labels.append(class_idx)
        
        print(f"  {class_name}: {len(image_files)} images")
    
    return samples, labels

def prepare_data(config, seed=42):
    set_seed(seed)
    
    print(f"\nLoading dataset from: {config.data_root}")
    samples, labels = load_images_from_directory(config.data_root, config.classes)
    
    if not samples:
        raise ValueError(f"No images found in {config.data_root}")
    
    print(f"Total images: {len(samples)}")
    
    train_samples, test_samples, train_labels, test_labels = train_test_split(
        samples, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        train_samples, train_labels, test_size=0.2, stratify=train_labels, random_state=seed
    )
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(config.imagenet_mean, config.imagenet_std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.imagenet_mean, config.imagenet_std)
    ])
    
    train_dataset = SimpleDataset(train_samples, train_labels, train_transform)
    val_dataset = SimpleDataset(val_samples, val_labels, test_transform)
    test_dataset = SimpleDataset(test_samples, test_labels, test_transform)
    
    return {
        'train_loader': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4),
        'val_loader': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4),
        'test_loader': DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    }

# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class MultiScaleSpatialPooling(nn.Module):
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

class SpatialAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized)
        x = x + attended
        return x + self.ffn(self.norm2(x))

class PrototypeModule(nn.Module):
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
        p_norm = F.normalize(self.prototypes, dim=-1)
        similarity = torch.mm(p_norm, p_norm.t())
        identity = torch.eye(self.num_prototypes, device=self.prototypes.device)
        return torch.norm(similarity - identity, p='fro') ** 2

class QualityEstimationBranch(nn.Module):
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

# ============================================================================
# MODEL WITH CONFIGURABLE HYPERPARAMETERS
# ============================================================================
class MPSRetNet(nn.Module):
    def __init__(self, cfg, num_heads=None, num_blocks=None, num_prototypes=None, proto_weight=None):
        super().__init__()
        
        # Use custom values if provided, otherwise use config defaults
        self.num_heads = num_heads if num_heads is not None else cfg.num_heads
        self.num_blocks = num_blocks if num_blocks is not None else cfg.num_transformer_blocks
        self.num_prototypes = num_prototypes if num_prototypes is not None else cfg.num_prototypes
        self.proto_weight = proto_weight if proto_weight is not None else cfg.prototype_loss_weight
        
        # Backbone
        convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(convnext.features.children()))
        
        # Multi-scale pooling
        self.mssp = MultiScaleSpatialPooling(1024, cfg.embed_dim, cfg.dilation_rates)
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            SpatialAttentionBlock(cfg.embed_dim, self.num_heads)
            for _ in range(self.num_blocks)
        ])
        
        # Prototype module
        self.prototype_module = PrototypeModule(cfg.embed_dim, self.num_prototypes)
        
        # Quality branch
        self.quality_branch = QualityEstimationBranch(cfg.quality_channels)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Dropout(0.2),
            nn.Linear(cfg.embed_dim, cfg.num_classes)
        )
    
    def forward(self, x, return_quality=False):
        features = self.backbone(x)
        mssp_features = self.mssp(features)
        
        B, C, H, W = mssp_features.shape
        tokens = mssp_features.flatten(2).transpose(1, 2)
        
        for block in self.attention_blocks:
            tokens = block(tokens)
        
        embedding = tokens.mean(dim=1)
        enhanced, _ = self.prototype_module(embedding)
        logits = self.classifier(enhanced)
        
        output = {'logits': logits}
        
        if return_quality:
            output['quality_score'] = self.quality_branch(x)
        
        return output
    
    def get_diversity_loss(self):
        return self.prototype_module.diversity_loss()

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss

def train_epoch(model, loader, optimizer, cfg, proto_weight):
    model.train()
    total_loss = 0
    focal_loss_fn = FocalLoss(cfg.focal_gamma)
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(images, return_quality=True)
        
        focal = focal_loss_fn(output['logits'], labels)
        sup_loss = (output['quality_score'] * focal).mean()
        proto_loss = model.get_diversity_loss()
        
        loss = sup_loss + proto_weight * proto_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        output = model(images)
        preds = output['logits'].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    return {
        'accuracy': acc * 100,
        'f1': f1 * 100,
        'kappa': kappa * 100
    }

def train_and_evaluate(config, param_name, param_value, **model_kwargs):
    print(f"\n{'='*60}")
    print(f"Testing {param_name} = {param_value}")
    print(f"{'='*60}")
    
    # Use cached data if available
    if not hasattr(train_and_evaluate, 'data_cache'):
        train_and_evaluate.data_cache = prepare_data(config)
    
    data = train_and_evaluate.data_cache
    
    # Create model with custom hyperparameters
    model = MPSRetNet(config, **model_kwargs).to(device)
    
    # Get proto_weight from model_kwargs if present
    proto_weight = model_kwargs.get('proto_weight', config.prototype_loss_weight)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0
    patience = 0
    max_patience = 10
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, data['train_loader'], optimizer, config, proto_weight)
        val_metrics = evaluate(model, data['val_loader'])
        
        scheduler.step(val_metrics['accuracy'])
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Acc: {val_metrics['accuracy']:.2f}%")
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            patience = 0
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final test evaluation
    test_metrics = evaluate(model, data['test_loader'])
    
    print(f"\nTest Results: Acc={test_metrics['accuracy']:.2f}%, F1={test_metrics['f1']:.2f}%, Kappa={test_metrics['kappa']:.2f}%")
    
    return test_metrics

# ============================================================================
# MAIN HYPERPARAMETER SENSITIVITY STUDY
# ============================================================================
def main():
    print("="*70)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS FOR MPS-RETNET")
    print("="*70)
    
    results = []
    
    # 1. Attention Heads
    print("\n\n" + "="*70)
    print("TESTING: Number of Attention Heads")
    print("="*70)
    
    for heads in [4, 8, 16]:
        try:
            metrics = train_and_evaluate(config, "Attention Heads", heads, num_heads=heads)
            results.append({
                'Parameter': 'Attention Heads',
                'Value': heads,
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 (%)': f"{metrics['f1']:.2f}",
                'Kappa (%)': f"{metrics['kappa']:.2f}"
            })
        except Exception as e:
            print(f"Error with heads={heads}: {e}")
    
    # 2. Transformer Blocks
    print("\n\n" + "="*70)
    print("TESTING: Number of Transformer Blocks")
    print("="*70)
    
    for blocks in [1, 2, 4]:
        try:
            metrics = train_and_evaluate(config, "Transformer Blocks", blocks, num_blocks=blocks)
            results.append({
                'Parameter': 'Transformer Blocks',
                'Value': blocks,
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 (%)': f"{metrics['f1']:.2f}",
                'Kappa (%)': f"{metrics['kappa']:.2f}"
            })
        except Exception as e:
            print(f"Error with blocks={blocks}: {e}")
    
    # 3. Number of Prototypes
    print("\n\n" + "="*70)
    print("TESTING: Number of Prototypes")
    print("="*70)
    
    for protos in [4, 8, 16]:
        try:
            metrics = train_and_evaluate(config, "Prototypes", protos, num_prototypes=protos)
            results.append({
                'Parameter': 'Prototypes',
                'Value': protos,
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 (%)': f"{metrics['f1']:.2f}",
                'Kappa (%)': f"{metrics['kappa']:.2f}"
            })
        except Exception as e:
            print(f"Error with protos={protos}: {e}")
    
    # 4. Lambda Proto (Prototype Loss Weight)
    print("\n\n" + "="*70)
    print("TESTING: Lambda Proto (Prototype Loss Weight)")
    print("="*70)
    
    for lambda_val in [0.01, 0.1, 0.5]:
        try:
            metrics = train_and_evaluate(config, "Lambda Proto", lambda_val, proto_weight=lambda_val)
            results.append({
                'Parameter': 'λ_proto',
                'Value': lambda_val,
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 (%)': f"{metrics['f1']:.2f}",
                'Kappa (%)': f"{metrics['kappa']:.2f}"
            })
        except Exception as e:
            print(f"Error with lambda={lambda_val}: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(config.results_dir, 'table_hyperparameter_sensitivity.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_file}")
    print(f"\nResults Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()