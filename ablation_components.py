#!/usr/bin/env python3
"""
Component Ablation Study for MPS-RetNet
Tests each architectural component individually to measure contribution
"""

import os
import sys
import random
import time
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
    
    batch_size = 32
    num_epochs = 100
    learning_rate = 3e-5
    weight_decay = 0.02
    
    labeled_ratio = 0.26
    ema_decay = 0.999
    confidence_threshold = 0.95
    
    embed_dim = 512
    num_heads = 8
    num_transformer_blocks = 2
    num_prototypes = 8
    prototype_loss_weight = 0.1
    quality_channels = [32, 64, 128]
    dilation_rates = [3, 6, 9]
    focal_gamma = 2.0

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
    
    # Train/Val/Test split
    train_samples, test_samples, train_labels, test_labels = train_test_split(
        samples, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        train_samples, train_labels, test_size=0.2, stratify=train_labels, random_state=seed
    )
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Transforms
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
        'train_loader': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8),
        'val_loader': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8),
        'test_loader': DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
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

class SimpleGlobalPooling(nn.Module):
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
# FULL MODEL WITH ABLATION OPTIONS
# ============================================================================
class MPSRetNet(nn.Module):
    def __init__(self, cfg, ablation_config: Dict = None):
        super().__init__()
        
        # Default: all components enabled
        if ablation_config is None:
            ablation_config = {}
        
        self.use_transfer = ablation_config.get('transfer', True)
        self.use_attention = ablation_config.get('attention', True)
        self.use_mssp = ablation_config.get('mssp', True)
        self.use_proto = ablation_config.get('proto', True)
        self.use_quality = ablation_config.get('quality', True)
        
        # Backbone
        convnext = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if self.use_transfer else None
        )
        self.backbone = nn.Sequential(*list(convnext.features.children()))
        
        # Multi-scale pooling
        if self.use_mssp:
            self.mssp = MultiScaleSpatialPooling(1024, cfg.embed_dim, cfg.dilation_rates)
        else:
            self.mssp = SimpleGlobalPooling(1024, cfg.embed_dim)
        
        # Attention blocks
        if self.use_attention:
            self.attention_blocks = nn.ModuleList([
                SpatialAttentionBlock(cfg.embed_dim, cfg.num_heads)
                for _ in range(cfg.num_transformer_blocks)
            ])
        else:
            self.attention_blocks = None
        
        # Prototype module
        if self.use_proto:
            self.prototype_module = PrototypeModule(cfg.embed_dim, cfg.num_prototypes)
        else:
            self.prototype_module = None
        
        # Quality branch
        if self.use_quality:
            self.quality_branch = QualityEstimationBranch(cfg.quality_channels)
        else:
            self.quality_branch = None
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Dropout(0.2),
            nn.Linear(cfg.embed_dim, cfg.num_classes)
        )
    
    def forward(self, x, return_quality=False):
        # Backbone
        features = self.backbone(x)
        
        # Multi-scale pooling
        mssp_features = self.mssp(features)
        
        # Reshape for attention
        B, C, H, W = mssp_features.shape
        tokens = mssp_features.flatten(2).transpose(1, 2)
        
        # Attention
        if self.attention_blocks:
            for block in self.attention_blocks:
                tokens = block(tokens)
        
        # Global pooling
        embedding = tokens.mean(dim=1)
        
        # Prototype enhancement
        if self.prototype_module:
            enhanced, _ = self.prototype_module(embedding)
        else:
            enhanced = embedding
        
        # Classification
        logits = self.classifier(enhanced)
        
        output = {'logits': logits}
        
        if return_quality and self.quality_branch:
            output['quality_score'] = self.quality_branch(x)
        elif return_quality:
            output['quality_score'] = torch.ones(x.size(0), device=x.device)
        
        return output
    
    def get_diversity_loss(self):
        if self.prototype_module:
            return self.prototype_module.diversity_loss()
        return torch.tensor(0.0, device=device)

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

def train_epoch(model, loader, optimizer, cfg):
    model.train()
    total_loss = 0
    focal_loss_fn = FocalLoss(cfg.focal_gamma)
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(images, return_quality=True)
        
        # Supervised loss
        focal = focal_loss_fn(output['logits'], labels)
        sup_loss = (output['quality_score'] * focal).mean()
        
        # Prototype diversity loss
        proto_loss = model.get_diversity_loss()
        
        loss = sup_loss + cfg.prototype_loss_weight * proto_loss
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
        'precision': prec * 100,
        'recall': rec * 100,
        'f1': f1 * 100,
        'kappa': kappa * 100
    }

def train_and_evaluate(config, ablation_config, config_name):
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")
    
    # Prepare data
    data = prepare_data(config)
    
    # Create model
    model = MPSRetNet(config, ablation_config).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training
    best_acc = 0
    patience = 0
    max_patience = 10
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, data['train_loader'], optimizer, config)
        val_metrics = evaluate(model, data['val_loader'])
        
        scheduler.step(val_metrics['accuracy'])
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {train_loss:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%, Val F1: {val_metrics['f1']:.2f}%")
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            patience = 0
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    test_metrics = evaluate(model, data['test_loader'])
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  F1 Score: {test_metrics['f1']:.2f}%")
    print(f"  Kappa: {test_metrics['kappa']:.2f}%")
    
    return test_metrics

# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================
def main():
    print("="*70)
    print("COMPONENT ABLATION STUDY FOR MPS-RETNET")
    print("="*70)
    
    # Define ablation configurations
    ablation_configs = [
        ({'transfer': True, 'attention': True, 'mssp': True, 'proto': True, 'quality': True}, 
         'Full Model'),
        
        ({'transfer': False, 'attention': True, 'mssp': True, 'proto': True, 'quality': True}, 
         'Without Transfer Learning'),
        
        ({'transfer': True, 'attention': False, 'mssp': True, 'proto': True, 'quality': True}, 
         'Without Spatial Attention'),
        
        ({'transfer': True, 'attention': True, 'mssp': False, 'proto': True, 'quality': True}, 
         'Without Multi-Scale Pooling'),
        
        ({'transfer': True, 'attention': True, 'mssp': True, 'proto': False, 'quality': True}, 
         'Without Prototype Module'),
        
        ({'transfer': True, 'attention': True, 'mssp': True, 'proto': True, 'quality': False}, 
         'Without Quality Branch'),
    ]
    
    # Run ablation studies
    results = []
    
    for ablation_config, config_name in ablation_configs:
        try:
            metrics = train_and_evaluate(config, ablation_config, config_name)
            results.append({
                'Configuration': config_name,
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 (%)': f"{metrics['f1']:.2f}",
                'Kappa (%)': f"{metrics['kappa']:.2f}"
            })
        except Exception as e:
            print(f"Error with {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(config.results_dir, 'table_ablation_components.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_file}")
    print(f"\nResults Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()