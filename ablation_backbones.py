#!/usr/bin/env python3
"""
Backbone Architecture Comparison for MPS-RetNet - FIXED VERSION
Compares ALL 7 backbone architectures with proper error handling

FIXED ISSUES:
- ViT forward pass corrected
- Proper channel dimensions for all backbones
- Better error handling
- GPU optimizations
"""

import os
import sys
import random
import warnings
import time
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
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    vit_b_16, ViT_B_16_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_base, ConvNeXt_Base_Weights
)
from PIL import Image
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============================================================================
# GPU SETUP
# ============================================================================
print("="*80)
print("BACKBONE COMPARISON FOR MPS-RETNET")
print("="*80)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print("⚠ Running on CPU")

print("="*80)

# Configuration
class Config:
    data_root = "retina_disease_classification"
    results_dir = "ablation_results"
    classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    num_classes = 4
    
    image_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    batch_size = 32  # Increased for GPU
    num_workers = 8
    pin_memory = True
    
    num_epochs = 40
    learning_rate = 3e-5
    weight_decay = 0.02
    
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
        transforms.RandomRotation(10),
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
        'train_loader': DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, 
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            persistent_workers=True
        ),
        'val_loader': DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, 
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            persistent_workers=True
        ),
        'test_loader': DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False, 
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            persistent_workers=True
        )
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
# BACKBONE-SPECIFIC MODEL - FULLY FIXED
# ============================================================================
class MPSRetNet_Backbone(nn.Module):
    def __init__(self, cfg, backbone_name='convnext_base'):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.is_vit = (backbone_name == 'vit_b_16')
        
        print(f"  Loading {backbone_name}...", end=" ")
        
        # Load backbone with proper configuration
        if backbone_name == 'resnet50':
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            backbone_out_channels = 2048
            
        elif backbone_name == 'resnet101':
            base_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            backbone_out_channels = 2048
            
        elif backbone_name == 'densenet121':
            base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.backbone = base_model.features
            self.backbone.add_module('relu', nn.ReLU(inplace=True))
            backbone_out_channels = 1024
            
        elif backbone_name == 'efficientnet_b3':
            base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.backbone = base_model.features
            backbone_out_channels = 1536
            
        elif backbone_name == 'vit_b_16':
            # ViT requires special handling
            self.vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            backbone_out_channels = 768
            
        elif backbone_name == 'convnext_small':
            base_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(base_model.features.children()))
            backbone_out_channels = 768
            
        elif backbone_name == 'convnext_base':
            base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(base_model.features.children()))
            backbone_out_channels = 1024
            
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        print(f"✓ ({backbone_out_channels} channels)")
        
        # Architecture-specific processing
        if self.is_vit:
            # ViT: Simple projection from CLS token
            self.adapter = nn.Sequential(
                nn.Linear(backbone_out_channels, cfg.embed_dim),
                nn.LayerNorm(cfg.embed_dim),
                nn.Dropout(0.1)
            )
            self.attention_blocks = None
        else:
            # CNN: Multi-scale pooling + attention
            self.mssp = MultiScaleSpatialPooling(backbone_out_channels, cfg.embed_dim, cfg.dilation_rates)
            self.attention_blocks = nn.ModuleList([
                SpatialAttentionBlock(cfg.embed_dim, cfg.num_heads)
                for _ in range(cfg.num_transformer_blocks)
            ])
        
        # Shared modules
        self.prototype_module = PrototypeModule(cfg.embed_dim, cfg.num_prototypes)
        self.quality_branch = QualityEstimationBranch(cfg.quality_channels)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Dropout(0.3),
            nn.Linear(cfg.embed_dim, cfg.num_classes)
        )
    
    def forward(self, x_input, return_quality=False):
        if self.is_vit:
            # ViT forward: Extract CLS token properly
            # Reshape input
            x = self.vit_model._process_input(x_input)
            n = x.shape[0]
            
            # Add CLS token
            batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            
            # Pass through encoder
            x = self.vit_model.encoder(x)
            
            # Extract CLS token and adapt
            cls_token = x[:, 0]
            embedding = self.adapter(cls_token)
            
            # Prototype enhancement
            enhanced, _ = self.prototype_module(embedding)
            
        else:
            # CNN forward
            features = self.backbone(x_input)
            mssp_features = self.mssp(features)
            
            B, C, H, W = mssp_features.shape
            tokens = mssp_features.flatten(2).transpose(1, 2)
            
            # Attention blocks
            if self.attention_blocks:
                for block in self.attention_blocks:
                    tokens = block(tokens)
            
            embedding = tokens.mean(dim=1)
            enhanced, _ = self.prototype_module(embedding)
        
        # Classification
        logits = self.classifier(enhanced)
        
        output = {'logits': logits}
        
        if return_quality:
            output['quality_score'] = self.quality_branch(x_input)
        
        return output
    
    def get_diversity_loss(self):
        return self.prototype_module.diversity_loss()

# ============================================================================
# TRAINING
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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        output = model(images, return_quality=True)
        
        focal = focal_loss_fn(output['logits'], labels)
        sup_loss = (output['quality_score'] * focal).mean()
        proto_loss = model.get_diversity_loss()
        
        loss = sup_loss + cfg.prototype_loss_weight * proto_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def train_and_evaluate_backbone(config, backbone_name, display_name):
    print(f"\n{'='*70}")
    print(f"BACKBONE: {display_name}")
    print(f"{'='*70}")
    
    # Cache data
    if not hasattr(train_and_evaluate_backbone, 'data_cache'):
        print("Loading data (first time)...")
        train_and_evaluate_backbone.data_cache = prepare_data(config)
        print("✓ Data loaded!")
    
    data = train_and_evaluate_backbone.data_cache
    
    try:
        # Create model
        model = MPSRetNet_Backbone(config, backbone_name).to(device)
        num_params = count_parameters(model)
        print(f"  Parameters: {num_params:.1f}M")
        
        # Training setup
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_acc = 0
        patience = 0
        max_patience = 8
        
        start_time = time.time()
        
        for epoch in range(config.num_epochs):
            train_loss = train_epoch(model, data['train_loader'], optimizer, config)
            val_metrics = evaluate(model, data['val_loader'])
            
            scheduler.step(val_metrics['accuracy'])
            
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                patience = 0
            else:
                patience += 1
            
            if epoch % 5 == 0 or patience == max_patience:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Time: {elapsed:.0f}s")
            
            if patience >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Test evaluation
        test_metrics = evaluate(model, data['test_loader'])
        
        total_time = time.time() - start_time
        print(f"\n  ✓ COMPLETE in {total_time/60:.1f} min")
        print(f"  Results: Acc={test_metrics['accuracy']:.2f}%, F1={test_metrics['f1']:.2f}%, κ={test_metrics['kappa']:.2f}%")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return test_metrics, num_params
        
    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'accuracy': 0.0, 'f1': 0.0, 'kappa': 0.0}, 0.0

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("TESTING 7 BACKBONE ARCHITECTURES")
    print("="*80)
    
    backbones = [
        ('resnet50', 'ResNet-50', 25.6),
        ('resnet101', 'ResNet-101', 44.5),
        ('densenet121', 'DenseNet-121', 8.0),
        ('efficientnet_b3', 'EfficientNet-B3', 12.2),
        ('vit_b_16', 'ViT-B/16', 86.6),
        ('convnext_small', 'ConvNeXt-Small', 50.2),
        ('convnext_base', 'ConvNeXt-Base', 88.6)
    ]
    
    results = []
    overall_start = time.time()
    
    for backbone_code, display_name, expected_params in backbones:
        try:
            metrics, actual_params = train_and_evaluate_backbone(config, backbone_code, display_name)
            
            results.append({
                'Backbone': display_name,
                'Params (M)': f"{expected_params}",
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 (%)': f"{metrics['f1']:.2f}",
                'κ (%)': f"{metrics['kappa']:.2f}"
            })
            
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR with {display_name}: {e}")
            results.append({
                'Backbone': display_name,
                'Params (M)': f"{expected_params}",
                'Accuracy (%)': 'FAILED',
                'F1 (%)': 'FAILED',
                'κ (%)': 'FAILED'
            })
    
    total_time = time.time() - overall_start
    
    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(config.results_dir, 'table_backbone_comparison.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("ALL BACKBONES TESTED!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nResults saved to: {output_file}")
    print(f"\n{'='*80}")
    print("FINAL RESULTS:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()