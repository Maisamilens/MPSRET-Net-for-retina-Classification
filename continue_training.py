#!/usr/bin/env python3
"""
Continue Training Script for MPS-RetNet
Loads from checkpoint and continues training for remaining folds
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the main script directory to path
sys.path.append('.')

import mpsretnet
from mpsretnet import Config, MPSRetNet, load_checkpoint, train_model, prepare_data_for_fold

def continue_training(checkpoint_path: str, fold_idx: int, seed: int, config: Config):
    """Continue training from checkpoint for a specific fold"""
    print(f"\n{'='*70}\nContinuing Training - Fold {fold_idx+1}/{config.n_folds} (Seed: {seed})\n{'='*70}")
    
    # Load data for this fold
    all_samples, all_labels = mpsretnet.load_images_from_directory(config.data_root, config.classes)
    fold_data = prepare_data_for_fold(config, all_samples, all_labels, fold_idx, config.n_folds, seed)
    
    # Continue training using the checkpoint
    _, _, test_metrics, _, calibration, train_time = train_model(
        config, seed=seed, fold_data=fold_data, checkpoint_path=checkpoint_path
    )
    
    return test_metrics, calibration, train_time

if __name__ == "__main__":
    # Initialize configuration
    config = mpsretnet.Config()
    
    # Checkpoint path (use the best model from Phase 1)
    checkpoint_path = "outputs/best_model.pt"
    
    # Continue with remaining folds (assuming fold 1 is already done)
    remaining_folds = [(i, seed) for i, seed in enumerate(config.cv_seeds) if i > 0]
    
    for fold_idx, seed in remaining_folds:
        test_metrics, calibration, train_time = continue_training(checkpoint_path, fold_idx, seed, config)
        
        print(f"\nFold {fold_idx+1} Results:")
        print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  F1: {test_metrics['f1']*100:.2f}%")
        print(f"  Kappa: {test_metrics['kappa']*100:.2f}%")
        print(f"  Training Time: {train_time/60:.2f} minutes")
    
    print(f"\nAll remaining folds completed!")