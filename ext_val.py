#!/usr/bin/env python3
"""
External Validation Script for MPS-RetNet
Uses saved checkpoints to evaluate on Messidor-2 dataset
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the main script directory to path
sys.path.append('.')

import mpsretnet  # Assuming your main script is named mpsretnet.py
from mpsretnet import Config, MPSRetNet, load_checkpoint, validate, calibration_metrics, bootstrap_ci
from mpsretnet import plot_confusion_matrix

def external_validation(checkpoint_path: str, config: Config):
    """External validation on Messidor-2 dataset using specified checkpoint"""
    print(f"\n{'='*70}\nExternal Validation (Messidor-2)\n{'='*70}")
    
    # Load external data
    external_data = mpsretnet.prepare_external_data(config)
    if not external_data:
        print("Skipped - dataset not found")
        return None
    
    # Load model from checkpoint
    model = MPSRetNet(config).to(mpsretnet.device)
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, mpsretnet.device)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    # Run validation
    external_metrics = validate(model, external_data['loader'], mpsretnet.device)
    external_cal = calibration_metrics(external_metrics['labels'], external_metrics['probabilities'])
    ci_lower, ci_upper = bootstrap_ci(external_metrics['labels'], external_metrics['predictions'])
    
    print(f"\nExternal Results ({len(external_data['samples'])} images):")
    print(f"  Accuracy: {external_metrics['accuracy']*100:.2f}%")
    print(f"  F1: {external_metrics['f1']*100:.2f}%")
    print(f"  Kappa: {external_metrics['kappa']*100:.2f}%")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Save results
    plot_confusion_matrix(external_metrics['labels'], external_metrics['predictions'], 
                         config.classes, f"{config.results_dir}/fig_confusion_matrix_external.pdf")
    
    # Save metrics
    df_ext = mpsretnet.pd.DataFrame({
        'Class': config.classes + ['Macro Avg'],
        'Sensitivity (%)': list(external_metrics['recall_per_class']*100) + [external_metrics['recall']*100],
        'Specificity (%)': list(external_metrics['specificity_per_class']*100) + [external_metrics['specificity_per_class'].mean()*100],
        'F1 (%)': list(external_metrics['f1_per_class']*100) + [external_metrics['f1']*100]
    })
    df_ext.to_csv(f"{config.results_dir}/table4_external_validation.csv", index=False)
    
    return {'metrics': external_metrics, 'calibration': external_cal, 'ci': (ci_lower, ci_upper)}

if __name__ == "__main__":
    # Initialize configuration
    config = mpsretnet.Config()
    
    # Checkpoint path (use the best model from Phase 1)
    checkpoint_path = "outputs/best_model.pt"
    
    # Run external validation
    external = external_validation(checkpoint_path, config)
    
    print(f"\nExternal validation completed. Results saved to: {config.results_dir}/")