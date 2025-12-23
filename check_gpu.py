#!/usr/bin/env python3
"""
GPU Diagnostic Script - Check if PyTorch can see and use your GPU
"""

import sys

print("="*80)
print("GPU DIAGNOSTIC CHECK")
print("="*80)

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
except ImportError:
    print("✗ PyTorch NOT installed!")
    sys.exit(1)

# Check CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    # Test GPU
    print("\n" + "="*80)
    print("TESTING GPU OPERATIONS")
    print("="*80)
    
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        print("✓ Can create tensor on GPU")
        
        # Matrix multiplication
        y = torch.mm(x, x)
        print("✓ Can perform operations on GPU")
        
        # Check memory
        print(f"✓ Allocated memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"✓ Cached memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        print("\n✓✓✓ GPU IS WORKING CORRECTLY ✓✓✓")
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        
else:
    print("\n✗✗✗ NO GPU DETECTED ✗✗✗")
    print("\nPossible reasons:")
    print("1. No NVIDIA GPU in system")
    print("2. CUDA drivers not installed")
    print("3. PyTorch CPU-only version installed")
    print("\nTo fix:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*80)