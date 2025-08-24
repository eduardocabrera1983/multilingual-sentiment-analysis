#!/usr/bin/env python
"""
GPU Setup Verification and Configuration Script
Run this to check GPU availability and set up CUDA support
"""

import subprocess
import sys
import torch
import platform

def check_gpu_status():
    """Check and display GPU information"""
    print("="*70)
    print("🖥️ GPU DETECTION AND SETUP")
    print("="*70)
    
    # System info
    print(f"\n📊 System Information:")
    print(f"   Python: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    
    # Check PyTorch and CUDA
    print(f"\n🔧 PyTorch Configuration:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"      Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("   ⚠️ CUDA is NOT available")
        print("\n   Possible reasons:")
        print("   1. NVIDIA drivers not installed")
        print("   2. PyTorch installed without CUDA support")
        print("   3. CUDA toolkit not installed")
    
    # Check NVIDIA drivers
    print(f"\n🎮 NVIDIA Driver Check:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ NVIDIA drivers installed")
            # Parse nvidia-smi output for driver version
            for line in result.stdout.split('\n'):
                if 'Driver Version' in line:
                    print(f"   {line.strip()}")
                    break
        else:
            print("   ❌ nvidia-smi not found")
    except FileNotFoundError:
        print("   ❌ nvidia-smi command not found")
        print("   Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
    except Exception as e:
        print(f"   ❌ Error checking NVIDIA drivers: {e}")
    
    return torch.cuda.is_available()

def install_cuda_pytorch():
    """Instructions for installing CUDA-enabled PyTorch"""
    print("\n" + "="*70)
    print("📦 INSTALLATION INSTRUCTIONS")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠️ CUDA support not detected. To enable GPU acceleration:")
        print("\n1️⃣ First, check your NVIDIA driver:")
        print("   Go to Device Manager > Display Adapters")
        print("   Right-click your NVIDIA GPU > Properties > Driver")
        print("   Ensure driver is up to date (version 450+ recommended)")
        
        print("\n2️⃣ Install CUDA-enabled PyTorch:")
        print("   Uninstall current PyTorch first:")
        print("   pip uninstall torch torchvision torchaudio")
        print("\n   Then install CUDA version:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n   Or for CUDA 11.8:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n3️⃣ Verify installation:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")
        
    else:
        print("\n✅ CUDA is already enabled! Your GPU is ready to use.")
        print("\nTo use GPU in your code:")
        print("   device = 'cuda' if torch.cuda.is_available() else 'cpu'")
        print("   model = model.to(device)")

def test_gpu_performance():
    """Quick GPU performance test"""
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("⚡ GPU PERFORMANCE TEST")
        print("="*70)
        
        device = torch.device('cuda')
        
        # Simple matrix multiplication test
        size = 10000
        print(f"\n🧪 Testing matrix multiplication ({size}x{size})...")
        
        # CPU test
        import time
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        print(f"   CPU Time: {cpu_time:.3f} seconds")
        
        # GPU test
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        torch.cuda.synchronize()  # Ensure GPU is ready
        
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Wait for computation to finish
        gpu_time = time.time() - start
        print(f"   GPU Time: {gpu_time:.3f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"\n   🚀 GPU Speedup: {speedup:.1f}x faster than CPU!")
        
        # Memory info
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

if __name__ == "__main__":
    # Check GPU status
    cuda_available = check_gpu_status()
    
    # Installation instructions
    install_cuda_pytorch()
    
    # Performance test if GPU available
    if cuda_available:
        test_gpu_performance()
    
    print("\n" + "="*70)
    print("✅ GPU check complete!")
    print("="*70)