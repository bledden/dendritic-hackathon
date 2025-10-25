#!/usr/bin/env python3
"""
GPU Verification Script for AMD MI300x
Run this first on your MI300x instance to verify ROCm/PyTorch setup
"""

import sys

def test_pytorch():
    """Test PyTorch installation"""
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def test_cuda_available():
    """Test CUDA/ROCm availability"""
    import torch
    if torch.cuda.is_available():
        print("✅ CUDA/ROCm is available")
        print(f"   CUDA version: {torch.version.cuda}")
        return True
    else:
        print("❌ CUDA/ROCm not available")
        print("   Make sure ROCm is installed and GPU is accessible")
        return False

def test_gpu_count():
    """Test GPU detection"""
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"✅ Detected {gpu_count} GPU(s)")
    return gpu_count > 0

def test_gpu_properties():
    """Test GPU properties"""
    import torch
    if not torch.cuda.is_available():
        return False

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n📊 GPU {i} Properties:")
        print(f"   Name: {torch.cuda.get_device_name(i)}")
        print(f"   Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"   Compute capability: {props.major}.{props.minor}")
        print(f"   Multi-processor count: {props.multi_processor_count}")
    return True

def test_gpu_computation():
    """Test actual GPU computation"""
    import torch
    try:
        print("\n🔬 Testing GPU computation...")

        # Create tensors on GPU
        x = torch.randn(5000, 5000, device='cuda')
        y = torch.randn(5000, 5000, device='cuda')

        # Matrix multiplication
        z = torch.mm(x, y)

        # Sync to ensure completion
        torch.cuda.synchronize()

        print("✅ GPU computation successful")
        print(f"   Result shape: {z.shape}")
        print(f"   Result device: {z.device}")
        return True
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def test_mixed_precision():
    """Test mixed precision (FP16) support"""
    import torch
    try:
        print("\n🔬 Testing mixed precision (FP16)...")

        with torch.cuda.amp.autocast():
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.mm(x, x)

        torch.cuda.synchronize()
        print("✅ Mixed precision (FP16) supported")
        return True
    except Exception as e:
        print(f"❌ Mixed precision failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Checking dependencies...")

    dependencies = [
        'whisper',
        'datasets',
        'soundfile',
        'numpy',
        'tqdm'
    ]

    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - NOT INSTALLED")
            all_ok = False

    return all_ok

def main():
    """Run all tests"""
    print("=" * 70)
    print("🚀 AMD MI300x GPU Verification Script")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("PyTorch Installation", test_pytorch()))

    if results[0][1]:  # Only continue if PyTorch is installed
        results.append(("CUDA/ROCm Available", test_cuda_available()))
        results.append(("GPU Detection", test_gpu_count()))

        if results[1][1]:  # If CUDA is available
            results.append(("GPU Properties", test_gpu_properties()))
            results.append(("GPU Computation", test_gpu_computation()))
            results.append(("Mixed Precision", test_mixed_precision()))

    results.append(("Dependencies", test_dependencies()))

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:30s} {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready for training.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
