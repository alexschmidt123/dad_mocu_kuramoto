#!/usr/bin/env python3
"""
Setup Verification Script
=========================
Checks that everything is installed correctly for GPU acceleration.
Run this after running optimize_rtx4090.py
"""

import sys
import os

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text)
    print("="*80)

def check_python_version():
    """Check Python version."""
    print("\n1. Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("   ✓ Python version OK")
        return True
    else:
        print("   ✗ Python 3.9+ required")
        return False

def check_imports():
    """Check required packages."""
    print("\n2. Checking required packages...")
    
    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'yaml': 'PyYAML',
        'torch': 'PyTorch',
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            if module == 'yaml':
                import yaml
                print(f"   ✓ {name}: {yaml.__version__ if hasattr(yaml, '__version__') else 'installed'}")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"   ✓ {name}: {version}")
        except ImportError:
            print(f"   ✗ {name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability."""
    print("\n3. Checking CUDA/GPU setup...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"   ✓ CUDA version: {torch.version.cuda}")
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"   ✓ VRAM: {props.total_memory / 1e9:.1f} GB")
            
            major, minor = torch.cuda.get_device_capability(0)
            print(f"   ✓ Compute Capability: {major}.{minor}")
            
            if major >= 8:
                print(f"   ✓ Tensor Cores: Available (Ampere or newer)")
            elif major >= 7:
                print(f"   ✓ Tensor Cores: Available (Volta/Turing)")
            else:
                print(f"   ⚠ Tensor Cores: Not available (older architecture)")
            
            # Test GPU computation
            try:
                x = torch.randn(100, 100, device='cuda')
                y = torch.randn(100, 100, device='cuda')
                z = torch.mm(x, y)
                print(f"   ✓ GPU computation test: PASSED")
            except Exception as e:
                print(f"   ✗ GPU computation test: FAILED ({e})")
                return False
            
            return True
        else:
            print("   ✗ CUDA not available")
            print("\n   Install CUDA-enabled PyTorch:")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False
            
    except ImportError:
        print("   ✗ PyTorch not installed")
        return False

def check_amp():
    """Check mixed precision support."""
    print("\n4. Checking mixed precision (AMP) support...")
    
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("   ✓ Mixed precision: Available")
        
        # Test AMP
        import torch
        if torch.cuda.is_available():
            try:
                scaler = GradScaler()
                with autocast():
                    x = torch.randn(10, 10, device='cuda')
                    y = torch.randn(10, 10, device='cuda')
                    z = torch.mm(x, y)
                print("   ✓ AMP test: PASSED")
                return True
            except Exception as e:
                print(f"   ✗ AMP test: FAILED ({e})")
                return False
        else:
            print("   ⚠ Cannot test AMP without GPU")
            return True
    except ImportError:
        print("   ✗ Mixed precision: Not available")
        return False

def check_optional_packages():
    """Check optional but recommended packages."""
    print("\n5. Checking optional packages...")
    
    optional = {
        'torchdiffeq': 'GPU ODE solver (HIGHLY RECOMMENDED for 10x speedup)',
        'psutil': 'System monitoring',
        'tqdm': 'Progress bars',
    }
    
    for module, description in optional.items():
        try:
            __import__(module)
            print(f"   ✓ {module}: installed - {description}")
        except ImportError:
            print(f"   ⚠ {module}: NOT installed - {description}")
            print(f"      Install with: pip install {module}")

def check_project_files():
    """Check that project files exist."""
    print("\n6. Checking project files...")
    
    required_files = [
        'main_demo.py',
        'comprehensive_test.py',
        'configs/exp_fixedK.yaml',
        'core/kuramoto_env.py',
        'surrogate/mpnn_surrogate.py',
        'surrogate/train_surrogate.py',
    ]
    
    optimized_files = [
        'configs/exp_fixedK_rtx4090.yaml',
        'main_demo_rtx4090.py',
    ]
    
    all_ok = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"   ✓ {filepath}")
        else:
            print(f"   ✗ {filepath} - MISSING")
            all_ok = False
    
    print("\n   Optimized files (created by optimize_rtx4090.py):")
    for filepath in optimized_files:
        if os.path.exists(filepath):
            print(f"   ✓ {filepath}")
        else:
            print(f"   ⚠ {filepath} - Run optimize_rtx4090.py to create")
    
    return all_ok

def check_gpu_settings():
    """Check GPU settings and recommendations."""
    print("\n7. Checking GPU settings...")
    
    try:
        import torch
        if torch.cuda.is_available():
            # Check cuDNN
            cudnn_available = torch.backends.cudnn.is_available()
            print(f"   cuDNN available: {cudnn_available}")
            
            if cudnn_available:
                print(f"   ✓ cuDNN version: {torch.backends.cudnn.version()}")
                
                # Check benchmark mode
                benchmark = torch.backends.cudnn.benchmark
                print(f"   Benchmark mode: {benchmark}")
                if not benchmark:
                    print("      ℹ Tip: Enable with torch.backends.cudnn.benchmark = True")
            
            # Check TF32
            if hasattr(torch.backends.cuda, 'matmul'):
                tf32 = torch.backends.cuda.matmul.allow_tf32
                print(f"   TF32 enabled: {tf32}")
            
            return True
        else:
            print("   ⚠ No GPU available")
            return False
    except:
        return False

def run_quick_benchmark():
    """Run a quick GPU benchmark."""
    print("\n8. Running quick GPU benchmark...")
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("   ⚠ Skipping (no GPU)")
            return True
        
        # Matrix multiplication benchmark
        size = 4096
        print(f"   Testing matrix multiplication ({size}x{size})...")
        
        A = torch.randn(size, size, device='cuda')
        B = torch.randn(size, size, device='cuda')
        
        # Warmup
        C = torch.mm(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        n_iter = 100
        start = time.time()
        for _ in range(n_iter):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tflops = (2 * size**3 * n_iter) / elapsed / 1e12
        print(f"   ✓ Performance: {tflops:.2f} TFLOPS")
        
        # Performance guide
        if tflops > 50:
            print(f"   ✓ Excellent performance! GPU is running optimally.")
        elif tflops > 30:
            print(f"   ✓ Good performance.")
        elif tflops > 15:
            print(f"   ⚠ Moderate performance. Check GPU clocks and power settings.")
        else:
            print(f"   ⚠ Low performance. Check:")
            print(f"      - GPU not throttling (temperature, power)")
            print(f"      - CUDA drivers up to date")
            print(f"      - No other processes using GPU")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Benchmark failed: {e}")
        return False

def print_recommendations():
    """Print final recommendations."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    try:
        import torch
        if torch.cuda.is_available():
            print("\n✓ GPU Setup Complete!")
            print("\nNext steps:")
            print("  1. Run tests: python comprehensive_test.py")
            print("  2. Quick training: python main_demo_rtx4090.py")
            print("  3. Monitor GPU: watch -n 1 nvidia-smi")
            
            print("\nOptimization tips:")
            print("  • Set GPU to max performance: sudo nvidia-smi -pm 1")
            print("  • Install torchdiffeq for 10x ODE speedup: pip install torchdiffeq")
            print("  • Monitor training with: watch -n 1 nvidia-smi")
            
        else:
            print("\n⚠ GPU Not Available")
            print("\nThe code will run on CPU, but GPU is recommended for best performance.")
            print("\nTo enable GPU:")
            print("  1. Install CUDA toolkit")
            print("  2. Install PyTorch with CUDA:")
            print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print("  3. Run this script again to verify")
    except:
        print("\nPlease install PyTorch first:")
        print("  pip install torch")

def main():
    """Main verification workflow."""
    print_header("SETUP VERIFICATION")
    print("Checking your installation for GPU-accelerated Kuramoto simulation...")
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Required Packages", check_imports()))
    results.append(("CUDA/GPU", check_cuda()))
    results.append(("Mixed Precision", check_amp()))
    check_optional_packages()  # Not critical
    results.append(("Project Files", check_project_files()))
    check_gpu_settings()  # Not critical
    run_quick_benchmark()  # Not critical
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for check, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:10} {check}")
    
    print("="*80)
    print(f"Result: {passed}/{total} checks passed")
    print("="*80)
    
    # Recommendations
    print_recommendations()
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())