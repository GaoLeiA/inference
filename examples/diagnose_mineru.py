#!/usr/bin/env python3
"""
Diagnostic script for MinerU vLLM integration

Usage:
    python diagnose_mineru.py
"""

import subprocess
import sys


def check_command(cmd, name):
    """Check if a command exists."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            timeout=5,
            text=True
        )
        print(f"✓ {name} is available")
        return True
    except FileNotFoundError:
        print(f"✗ {name} NOT FOUND")
        return False
    except subprocess.TimeoutExpired:
        print(f"✓ {name} is available (timed out but exists)")
        return True
    except Exception as e:
        print(f"? {name} check failed: {e}")
        return False


def check_gpu():
    """Check GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            timeout=5,
            text=True
        )
        print(f"✓ NVIDIA GPU detected")
        # Print GPU info
        lines = result.stdout.split('\n')
        for line in lines:
            if 'GeForce' in line or 'Tesla' in line or 'RTX' in line or 'TITAN' in line:
                print(f"  {line.strip()}")
        return True
    except Exception as e:
        print(f"✗ No NVIDIA GPU or nvidia-smi not available: {e}")
        return False


def check_python_packages():
    """Check required Python packages."""
    packages = {
        "xinference": "xinference",
        "vllm": "vllm",
        "mineru": "mineru[all]",
        "openai": "openai",
        "pdf2image": "pdf2image",
    }
    
    all_ok = True
    for module, install_name in packages.items():
        try:
            __import__(module)
            print(f"✓ {module} is installed")
        except ImportError:
            print(f"✗ {module} is NOT installed (pip install {install_name})")
            all_ok = False
    
    return all_ok


def check_xinference_server():
    """Check if Xinference server is running."""
    try:
        from xinference.client import Client
        client = Client("http://localhost:9997")
        models = client.list_models()
        print(f"✓ Xinference server is running on port 9997")
        if models:
            print(f"  Currently running models: {len(models)}")
            for model in models:
                print(f"    - {model.get('model_name')} ({model.get('id')})")
        return True
    except Exception as e:
        print(f"✗ Xinference server is NOT running: {e}")
        print(f"  Start it with: xinference-local --host 0.0.0.0 --port 9997")
        return False


def main():
    print("=" * 60)
    print("MinerU vLLM Integration Diagnostic")
    print("=" * 60)
    
    print("\n1. Checking system commands...")
    check_command(["python3", "--version"], "Python 3")
    check_command(["xinference-local", "--help"], "xinference-local")
    check_command(["mineru", "--help"], "mineru CLI")
    
    print("\n2. Checking GPU...")
    check_gpu()
    
    print("\n3. Checking Python packages...")
    check_python_packages()
    
    print("\n4. Checking Xinference server...")
    check_xinference_server()
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)
    print("\nIf you see any ✗ marks above, please fix those issues first.")
    print("\nCommon solutions:")
    print("  1. Install missing packages: pip install xinference vllm 'mineru[all]' openai pdf2image")
    print("  2. Start Xinference: xinference-local --host 0.0.0.0 --port 9997")
    print("  3. Check GPU drivers: nvidia-smi")


if __name__ == "__main__":
    main()
