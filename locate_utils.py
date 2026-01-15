import sys
import os

# Add potential paths
sys.path.append(os.getcwd())
sys.path.append(r'c:\projects\inference')

try:
    import qwen_omni_utils
    print(f"Found qwen_omni_utils at: {qwen_omni_utils.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")
    # Try to verify if we can find it by walking
    for root, dirs, files in os.walk(r'c:\projects\inference'):
        if 'qwen_omni_utils.py' in files:
            print(f"Found file at: {os.path.join(root, 'qwen_omni_utils.py')}")
