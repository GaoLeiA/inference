#!/usr/bin/env python3
"""
Quick test script to verify model detection is working correctly.
"""

import sys
sys.path.insert(0, "/home/gaolei/inference/examples")

from mineru_http_client_example import check_model_running

endpoint = "http://192.168.30.161:9997"

print("="*60)
print("Testing Model Detection")
print("="*60)

result = check_model_running(endpoint, "mineru-vlm")

print("\n" + "="*60)
if result:
    print(f"✓ Result: Found model with ID: {result}")
else:
    print("✗ Result: No model found")
print("="*60)
