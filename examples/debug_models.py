#!/usr/bin/env python3
"""Debug script to see what list_models returns"""

from xinference.client import Client

client = Client("http://192.168.30.161:9997")
models = client.list_models()

print(f"Type of models: {type(models)}")
print(f"Length: {len(models)}")
print(f"\nFirst model:")
print(f"  Type: {type(models[0])}")
print(f"  Content: {models[0]}")
print(f"\nAll models:")
for i, m in enumerate(models):
    print(f"{i}: {type(m)} -> {m}")
