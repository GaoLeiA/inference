#!/usr/bin/env python3
"""
Direct test of xinference client API to understand the correct usage
"""

from xinference.client import Client

endpoint = "http://192.168.30.161:9997"
client = Client(endpoint)

print("="*60)
print("Testing Xinference Client API")
print("="*60)

# Test 1: list_models()
print("\n1. Testing list_models():")
model_uids = client.list_models()
print(f"   Type: {type(model_uids)}")
print(f"   Content: {model_uids}")

# Test 2: get_model() for each UID
print("\n2. Testing get_model() for each UID:")
for uid in model_uids:
    print(f"\n   UID: {uid}")
    try:
        model = client.get_model(uid)
        print(f"   Type: {type(model)}")
        print(f"   Dir: {[x for x in dir(model) if not x.startswith('_')]}")
        
        # Try different ways to access model_name
        print(f"\n   Trying to get model_name:")
        print(f"     - getattr(model, 'model_name'): {getattr(model, 'model_name', 'NOT FOUND')}")
        print(f"     - model.model_name: {model.model_name if hasattr(model, 'model_name') else 'NOT FOUND'}")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
