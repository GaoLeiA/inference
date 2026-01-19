# Copyright 2022-2025 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example: Launch MinerU VLM model via Xinference with vLLM engine

This example demonstrates how to:
1. Launch the mineru-vlm model using Xinference with vLLM engine
2. Use the OpenAI-compatible API to process document images
3. Extract text and structure from PDF pages

Prerequisites:
    pip install xinference
    pip install "vllm>=0.6.0"
    pip install openai
    pip install pdf2image  # For PDF to image conversion
    
Usage:
    # Method 1: Start Xinference server first, then run this script
    xinference-local --host 0.0.0.0 --port 9997
    python launch_mineru_vllm.py --pdf your_document.pdf
    
    # Method 2: Let the script handle everything
    python launch_mineru_vllm.py --pdf your_document.pdf --auto-start
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from typing import List, Optional


def check_xinference_server(endpoint: str) -> bool:
    """Check if Xinference server is running."""
    try:
        from xinference.client import Client
        client = Client(endpoint)
        client.list_models()
        return True
    except Exception:
        return False


def start_xinference_server(host: str = "0.0.0.0", port: int = 9997) -> subprocess.Popen:
    """
    Start Xinference server as a subprocess.
    
    Returns:
        subprocess.Popen: The server process
    """
    print(f"Starting Xinference server on {host}:{port}...")
    
    # Use CREATE_NEW_PROCESS_GROUP on Windows to allow proper cleanup
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    
    # Try xinference-local command first
    process = subprocess.Popen(
        ["xinference-local", "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout for debugging
        **kwargs
    )
    
    # Wait for server to start
    endpoint = f"http://localhost:{port}"
    for i in range(60):  # Wait up to 60 seconds
        time.sleep(1)
        
        # Check if process died
        if process.poll() is not None:
            # Process died, get output for debugging
            output, _ = process.communicate()
            print(f"Xinference server failed to start!")
            print(f"Output: {output.decode('utf-8', errors='ignore')}")
            raise RuntimeError("Xinference server process terminated unexpectedly")
        
        if check_xinference_server(endpoint):
            print(f"Xinference server started successfully on port {port}")
            return process
        print(f"Waiting for server to start... ({i+1}s)")
    
    # Timeout - kill process and show output
    process.terminate()
    output, _ = process.communicate()
    print(f"Server output: {output.decode('utf-8', errors='ignore')}")
    raise RuntimeError("Xinference server failed to start within 60 seconds")




def launch_mineru_model(
    endpoint: str = "http://localhost:9997",
    model_name: str = "mineru-vlm",
    size_in_billions: str = "1_2",
    gpu_memory_utilization: float = 0.9,
) -> str:
    """
    Launch the MinerU VLM model via Xinference with vLLM engine.
    
    Args:
        endpoint: Xinference server endpoint
        model_name: Name of the model (mineru-vlm)
        size_in_billions: Model size parameter
        gpu_memory_utilization: GPU memory fraction to use
    
    Returns:
        model_uid: The unique identifier for the launched model
    """
    from xinference.client import Client
    
    client = Client(endpoint)
    
    # Check if model is already running
    running_models = client.list_models()
    for model in running_models:
        if model.get("model_name") == model_name:
            model_uid = model["id"]
            print(f"Model {model_name} is already running with uid: {model_uid}")
            return model_uid
    
    # Launch the model with vLLM engine
    print(f"Launching {model_name} with vLLM engine...")
    print(f"  - Model size: {size_in_billions}B")
    print(f"  - GPU memory utilization: {gpu_memory_utilization}")
    print(f"  - Engine: vllm")
    
    model_uid = client.launch_model(
        model_name=model_name,
        model_type="LLM",
        model_engine="vllm",
        model_size_in_billions=size_in_billions,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    print(f"Model launched successfully!")
    print(f"  - Model UID: {model_uid}")
    print(f"  - API Endpoint: {endpoint}/v1")
    
    return model_uid


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[str]:
    """
    Convert PDF pages to base64-encoded images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for converting PDF to images
        
    Returns:
        List of base64-encoded PNG images
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: pdf2image is not installed.")
        print("Install it with: pip install pdf2image")
        print("Also install poppler:")
        print("  - Windows: https://github.com/oschwartz10612/poppler-windows/releases")
        print("  - Linux: apt-get install poppler-utils")
        print("  - macOS: brew install poppler")
        sys.exit(1)
    
    print(f"Converting PDF to images (DPI: {dpi})...")
    images = convert_from_path(pdf_path, dpi=dpi)
    base64_images = []
    
    for i, img in enumerate(images):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode("utf-8")
        base64_images.append(base64_data)
        print(f"  Converted page {i + 1}/{len(images)}")
    
    return base64_images


def process_document_with_vlm(
    endpoint: str,
    model_uid: str,
    base64_images: List[str],
    prompt: str = "Please extract all text from this document page and describe its layout structure."
) -> List[str]:
    """
    Process document images using the MinerU VLM via OpenAI API.
    
    Args:
        endpoint: Xinference server endpoint
        model_uid: The launched model's UID
        base64_images: List of base64-encoded images
        prompt: User prompt for document extraction
        
    Returns:
        List of extracted text results for each page
    """
    from openai import OpenAI
    
    # Connect to Xinference OpenAI-compatible endpoint
    client = OpenAI(
        base_url=f"{endpoint}/v1",
        api_key="not-used"  # Xinference doesn't require API key
    )
    
    results = []
    
    for i, image_b64 in enumerate(base64_images):
        print(f"Processing page {i + 1}/{len(base64_images)}...")
        
        # Create message with image (Qwen2VL style)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        try:
            # Call the VLM
            response = client.chat.completions.create(
                model=model_uid,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
            )
            
            result = response.choices[0].message.content
            results.append(result)
            print(f"  Page {i + 1} processed successfully")
        except Exception as e:
            print(f"  Error processing page {i + 1}: {e}")
            results.append(f"[Error: {e}]")
    
    return results


def save_results(results: List[str], output_path: str):
    """Save extraction results to a markdown file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Document Extraction Results\n\n")
        for i, result in enumerate(results):
            f.write(f"## Page {i + 1}\n\n")
            f.write(result if result else "[No content extracted]")
            f.write("\n\n---\n\n")
    print(f"Results saved to: {output_path}")


def main():
    """Main function demonstrating the full workflow."""
    parser = argparse.ArgumentParser(
        description="Launch MinerU VLM with vLLM via Xinference and process documents"
    )
    parser.add_argument(
        "--endpoint", 
        default="http://localhost:9997", 
        help="Xinference server endpoint"
    )
    parser.add_argument(
        "--pdf", 
        type=str, 
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.md", 
        help="Output file path"
    )
    parser.add_argument(
        "--gpu-memory", 
        type=float, 
        default=0.9, 
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically start Xinference server if not running"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9997,
        help="Port for Xinference server (used with --auto-start)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion"
    )
    
    args = parser.parse_args()
    
    server_process = None
    endpoint = args.endpoint
    
    try:
        # Step 1: Ensure Xinference server is running
        print("=" * 60)
        print("Step 1: Checking Xinference Server")
        print("=" * 60)
        
        if not check_xinference_server(endpoint):
            if args.auto_start:
                server_process = start_xinference_server(port=args.port)
                endpoint = f"http://localhost:{args.port}"
            else:
                print(f"Error: Xinference server is not running at {endpoint}")
                print("\nPlease start it manually:")
                print(f"  xinference-local --host 0.0.0.0 --port {args.port}")
                print("\nOr use --auto-start to start it automatically:")
                print(f"  python {sys.argv[0]} --auto-start --pdf your_document.pdf")
                sys.exit(1)
        else:
            print(f"Xinference server is running at {endpoint}")
        
        # Step 2: Launch the model
        print("\n" + "=" * 60)
        print("Step 2: Launching MinerU VLM Model with vLLM Engine")
        print("=" * 60)
        
        model_uid = launch_mineru_model(
            endpoint=endpoint,
            gpu_memory_utilization=args.gpu_memory,
        )
        
        # Step 3: Process PDF if provided
        if args.pdf:
            if not os.path.exists(args.pdf):
                print(f"Error: PDF file not found: {args.pdf}")
                sys.exit(1)
            
            print("\n" + "=" * 60)
            print("Step 3: Converting PDF to Images")
            print("=" * 60)
            
            base64_images = pdf_to_images(args.pdf, dpi=args.dpi)
            
            print("\n" + "=" * 60)
            print("Step 4: Processing Document with VLM")
            print("=" * 60)
            
            results = process_document_with_vlm(
                endpoint=endpoint,
                model_uid=model_uid,
                base64_images=base64_images,
            )
            
            print("\n" + "=" * 60)
            print("Step 5: Saving Results")
            print("=" * 60)
            
            save_results(results, args.output)
            
        else:
            print("\n" + "=" * 60)
            print("Model Ready!")
            print("=" * 60)
            print(f"\nThe MinerU VLM model is now running with vLLM engine.")
            print(f"\nYou can use it via OpenAI API:")
            print(f"  - Base URL: {endpoint}/v1")
            print(f"  - Model: {model_uid}")
            print(f"\nExample with curl:")
            print(f'  curl {endpoint}/v1/chat/completions \\')
            print(f'    -H "Content-Type: application/json" \\')
            print(f'    -d \'{{"model": "{model_uid}", "messages": [...]}}\' ')
            print(f"\nOr process a PDF:")
            print(f"  python {sys.argv[0]} --pdf your_document.pdf")
            
            # Keep running until interrupted
            if server_process:
                print("\nPress Ctrl+C to stop the server...")
                try:
                    server_process.wait()
                except KeyboardInterrupt:
                    pass
                    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        if server_process:
            print("Stopping Xinference server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
