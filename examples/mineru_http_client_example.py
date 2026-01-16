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
Example: Use MinerU CLI with Xinference vLLM Server (HTTP Client Mode)

This example demonstrates how to:
1. Launch MinerU VLM model via Xinference with vLLM engine
2. Use MinerU CLI in hybrid-http-client mode to connect to the Xinference server
3. Process PDFs through the external vLLM server for document parsing

Prerequisites:
    pip install xinference
    pip install "vllm>=0.6.0"
    pip install "mineru[all]>=2.5.0"
    
Usage:
    # Ensure Xinference server is running with mineru-vlm model
    xinference-local --host 0.0.0.0 --port 9997
    xinference launch --model-engine vllm --model-name mineru-vlm --size-in-billions 1_2
    
    # Then run this script
    python mineru_http_client_example.py --pdf input.pdf --output output_dir
    
    # Or use MinerU CLI directly
    mineru -p input.pdf -o output_dir -b hybrid-http-client -u http://localhost:9997/v1
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Optional


def check_xinference_server(endpoint: str) -> bool:
    """Check if Xinference server is running."""
    try:
        from xinference.client import Client
        client = Client(endpoint)
        client.list_models()
        return True
    except Exception:
        return False


def check_model_running(endpoint: str, model_name: str = "mineru-vlm") -> Optional[str]:
    """Check if the model is already running and return its UID."""
    try:
        from xinference.client import Client
        client = Client(endpoint)
        for model in client.list_models():
            if model.get("model_name") == model_name:
                return model["id"]
    except Exception:
        pass
    return None


def launch_model_on_xinference(
    endpoint: str,
    model_name: str = "mineru-vlm",
    size_in_billions: str = "1_2",
    gpu_memory_utilization: float = 0.9,
) -> str:
    """
    Launch the mineru-vlm model on Xinference with vLLM engine.
    
    Returns:
        model_uid: The model's unique identifier
    """
    from xinference.client import Client
    
    client = Client(endpoint)
    
    # Check if model already exists
    existing_uid = check_model_running(endpoint, model_name)
    if existing_uid:
        print(f"Model {model_name} is already running (UID: {existing_uid})")
        return existing_uid
    
    print(f"Launching {model_name} with vLLM engine...")
    model_uid = client.launch_model(
        model_name=model_name,
        model_type="LLM",
        model_engine="vllm",
        model_size_in_billions=size_in_billions,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print(f"Model launched successfully (UID: {model_uid})")
    
    # Wait a moment for the model to be fully ready
    time.sleep(2)
    
    return model_uid


def run_mineru_cli(
    pdf_path: str,
    output_dir: str,
    server_url: str,
    backend: str = "hybrid-http-client",
    language: str = "ch",
    verbose: bool = True,
) -> bool:
    """
    Run MinerU CLI with HTTP client mode to process PDF.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Output directory
        server_url: URL of the OpenAI-compatible server (e.g., http://localhost:9997/v1)
        backend: MinerU backend (hybrid-http-client, vlm-http-client)
        language: OCR language (ch, en, etc.)
        verbose: Print detailed output
        
    Returns:
        True if successful, False otherwise
    """
    # Build command
    cmd = [
        "mineru",
        "-p", pdf_path,
        "-o", output_dir,
        "-b", backend,
        "-u", server_url,
        "-l", language,
    ]
    
    if verbose:
        print("=" * 60)
        print("Running MinerU CLI")
        print("=" * 60)
        print(f"  PDF Input: {pdf_path}")
        print(f"  Output Dir: {output_dir}")
        print(f"  Backend: {backend}")
        print(f"  Server URL: {server_url}")
        print(f"  Language: {language}")
        print(f"\nCommand: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
        )
        
        if result.returncode == 0:
            print(f"\n✓ PDF processed successfully!")
            print(f"  Output saved to: {output_dir}")
            return True
        else:
            print(f"\n✗ MinerU processing failed (exit code: {result.returncode})")
            if not verbose and result.stderr:
                print(f"  Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("\n✗ Error: 'mineru' command not found.")
        print("  Please install MinerU: pip install 'mineru[all]>=2.5.0'")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Use MinerU with Xinference vLLM Server for PDF parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires Xinference server already running)
  python mineru_http_client_example.py --pdf document.pdf --output results/

  # Specify custom endpoint
  python mineru_http_client_example.py --pdf document.pdf --output results/ --endpoint http://localhost:9997

  # Use English language
  python mineru_http_client_example.py --pdf document.pdf --output results/ --language en
        """
    )
    
    parser.add_argument(
        "--pdf", 
        type=str, 
        required=True, 
        help="Input PDF file path"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Output directory path"
    )
    parser.add_argument(
        "--endpoint", 
        default="http://localhost:9997",
        help="Xinference server endpoint (default: http://localhost:9997)"
    )
    parser.add_argument(
        "--language", 
        default="ch", 
        help="OCR language: ch (Chinese), en (English), etc. (default: ch)"
    )
    parser.add_argument(
        "--backend",
        default="hybrid-http-client",
        choices=["hybrid-http-client", "vlm-http-client"],
        help="MinerU backend to use (default: hybrid-http-client)"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)"
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Skip model launching (assume already running)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)
    
    verbose = not args.quiet
    
    try:
        # Step 1: Check Xinference server
        if verbose:
            print("=" * 60)
            print("Step 1: Checking Xinference Server")
            print("=" * 60)
        
        if not check_xinference_server(args.endpoint):
            print(f"Error: Xinference server is not running at {args.endpoint}")
            print("\nPlease start it with:")
            print(f"  xinference-local --host 0.0.0.0 --port 9997")
            sys.exit(1)
        
        if verbose:
            print(f"✓ Xinference server is running at {args.endpoint}")
        
        # Step 2: Launch model if needed
        if not args.skip_launch:
            if verbose:
                print("\n" + "=" * 60)
                print("Step 2: Launching MinerU VLM Model")
                print("=" * 60)
            
            model_uid = launch_model_on_xinference(
                endpoint=args.endpoint,
                gpu_memory_utilization=args.gpu_memory,
            )
        else:
            model_uid = check_model_running(args.endpoint)
            if not model_uid:
                print("Error: No mineru-vlm model running. Remove --skip-launch to auto-launch.")
                sys.exit(1)
        
        # Step 3: Run MinerU CLI
        if verbose:
            print("\n" + "=" * 60)
            print("Step 3: Processing PDF with MinerU")
            print("=" * 60)
        
        # MinerU expects the OpenAI-compatible endpoint at /v1
        server_url = f"{args.endpoint}/v1"
        
        success = run_mineru_cli(
            pdf_path=args.pdf,
            output_dir=args.output,
            server_url=server_url,
            backend=args.backend,
            language=args.language,
            verbose=verbose,
        )
        
        if not success:
            sys.exit(1)
        
        # Print summary
        if verbose:
            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            print(f"  Input PDF: {args.pdf}")
            print(f"  Output Directory: {args.output}")
            print(f"  Model UID: {model_uid}")
            print(f"  Server: {args.endpoint}")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
