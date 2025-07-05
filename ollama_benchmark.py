#!/usr/bin/env python3
"""
Ollama Benchmark Tool
This script benchmarks Ollama models and saves results to an Excel file.
"""

import requests
import json
import time
import subprocess
import psutil
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
import threading
import sys

class OllamaBenchmark:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.benchmark_file = "ollama_benchmark_results.xlsx"
        self.test_duration = 120  # 2 minutes in seconds
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Fetch all available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def display_models(self, models: List[Dict[str, Any]]) -> int:
        """Display available models and get user selection"""
        if not models:
            print("No models found!")
            return -1
        
        print("\nAvailable Ollama Models:")
        print("-" * 50)
        for i, model in enumerate(models):
            name = model.get('name', 'Unknown')
            size = model.get('size', 0)
            size_gb = size / (1024**3) if size > 0 else 0
            print(f"{i + 1}. {name} ({size_gb:.2f} GB)")
        
        while True:
            try:
                choice = int(input(f"\nSelect a model (1-{len(models)}): ")) - 1
                if 0 <= choice < len(models):
                    return choice
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def get_system_info(self) -> Dict[str, str]:
        """Get system information including GPU details"""
        info = {
            'software': 'Ollama',
            'gpu': 'N/A',
            'driver_version': 'N/A',
            'cuda_version': 'N/A'
        }
        
        try:
            # Get GPU information using nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip()
                    if gpu_name:
                        info['gpu'] = gpu_name
            except:
                pass
            
            # Get NVIDIA driver version
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    info['driver_version'] = result.stdout.strip()
            except:
                pass
            
            # Get CUDA version
            try:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'release' in line.lower():
                            parts = line.split('release')
                            if len(parts) > 1:
                                info['cuda_version'] = parts[1].split(',')[0].strip()
                            break
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Could not get complete system info: {e}")
        
        return info
    
    def monitor_vram_usage(self, duration: int) -> float:
        """Monitor VRAM usage during benchmark"""
        max_vram = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Use nvidia-smi to get VRAM usage
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    vram_str = result.stdout.strip()
                    if vram_str:
                        current_vram = float(vram_str)
                        max_vram = max(max_vram, current_vram)
                time.sleep(1)
            except:
                break
        
        return max_vram
    
    def run_benchmark(self, model_name: str) -> Dict[str, Any]:
        """Run benchmark on selected model"""
        print(f"\nStarting benchmark for model: {model_name}")
        print(f"Duration: {self.test_duration} seconds")
        print("=" * 50)
        
        # Test prompt for consistent benchmarking
        test_prompt = """Write a detailed explanation of machine learning, including its applications, 
        benefits, and challenges. Discuss different types of machine learning algorithms and provide 
        examples of how they are used in real-world scenarios. Make the response comprehensive and 
        educational."""
        
        # Start VRAM monitoring in separate thread
        vram_usage = 0
        vram_thread = None
        try:
            vram_thread = threading.Thread(target=lambda: setattr(self, '_max_vram', self.monitor_vram_usage(self.test_duration)))
            vram_thread.daemon = True
            vram_thread.start()
        except:
            pass
        
        tokens_generated = 0
        total_time = 0
        start_time = time.time()
        request_count = 0
        
        print("Benchmarking in progress...")
        print("Progress: ", end="", flush=True)
        
        while time.time() - start_time < self.test_duration:
            try:
                request_start = time.time()
                
                # Make request to Ollama
                payload = {
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100  # Limit tokens per request for consistent testing
                    }
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                
                request_end = time.time()
                request_time = request_end - request_start
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                    estimated_tokens = len(response_text) // 4
                    tokens_generated += estimated_tokens
                    total_time += request_time
                    request_count += 1
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    progress = int((elapsed / self.test_duration) * 20)
                    print(f"\rProgress: {'█' * progress}{'░' * (20 - progress)} {elapsed:.1f}s", end="", flush=True)
                else:
                    print(f"\nRequest failed with status: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print("\nRequest timed out, continuing...")
                continue
            except Exception as e:
                print(f"\nError during benchmark: {e}")
                continue
        
        print("\nBenchmark completed!")
        
        # Wait for VRAM monitoring to complete
        if vram_thread:
            vram_thread.join(timeout=5)
            vram_usage = getattr(self, '_max_vram', 0)
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        throughput = tokens_generated / actual_duration if actual_duration > 0 else 0
        avg_latency = (total_time * 1000) / tokens_generated if tokens_generated > 0 else 0
        
        return {
            'tokens_generated': tokens_generated,
            'total_time': actual_duration,
            'throughput': throughput,
            'latency': avg_latency,
            'vram_usage': vram_usage,
            'request_count': request_count
        }
    
    def extract_quantization_info(self, model_name: str) -> str:
        """Extract quantization information from model name"""
        model_lower = model_name.lower()
        
        # Common quantization patterns
        if 'q2' in model_lower:
            return 'Q2'
        elif 'q3' in model_lower:
            return 'Q3'
        elif 'q4' in model_lower:
            return 'Q4'
        elif 'q5' in model_lower:
            return 'Q5'
        elif 'q6' in model_lower:
            return 'Q6'
        elif 'q8' in model_lower:
            return 'Q8'
        elif 'fp16' in model_lower or 'f16' in model_lower:
            return 'FP16'
        elif 'fp32' in model_lower or 'f32' in model_lower:
            return 'FP32'
        else:
            return 'Unknown'
    
    def save_results(self, model_name: str, benchmark_results: Dict[str, Any], system_info: Dict[str, str]):
        """Save benchmark results to Excel file"""
        quantization = self.extract_quantization_info(model_name)
        
        # Prepare data for Excel
        data = {
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'LLM Model': [model_name],
            'Quantization': [quantization],
            'Software': [system_info['software']],
            'VRAM Usage': [f"{benchmark_results['vram_usage']:.2f} MB"],
            'Throughput (tokens/s)': [f"{benchmark_results['throughput']:.2f}"],
            'Latency (ms/token)': [f"{benchmark_results['latency']:.2f}"],
            'GPU': [system_info['gpu']],
            'Driver Version': [system_info['driver_version']],
            'CUDA Version': [system_info['cuda_version']],
            'Total Tokens': [benchmark_results['tokens_generated']],
            'Test Duration (s)': [f"{benchmark_results['total_time']:.2f}"],
            'Requests Made': [benchmark_results['request_count']]
        }
        
        new_df = pd.DataFrame(data)
        
        # Load existing data or create new file
        if os.path.exists(self.benchmark_file):
            try:
                existing_df = pd.read_excel(self.benchmark_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not read existing file, creating new one: {e}")
                combined_df = new_df
        else:
            combined_df = new_df
        
        # Save to Excel
        try:
            combined_df.to_excel(self.benchmark_file, index=False)
            print(f"\nResults saved to: {self.benchmark_file}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            # Fallback to CSV
            csv_file = self.benchmark_file.replace('.xlsx', '.csv')
            combined_df.to_csv(csv_file, index=False)
            print(f"Results saved to CSV instead: {csv_file}")
    
    def display_results(self, model_name: str, benchmark_results: Dict[str, Any], system_info: Dict[str, str]):
        """Display benchmark results"""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Quantization: {self.extract_quantization_info(model_name)}")
        print(f"GPU: {system_info['gpu']}")
        print(f"Driver: {system_info['driver_version']}")
        print(f"CUDA: {system_info['cuda_version']}")
        print("-" * 60)
        print(f"Throughput: {benchmark_results['throughput']:.2f} tokens/s")
        print(f"Latency: {benchmark_results['latency']:.2f} ms/token")
        print(f"VRAM Usage: {benchmark_results['vram_usage']:.2f} MB")
        print(f"Total Tokens: {benchmark_results['tokens_generated']}")
        print(f"Test Duration: {benchmark_results['total_time']:.2f} seconds")
        print(f"Requests Made: {benchmark_results['request_count']}")
        print("=" * 60)
    
    def run(self):
        """Main application loop"""
        print("Ollama Benchmark Tool")
        print("=" * 30)
        
        # Check Ollama connection
        if not self.check_ollama_connection():
            print("Error: Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
            return
        
        print("✓ Connected to Ollama")
        
        # Get system information
        system_info = self.get_system_info()
        print(f"✓ System info collected - GPU: {system_info['gpu']}")
        
        # Get available models
        models = self.get_available_models()
        if not models:
            print("No models found. Please install some models first using 'ollama pull <model_name>'")
            return
        
        # Let user select model
        choice = self.display_models(models)
        if choice == -1:
            return
        
        selected_model = models[choice]['name']
        
        # Confirm benchmark
        print(f"\nYou selected: {selected_model}")
        confirm = input("Start benchmark? This will run for 2 minutes (y/n): ").lower().strip()
        if confirm != 'y' and confirm != 'yes':
            print("Benchmark cancelled.")
            return
        
        # Run benchmark
        benchmark_results = self.run_benchmark(selected_model)
        
        # Display and save results
        self.display_results(selected_model, benchmark_results, system_info)
        self.save_results(selected_model, benchmark_results, system_info)

def main():
    """Main entry point"""
    try:
        # Check required packages
        required_packages = ['requests', 'pandas', 'psutil', 'openpyxl']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("Missing required packages. Please install them using:")
            print(f"pip install {' '.join(missing_packages)}")
            return
        
        # Run the benchmark application
        benchmark = OllamaBenchmark()
        benchmark.run()
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
