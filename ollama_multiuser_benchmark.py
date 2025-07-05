#!/usr/bin/env python3
"""
Ollama Multi-User Benchmark Tool
This script simulates multiple concurrent users benchmarking Ollama models and saves results to an Excel file.
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
import concurrent.futures
import queue
import random

class OllamaMultiUserBenchmark:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.benchmark_file = "ollama_multiuser_benchmark_results.xlsx"
        self.test_duration = 120  # 2 minutes in seconds
        self.max_concurrent_users = 5000  # Maximum users to test
        self.results_queue = queue.Queue()
        
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
    
    def get_concurrent_user_count(self) -> int:
        """Get the number of concurrent users to simulate"""
        print(f"\nConcurrent User Configuration:")
        print(f"Available range: 1-{self.max_concurrent_users}")
        print("\nRecommended starting points:")
        print("  • Light load: 1-10 users")
        print("  • Medium load: 10-50 users") 
        print("  • Heavy load: 50-200 users")
        print("  • Stress test: 200-1000 users")
        print("  • Extreme test: 1000+ users")
        print("\nWarning: High user counts (>500) may consume significant system resources!")
        
        while True:
            try:
                count = int(input(f"\nEnter number of concurrent users (1-{self.max_concurrent_users}): "))
                if 1 <= count <= self.max_concurrent_users:
                    # Add warning for high user counts
                    if count > 500:
                        print(f"\n⚠️  WARNING: Testing with {count} concurrent users!")
                        print("This may:")
                        print("  • Consume significant CPU and memory")
                        print("  • Create many network connections")
                        print("  • Potentially impact system stability")
                        confirm = input("Are you sure you want to proceed? (y/n): ").lower().strip()
                        if confirm != 'y' and confirm != 'yes':
                            continue
                    return count
                else:
                    print(f"Please enter a number between 1 and {self.max_concurrent_users}.")
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
    
    def monitor_system_resources(self, duration: int) -> Dict[str, float]:
        """Monitor system resources during benchmark"""
        max_vram = 0
        max_cpu = 0
        max_memory = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Monitor VRAM
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    vram_str = result.stdout.strip()
                    if vram_str:
                        current_vram = float(vram_str)
                        max_vram = max(max_vram, current_vram)
                
                # Monitor CPU and RAM
                current_cpu = psutil.cpu_percent(interval=0.1)
                current_memory = psutil.virtual_memory().percent
                max_cpu = max(max_cpu, current_cpu)
                max_memory = max(max_memory, current_memory)
                
                time.sleep(1)
            except:
                break
        
        return {
            'max_vram': max_vram,
            'max_cpu': max_cpu,
            'max_memory': max_memory
        }
    
    def simulate_user(self, user_id: int, model_name: str, duration: int) -> Dict[str, Any]:
        """Simulate a single user making requests"""
        # Varied prompts to simulate different user behaviors
        prompts = [
            "Explain artificial intelligence in simple terms.",
            "Write a short story about a robot.",
            "What are the benefits of machine learning?",
            "Describe how neural networks work.",
            "List 5 programming languages and their uses.",
            "Explain the difference between AI and ML.",
            "What is deep learning?",
            "How does computer vision work?",
            "Describe natural language processing.",
            "What are the ethical concerns with AI?"
        ]
        
        user_stats = {
            'user_id': user_id,
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_response_time': 0,
            'min_response_time': float('inf'),
            'max_response_time': 0,
            'timeouts': 0,
            'errors': 0
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Select random prompt to simulate varied user behavior
                prompt = random.choice(prompts)
                request_start = time.time()
                
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": random.uniform(0.3, 0.9),  # Vary temperature
                        "num_predict": random.randint(50, 150)   # Vary response length
                    }
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=20  # Shorter timeout for multi-user scenario
                )
                
                request_end = time.time()
                response_time = request_end - request_start
                
                user_stats['requests_made'] += 1
                user_stats['total_response_time'] += response_time
                user_stats['min_response_time'] = min(user_stats['min_response_time'], response_time)
                user_stats['max_response_time'] = max(user_stats['max_response_time'], response_time)
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    estimated_tokens = len(response_text) // 4
                    user_stats['total_tokens'] += estimated_tokens
                    user_stats['successful_requests'] += 1
                else:
                    user_stats['failed_requests'] += 1
                    
                # Add small random delay to simulate human behavior
                time.sleep(random.uniform(0.5, 2.0))
                    
            except requests.exceptions.Timeout:
                user_stats['timeouts'] += 1
                user_stats['failed_requests'] += 1
                continue
            except Exception as e:
                user_stats['errors'] += 1
                user_stats['failed_requests'] += 1
                continue
        
        # Fix min_response_time if no requests were successful
        if user_stats['min_response_time'] == float('inf'):
            user_stats['min_response_time'] = 0
            
        return user_stats
    
    def run_multiuser_benchmark(self, model_name: str, concurrent_users: int) -> Dict[str, Any]:
        """Run benchmark with multiple concurrent users"""
        print(f"\nStarting multi-user benchmark for model: {model_name}")
        print(f"Concurrent users: {concurrent_users}")
        print(f"Duration: {self.test_duration} seconds")
        print("=" * 60)
        
        # Optimize thread pool size for high concurrent loads
        # Use reasonable thread pool size to avoid overwhelming the system
        max_workers = min(concurrent_users, 500)  # Cap at 500 threads
        if concurrent_users > 500:
            print(f"Note: Using {max_workers} threads to manage {concurrent_users} users")
        
        # Start system monitoring in separate thread
        system_resources = {}
        monitor_thread = None
        try:
            monitor_thread = threading.Thread(
                target=lambda: setattr(self, '_system_resources', 
                                     self.monitor_system_resources(self.test_duration))
            )
            monitor_thread.daemon = True
            monitor_thread.start()
        except:
            pass
        
        start_time = time.time()
        
        # Create thread pool for concurrent users with optimized worker count
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            print("Starting concurrent user simulation...")
            if concurrent_users > 100:
                print("High load test - monitoring system resources...")
            print("Progress: ", end="", flush=True)
            
            # Submit user simulation tasks in batches for very high user counts
            futures = []
            batch_size = 100  # Process users in batches to avoid memory issues
            
            for batch_start in range(0, concurrent_users, batch_size):
                batch_end = min(batch_start + batch_size, concurrent_users)
                batch_futures = []
                
                for user_id in range(batch_start + 1, batch_end + 1):
                    future = executor.submit(self.simulate_user, user_id, model_name, self.test_duration)
                    batch_futures.append(future)
                
                futures.extend(batch_futures)
                
                # Small delay between batches for very high user counts to avoid overwhelming
                if concurrent_users > 1000 and batch_end < concurrent_users:
                    time.sleep(0.1)
            
            # Monitor progress with better formatting for high user counts
            progress_update_interval = 2 if concurrent_users > 1000 else 1
            last_update = 0
            
            while time.time() - start_time < self.test_duration:
                elapsed = time.time() - start_time
                
                # Update progress display less frequently for high loads
                if elapsed - last_update >= progress_update_interval:
                    progress = int((elapsed / self.test_duration) * 30)
                    remaining_time = self.test_duration - elapsed
                    
                    # Show additional info for high loads
                    if concurrent_users > 100:
                        completed_futures = sum(1 for f in futures if f.done())
                        print(f"\rProgress: {'█' * progress}{'░' * (30 - progress)} {elapsed:.1f}s | Users: {concurrent_users} | Completed: {completed_futures}", end="", flush=True)
                    else:
                        print(f"\rProgress: {'█' * progress}{'░' * (30 - progress)} {elapsed:.1f}s (Remaining: {remaining_time:.1f}s)", end="", flush=True)
                    
                    last_update = elapsed
                
                time.sleep(0.5)
            
            print("\nCollecting results from all users...")
            
            # Collect results with timeout to handle potential hangs
            user_results = []
            timeout_per_result = 5.0  # 5 seconds per result
            
            for i, future in enumerate(futures):
                try:
                    if concurrent_users > 1000 and i % 100 == 0:
                        print(f"Collecting results: {i+1}/{len(futures)}")
                    
                    user_stats = future.result(timeout=timeout_per_result)
                    user_results.append(user_stats)
                except concurrent.futures.TimeoutError:
                    print(f"\nTimeout collecting result from user {i+1}")
                    # Create dummy result for timed out user
                    user_results.append({
                        'user_id': i+1,
                        'requests_made': 0,
                        'successful_requests': 0,
                        'failed_requests': 0,
                        'total_tokens': 0,
                        'total_response_time': 0,
                        'min_response_time': 0,
                        'max_response_time': 0,
                        'timeouts': 1,
                        'errors': 0
                    })
                except Exception as e:
                    print(f"\nError collecting user {i+1} result: {e}")
                    # Create dummy result for errored user
                    user_results.append({
                        'user_id': i+1,
                        'requests_made': 0,
                        'successful_requests': 0,
                        'failed_requests': 1,
                        'total_tokens': 0,
                        'total_response_time': 0,
                        'min_response_time': 0,
                        'max_response_time': 0,
                        'timeouts': 0,
                        'errors': 1
                    })
        
        print(f"\nMulti-user benchmark completed! Processed {len(user_results)} users.")
        
        # Wait for system monitoring to complete
        if monitor_thread:
            monitor_thread.join(timeout=5)
            system_resources = getattr(self, '_system_resources', {})
        
        # Calculate aggregate metrics
        total_requests = sum(user['requests_made'] for user in user_results)
        total_successful = sum(user['successful_requests'] for user in user_results)
        total_failed = sum(user['failed_requests'] for user in user_results)
        total_tokens = sum(user['total_tokens'] for user in user_results)
        total_timeouts = sum(user['timeouts'] for user in user_results)
        total_errors = sum(user['errors'] for user in user_results)
        
        actual_duration = time.time() - start_time
        
        # Calculate throughput and success rate
        aggregate_throughput = total_tokens / actual_duration if actual_duration > 0 else 0
        success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate average response times (only for users with requests)
        valid_users = [user for user in user_results if user['requests_made'] > 0]
        all_response_times = [user['total_response_time'] / user['requests_made'] for user in valid_users]
        avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
        
        # Calculate user handling potential (successful requests per second per user)
        user_handling_potential = (total_successful / actual_duration / concurrent_users) if actual_duration > 0 and concurrent_users > 0 else 0
        
        # Calculate additional metrics for high-load scenarios
        active_users = len(valid_users)  # Users who made at least one request
        active_user_rate = (active_users / concurrent_users * 100) if concurrent_users > 0 else 0
        
        return {
            'user_results': user_results,
            'aggregate_stats': {
                'concurrent_users': concurrent_users,
                'active_users': active_users,
                'active_user_rate': active_user_rate,
                'total_requests': total_requests,
                'successful_requests': total_successful,
                'failed_requests': total_failed,
                'total_tokens': total_tokens,
                'total_timeouts': total_timeouts,
                'total_errors': total_errors,
                'actual_duration': actual_duration,
                'aggregate_throughput': aggregate_throughput,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'user_handling_potential': user_handling_potential,
                'system_resources': system_resources
            }
        }
    
    def test_excel_functionality(self):
        """Test if Excel file operations work correctly"""
        test_file = "test_multiuser_benchmark.xlsx"
        try:
            # Create test data
            test_data = pd.DataFrame({'Test': ['Value1', 'Value2'], 'Number': [1, 2]})
            
            # Try writing
            with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
                test_data.to_excel(writer, sheet_name='Test', index=False)
            
            # Try reading
            read_data = pd.read_excel(test_file, sheet_name='Test')
            
            # Clean up
            os.remove(test_file)
            
            print("✓ Excel functionality test passed")
            return True
            
        except Exception as e:
            print(f"✗ Excel functionality test failed: {e}")
            if os.path.exists(test_file):
                try:
                    os.remove(test_file)
                except:
                    pass
            return False
    
    def backup_existing_file(self):
        """Create a backup of existing Excel file"""
        if os.path.exists(self.benchmark_file):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.benchmark_file.replace('.xlsx', f'_backup_{timestamp}.xlsx')
                
                # Copy the file
                import shutil
                shutil.copy2(self.benchmark_file, backup_file)
                print(f"Backup created: {backup_file}")
                return backup_file
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
                return None
        return None
    
    def extract_quantization_info(self, model_name: str) -> str:
        """Extract quantization information from model name"""
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
        """Save multi-user benchmark results to Excel file"""
        quantization = self.extract_quantization_info(model_name)
        aggregate = benchmark_results['aggregate_stats']
        resources = aggregate.get('system_resources', {})
        
        # Prepare main summary data
        summary_data = {
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'LLM Model': [model_name],
            'Quantization': [quantization],
            'Software': [system_info['software']],
            'Concurrent Users': [aggregate['concurrent_users']],
            'Active Users': [aggregate['active_users']],
            'Active User Rate (%)': [f"{aggregate['active_user_rate']:.2f}"],
            'Total Requests': [aggregate['total_requests']],
            'Successful Requests': [aggregate['successful_requests']],
            'Failed Requests': [aggregate['failed_requests']],
            'Success Rate (%)': [f"{aggregate['success_rate']:.2f}"],
            'Total Tokens': [aggregate['total_tokens']],
            'Aggregate Throughput (tokens/s)': [f"{aggregate['aggregate_throughput']:.2f}"],
            'Avg Response Time (s)': [f"{aggregate['avg_response_time']:.2f}"],
            'User Handling Potential (req/s/user)': [f"{aggregate['user_handling_potential']:.3f}"],
            'Timeouts': [aggregate['total_timeouts']],
            'Errors': [aggregate['total_errors']],
            'Test Duration (s)': [f"{aggregate['actual_duration']:.2f}"],
            'Max VRAM Usage (MB)': [f"{resources.get('max_vram', 0):.2f}"],
            'Max CPU Usage (%)': [f"{resources.get('max_cpu', 0):.1f}"],
            'Max Memory Usage (%)': [f"{resources.get('max_memory', 0):.1f}"],
            'GPU': [system_info['gpu']],
            'Driver Version': [system_info['driver_version']],
            'CUDA Version': [system_info['cuda_version']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Prepare detailed user data (limit for very high user counts to avoid memory issues)
        user_data = []
        max_users_to_save = 1000  # Limit detailed user data to prevent Excel issues
        
        users_to_save = benchmark_results['user_results']
        if len(users_to_save) > max_users_to_save:
            print(f"Note: Saving detailed data for first {max_users_to_save} users to prevent file size issues")
            users_to_save = users_to_save[:max_users_to_save]
        
        for user in users_to_save:
            user_data.append({
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Model': model_name,
                'User ID': user['user_id'],
                'Requests Made': user['requests_made'],
                'Successful Requests': user['successful_requests'],
                'Failed Requests': user['failed_requests'],
                'Total Tokens': user['total_tokens'],
                'Avg Response Time (s)': f"{user['total_response_time'] / user['requests_made']:.2f}" if user['requests_made'] > 0 else "0.00",
                'Min Response Time (s)': f"{user['min_response_time']:.2f}",
                'Max Response Time (s)': f"{user['max_response_time']:.2f}",
                'Timeouts': user['timeouts'],
                'Errors': user['errors']
            })
        
        user_df = pd.DataFrame(user_data)
        
        # Save to Excel with multiple sheets
        try:
            # Create backup of existing file
            backup_file = self.backup_existing_file()
            
            # Load existing data if file exists
            combined_summary = summary_df
            combined_users = user_df
            
            if os.path.exists(self.benchmark_file):
                try:
                    # Read existing sheets
                    existing_summary = pd.read_excel(self.benchmark_file, sheet_name='Summary')
                    existing_users = pd.read_excel(self.benchmark_file, sheet_name='User_Details')
                    
                    print(f"Found existing data: {len(existing_summary)} summary records, {len(existing_users)} user records")
                    
                    # Append new data to existing data
                    combined_summary = pd.concat([existing_summary, summary_df], ignore_index=True)
                    combined_users = pd.concat([existing_users, user_df], ignore_index=True)
                    
                    print(f"After combining: {len(combined_summary)} summary records, {len(combined_users)} user records")
                except Exception as e:
                    print(f"Warning: Could not read existing Excel file, creating new one: {e}")
                    # If reading fails, use only new data
                    combined_summary = summary_df
                    combined_users = user_df
            else:
                print("Creating new Excel file with initial data")
            
            # Write to Excel file
            with pd.ExcelWriter(self.benchmark_file, engine='openpyxl', mode='w') as writer:
                # Write to sheets
                combined_summary.to_excel(writer, sheet_name='Summary', index=False)
                combined_users.to_excel(writer, sheet_name='User_Details', index=False)
                
                # Auto-adjust column widths for better readability
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"\nResults successfully saved to: {self.benchmark_file}")
            print(f"  - Summary sheet: {len(combined_summary)} total benchmark runs")
            print(f"  - User_Details sheet: {len(combined_users)} total user records")
            
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            print("Attempting to save as CSV files instead...")
            
            # Fallback to CSV with timestamped filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file_summary = f"multiuser_benchmark_summary_{timestamp}.csv"
            csv_file_users = f"multiuser_benchmark_users_{timestamp}.csv"
            
            try:
                # Try to load and append to existing CSV files
                existing_csv_summary = f"multiuser_benchmark_summary_combined.csv"
                existing_csv_users = f"multiuser_benchmark_users_combined.csv"
                
                if os.path.exists(existing_csv_summary):
                    existing_summary_df = pd.read_csv(existing_csv_summary)
                    combined_summary = pd.concat([existing_summary_df, summary_df], ignore_index=True)
                else:
                    combined_summary = summary_df
                    
                if os.path.exists(existing_csv_users):
                    existing_users_df = pd.read_csv(existing_csv_users)
                    combined_users = pd.concat([existing_users_df, user_df], ignore_index=True)
                else:
                    combined_users = user_df
                
                # Save combined data
                combined_summary.to_csv(existing_csv_summary, index=False)
                combined_users.to_csv(existing_csv_users, index=False)
                
                # Also save timestamped copies
                summary_df.to_csv(csv_file_summary, index=False)
                user_df.to_csv(csv_file_users, index=False)
                
                print(f"Results saved to CSV files:")
                print(f"  - {existing_csv_summary} (combined summary data)")
                print(f"  - {existing_csv_users} (combined user data)")
                print(f"  - {csv_file_summary} (this run's summary)")
                print(f"  - {csv_file_users} (this run's user data)")
                
            except Exception as csv_error:
                print(f"Error saving CSV files: {csv_error}")
                # Last resort: save with simple timestamped names
                summary_df.to_csv(csv_file_summary, index=False)
                user_df.to_csv(csv_file_users, index=False)
                print(f"Results saved to timestamped CSV files:")
                print(f"  - {csv_file_summary}")
                print(f"  - {csv_file_users}")
    
    def display_results(self, model_name: str, benchmark_results: Dict[str, Any], system_info: Dict[str, str]):
        """Display multi-user benchmark results"""
        aggregate = benchmark_results['aggregate_stats']
        resources = aggregate.get('system_resources', {})
        
        print("\n" + "=" * 80)
        print("MULTI-USER BENCHMARK RESULTS")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Quantization: {self.extract_quantization_info(model_name)}")
        print(f"GPU: {system_info['gpu']}")
        print(f"Driver: {system_info['driver_version']}")
        print(f"CUDA: {system_info['cuda_version']}")
        print("-" * 80)
        print("CONCURRENT USER PERFORMANCE:")
        print(f"Concurrent Users: {aggregate['concurrent_users']}")
        print(f"Active Users: {aggregate['active_users']} ({aggregate['active_user_rate']:.1f}%)")
        print(f"Total Requests: {aggregate['total_requests']}")
        print(f"Successful Requests: {aggregate['successful_requests']}")
        print(f"Failed Requests: {aggregate['failed_requests']}")
        print(f"Success Rate: {aggregate['success_rate']:.2f}%")
        print(f"Timeouts: {aggregate['total_timeouts']}")
        print(f"Errors: {aggregate['total_errors']}")
        print("-" * 80)
        print("PERFORMANCE METRICS:")
        print(f"User Handling Potential: {aggregate['user_handling_potential']:.3f} req/s/user")
        print(f"Aggregate Throughput: {aggregate['aggregate_throughput']:.2f} tokens/s")
        print(f"Average Response Time: {aggregate['avg_response_time']:.2f} seconds")
        print(f"Total Tokens Generated: {aggregate['total_tokens']}")
        print(f"Test Duration: {aggregate['actual_duration']:.2f} seconds")
        print("-" * 80)
        print("SYSTEM RESOURCE USAGE:")
        print(f"Max VRAM Usage: {resources.get('max_vram', 0):.2f} MB")
        print(f"Max CPU Usage: {resources.get('max_cpu', 0):.1f}%")
        print(f"Max Memory Usage: {resources.get('max_memory', 0):.1f}%")
        print("=" * 80)
    
    def run(self):
        """Main application loop"""
        print("Ollama Multi-User Benchmark Tool")
        print("=" * 40)
        
        # Test Excel functionality first
        if not self.test_excel_functionality():
            print("Warning: Excel functionality issues detected. Results will be saved as CSV.")
        
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
        
        # Get concurrent user count
        concurrent_users = self.get_concurrent_user_count()
        
        # Confirm benchmark
        print(f"\nConfiguration Summary:")
        print(f"Model: {selected_model}")
        print(f"Concurrent Users: {concurrent_users}")
        print(f"Duration: {self.test_duration} seconds")
        print("\nThis will simulate multiple users making concurrent requests to test user handling capacity.")
        
        confirm = input("\nStart multi-user benchmark? (y/n): ").lower().strip()
        if confirm != 'y' and confirm != 'yes':
            print("Benchmark cancelled.")
            return
        
        # Run benchmark
        benchmark_results = self.run_multiuser_benchmark(selected_model, concurrent_users)
        
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
        
        # Run the multi-user benchmark application
        benchmark = OllamaMultiUserBenchmark()
        benchmark.run()
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
