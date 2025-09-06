# Ollama Benchmark Tool

A Python application to benchmark Ollama models on your local machine and save results to an Excel file.

## Features

- **Automatic Model Discovery**: Fetches all available models from your local Ollama installation
- **Interactive Model Selection**: Choose which model to benchmark from a list
- **Comprehensive Benchmarking**: Runs for 2 minutes at full potential
- **System Information**: Automatically detects GPU, driver, and CUDA versions
- **Excel Export**: Saves results with all required metrics to an Excel file
- **Progress Tracking**: Shows real-time progress during benchmarking
- **Docker-based Model Offloading**: Automatically unloads previous models for accurate VRAM measurements

## Metrics Collected

The tool collects and saves the following metrics:

- **LLM Model**: Name of the model being tested
- **Quantization**: Detected quantization level (Q2, Q4, Q8, FP16, etc.)
- **Software**: Always "Ollama"
- **VRAM Usage**: Maximum VRAM usage during benchmark (MB)
- **Throughput**: Tokens generated per second
- **Latency**: Milliseconds per token
- **GPU**: Graphics card information
- **Driver Version**: NVIDIA driver version
- **CUDA Version**: CUDA toolkit version

## Prerequisites

1. **Ollama installed and running**: Make sure Ollama is installed and running on your system
2. **Python 3.7+**: Required for the application
3. **NVIDIA GPU** (optional): For GPU benchmarking and VRAM monitoring
4. **Docker** (recommended): For accurate model offloading and VRAM measurements

## Docker-based Model Offloading

For the most accurate benchmark results, the tool now uses Docker to completely unload all models before each benchmark:

- **Automatic Container Restart**: Stops and restarts the Ollama Docker container
- **Clean Memory State**: Ensures no previous models interfere with VRAM measurements
- **20-Second Warm-up**: Waits for Ollama to fully initialize after restart
- **Accurate Results**: Provides the most precise VRAM usage and performance metrics

### Docker Setup (Recommended)

1. **Run Ollama in Docker**:
```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

2. **Verify container is running**:
```bash
docker ps | grep ollama
```

The benchmark tool will automatically handle model offloading when Docker is used.

## Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install requests pandas GPUtil psutil openpyxl
```

2. Make sure Ollama is running:
```bash
ollama serve
```

3. Ensure you have some models installed:
```bash
ollama pull llama2
ollama pull mistral
# etc.
```

## Usage

1. **Run the application**:
```bash
python ollama_benchmark.py
```

2. **Model Offloading**: If using Docker, the app will automatically restart the Ollama container to unload previous models for accurate measurements.

3. **Select a model**: The app will display all available models. Choose one by entering its number.

4. **Confirm benchmark**: The app will ask for confirmation before starting the 2-minute benchmark.

5. **Wait for completion**: The benchmark will run for 2 minutes, showing progress.

6. **View results**: Results will be displayed on screen and automatically saved to `ollama_benchmark_results.xlsx`.

## Output

The application creates/updates an Excel file `ollama_benchmark_results.xlsx` with the following columns:

| Column | Description |
|--------|-------------|
| Timestamp | When the benchmark was run |
| LLM Model | Model name (e.g., "llama2:7b") |
| Quantization | Detected quantization (Q4, Q8, etc.) |
| Software | Always "Ollama" |
| VRAM Usage | Peak VRAM usage in MB |
| Throughput (tokens/s) | Tokens generated per second |
| Latency (ms/token) | Milliseconds per token |
| GPU | GPU model name |
| Driver Version | NVIDIA driver version |
| CUDA Version | CUDA toolkit version |
| Total Tokens | Total tokens generated during test |
| Test Duration (s) | Actual test duration |
| Requests Made | Number of API requests made |

## Example Output

```
Ollama Benchmark Tool
==============================
✓ Connected to Ollama
✓ System info collected - GPU: NVIDIA GeForce RTX 4090

Available Ollama Models:
--------------------------------------------------
1. llama2:7b (3.83 GB)
2. mistral:7b (4.11 GB)
3. codellama:13b (7.37 GB)

Select a model (1-3): 1

You selected: llama2:7b
Start benchmark? This will run for 2 minutes (y/n): y

Starting benchmark for model: llama2:7b
Duration: 120 seconds
==================================================
Benchmarking in progress...
Progress: ████████████████████ 120.0s
Benchmark completed!

============================================================
BENCHMARK RESULTS
============================================================
Model: llama2:7b
Quantization: Q4
GPU: NVIDIA GeForce RTX 4090
Driver: 535.129.03
CUDA: 12.2
------------------------------------------------------------
Throughput: 45.23 tokens/s
Latency: 22.11 ms/token
VRAM Usage: 4567.89 MB
Total Tokens: 5428
Test Duration: 120.05 seconds
Requests Made: 54
============================================================

Results saved to: ollama_benchmark_results.xlsx
```

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if accessible: `curl http://localhost:11434/api/version`

### Missing System Information
- For NVIDIA info, ensure drivers are installed
- For CUDA version, ensure CUDA toolkit is installed

### Permission Issues
- Make sure you have write permissions in the directory
- The app will fallback to CSV if Excel writing fails

### Docker Issues
- Ensure Ollama container is named 'ollama': `docker ps`
- Check Docker permissions: `docker --version`
- For GPU support, ensure `--gpus=all` flag is used when starting container
- Model offloading will be skipped if Docker commands fail

### No Models Found
- Install models: `ollama pull <model_name>`
- Check available models: `ollama list`

## Notes

- The benchmark uses a consistent prompt for fair comparison across models
- Each request is limited to 100 tokens for consistent testing
- VRAM monitoring requires GPUtil and may not work on all systems
- Results are appended to the Excel file, allowing historical tracking
- The application handles timeouts and errors gracefully
- **Docker offloading**: For most accurate results, run Ollama in Docker - the tool will automatically restart the container before each benchmark
