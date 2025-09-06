# Ollama Benchmark Tool

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

**Comprehensive Ollama Model Benchmarking Suite** - Accurately measure performance, VRAM usage, and throughput for all your local LLM models. Perfect for comparing different models, quantization levels, and hardware configurations.

**Keywords**: Ollama, LLM Benchmarking, AI Model Performance, GPU Benchmarking, VRAM Monitoring, Local AI, Machine Learning Benchmark, Model Comparison, Quantization Testing, Docker Container Benchmarking

## 🚀 Features

- **🔍 Automatic Model Discovery**: Automatically detects and lists all available Ollama models on your system
- **🎯 Interactive Model Selection**: User-friendly interface for selecting specific models to benchmark
- **📊 Batch Benchmarking**: Automatically benchmark all available models with `--all` flag for comprehensive testing
- **⚡ Comprehensive Benchmarking**: Runs intensive 2-minute benchmarks to get accurate performance metrics
- **🖥️ System Information Detection**: Automatically detects GPU, driver versions, and CUDA toolkit
- **📈 Excel Export**: Saves detailed results to Excel with timestamps and all performance metrics
- **📋 Real-time Progress Tracking**: Live progress bars and status updates during benchmarking
- **🐳 Docker-based Model Offloading**: Advanced Docker container restart for accurate VRAM measurements
- **🔧 Command Line Interface**: Multiple modes including interactive, batch, and testing options

## 📋 Performance Metrics Collected

Our advanced benchmarking tool captures comprehensive performance data for thorough analysis:

### Core Performance Metrics
- **🏷️ LLM Model**: Complete model identification (e.g., "llama2:7b", "mistral:7b")
- **🔢 Quantization Level**: Automatic detection of Q2, Q4, Q8, FP16, FP32 quantization
- **💾 VRAM Usage**: Peak memory consumption during benchmark execution
- **⚡ Throughput**: Tokens generated per second (higher = better performance)
- **⏱️ Latency**: Milliseconds per token (lower = faster response)
- **🖥️ GPU Information**: Detailed graphics card specifications
- **🎮 Driver & CUDA**: NVIDIA driver and CUDA toolkit versions

### Advanced Analytics
- **📊 Total Tokens Generated**: Complete token count for workload assessment
- **⏰ Test Duration**: Actual benchmark runtime with precision timing
- **🔄 API Requests**: Request frequency and success rate monitoring
- **📈 Historical Tracking**: Timestamped results for performance trending

**Perfect for**: Model comparison, hardware optimization, quantization analysis, performance monitoring, AI infrastructure planning

## 🔧 Prerequisites & System Requirements

### Required Software
1. **🐳 Ollama Installation**: Local LLM runtime environment must be installed and operational
   - Download from: https://ollama.ai
   - Supports Windows, macOS, Linux platforms
   - Compatible with NVIDIA GPU acceleration

2. **🐍 Python Environment**: Version 3.7 or higher required
   - Core language for benchmark execution
   - Cross-platform compatibility
   - Extensive library ecosystem support

3. **🖥️ NVIDIA GPU** (Highly Recommended): Enhanced performance with GPU acceleration
   - CUDA-compatible graphics cards
   - Automatic GPU detection and utilization
   - VRAM monitoring and optimization

4. **🐳 Docker Environment** (Recommended for Accuracy): Containerized Ollama deployment
   - Isolated environment for consistent testing
   - Automatic model offloading capabilities
   - GPU passthrough support with `--gpus=all`

### Hardware Recommendations
- **RAM**: 16GB+ for large model testing
- **Storage**: SSD recommended for faster model loading
- **GPU VRAM**: 8GB+ for optimal performance
- **CPU**: Multi-core processor for parallel processing

**Supported Platforms**: Windows 10/11, macOS 12+, Ubuntu 18.04+, CentOS 7+, Docker containers

## 🐳 Docker-based Model Offloading (Advanced Feature)

**Revolutionary Approach**: Our Docker-based model offloading ensures the most accurate benchmark results by completely resetting the AI environment between tests.

### Key Benefits
- **🎯 Zero Contamination**: Complete model unloading prevents interference between benchmarks
- **📏 Precise Measurements**: Accurate VRAM usage tracking without cached model data
- **🔄 Automated Process**: No manual intervention required - fully automated workflow
- **⚡ Performance Consistency**: Standardized testing environment for reliable comparisons
- **🛡️ Isolation**: Containerized environment protects system stability

### Docker Setup Instructions

1. **Launch Ollama Container with GPU Support**:
```bash
docker run -d --gpus=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  --restart unless-stopped \
  ollama/ollama
```

2. **Verify Container Status**:
```bash
docker ps | grep ollama
docker logs ollama
```

3. **Test Connection**:
```bash
curl http://localhost:11434/api/version
```

### Advanced Docker Configuration
- **Persistent Storage**: `-v ollama:/root/.ollama` maintains model library
- **GPU Acceleration**: `--gpus=all` enables full GPU utilization
- **Port Mapping**: `-p 11434:11434` exposes Ollama API
- **Auto-restart**: `--restart unless-stopped` ensures high availability

**Pro Tip**: Docker mode provides the most accurate and reproducible benchmark results for serious AI performance analysis.

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

## 🎯 Use Cases & Applications

### 🤖 **AI Model Selection & Optimization**
- Compare performance across different model architectures
- Evaluate quantization impact on speed vs accuracy
- Optimize model selection for specific use cases
- Hardware resource planning and allocation

### 📈 **Performance Monitoring & Trending**
- Track model performance over time
- Identify performance regressions
- Monitor hardware utilization patterns
- Establish performance baselines

### 🏗️ **Infrastructure Planning**
- GPU memory requirements assessment
- CPU utilization analysis
- Network bandwidth planning
- Scalability testing and validation

### 🔬 **Research & Development**
- Model benchmarking for academic research
- Comparative analysis of LLM architectures
- Hardware acceleration optimization
- Performance characterization studies

### 🏢 **Enterprise Applications**
- AI infrastructure capacity planning
- Cost optimization for cloud deployments
- Performance SLA monitoring
- Multi-model deployment strategies

### 🎓 **Educational & Training**
- Hands-on AI performance analysis
- Understanding LLM resource requirements
- Hardware-software interaction studies
- Performance optimization techniques

**Industries**: Machine Learning, AI Research, Cloud Computing, Enterprise IT, Academic Research, Hardware Development

## 💻 Command Line Options & Usage Modes

Our flexible command-line interface supports multiple benchmarking workflows:

### 🚀 **Normal Interactive Mode** (Default)
```bash
python ollama_benchmark.py
```
**Best for**: Single model testing, exploratory analysis, manual model selection
- Interactive model browser with size information
- Step-by-step confirmation process
- Detailed result display and analysis

### 📊 **Batch Mode - All Models** (Automated)
```bash
python ollama_benchmark.py --all
```
**Best for**: Comprehensive model comparison, regression testing, automated benchmarking
- Automatically discovers all available models
- Sequential processing with progress tracking `[current/total]`
- Unattended operation for large model collections
- Graceful error handling and recovery

### 🔧 **VRAM Testing Mode** (Diagnostic)
```bash
python ollama_benchmark.py --test-vram
```
**Best for**: Hardware validation, GPU monitoring setup, troubleshooting
- Tests VRAM monitoring functionality
- Validates GPU detection and drivers
- Multi-GPU system diagnostics
- Performance baseline establishment

### ⚙️ **Advanced Usage Examples**

**Custom Output Location**:
```bash
python ollama_benchmark.py --all  # Results saved to ollama_benchmark_results.xlsx
```

**Integration with CI/CD**:
```bash
# Automated nightly benchmarking
python ollama_benchmark.py --all
```

**Performance Regression Testing**:
```bash
# Compare model performance over time
python ollama_benchmark.py --all
```

**Hardware Performance Analysis**:
```bash
# Test different GPU configurations
python ollama_benchmark.py --all
```

### Single Model Benchmark (Interactive)
1. **Run the application**:
```bash
python ollama_benchmark.py
```

2. **Model Offloading**: If using Docker, the app will automatically restart the Ollama container to unload previous models for accurate measurements.

3. **Select a model**: The app will display all available models. Choose one by entering its number.

4. **Confirm benchmark**: The app will ask for confirmation before starting the 2-minute benchmark.

5. **Wait for completion**: The benchmark will run for 2 minutes, showing progress.

6. **View results**: Results will be displayed on screen and automatically saved to `ollama_benchmark_results.xlsx`.

### Batch Benchmark All Models (Automatic)
1. **Run batch mode**:
```bash
python ollama_benchmark.py --all
```

2. **Automatic execution**: The tool will automatically benchmark all available models one by one.

3. **Progress tracking**: Shows progress for each model with format `[current/total]`.

4. **Results**: All results are saved to the Excel file with timestamps.

5. **Interruption**: You can interrupt the batch process at any time with Ctrl+C.

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

## 🤝 Contributing & Community

We welcome contributions from the AI and machine learning community!

### Ways to Contribute
- 🐛 **Bug Reports**: Found an issue? [Open an issue](https://github.com/ranjanjyoti152/LLM-BECHMARK/issues)
- 💡 **Feature Requests**: Have ideas for new features? [Share them](https://github.com/ranjanjyoti152/LLM-BECHMARK/issues)
- 🔧 **Code Contributions**: Submit pull requests to improve the tool
- 📖 **Documentation**: Help improve documentation and tutorials
- 🧪 **Testing**: Test on different hardware configurations

### Development Setup
```bash
git clone https://github.com/ranjanjyoti152/LLM-BECHMARK.git
cd LLM-BECHMARK
pip install -r requirements.txt
python ollama_benchmark.py --test-vram
```

### Community Guidelines
- Be respectful and inclusive
- Provide detailed bug reports with system information
- Test changes thoroughly before submitting
- Follow Python best practices and PEP 8 style

### Support & Discussion
- 📧 **Issues**: [GitHub Issues](https://github.com/ranjanjyoti152/LLM-BECHMARK/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ranjanjyoti152/LLM-BECHMARK/discussions)
- 📧 **Email**: For security issues or private matters

**Star this repository** ⭐ if you find it useful for your AI benchmarking needs!

---

## 📄 License & Attribution

**MIT License** - Open source and free to use for personal and commercial projects.

```text
Copyright (c) 2025 LLM-Benchmark Tool

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### Acknowledgments
- **Ollama**: For providing the local LLM runtime environment
- **Python Community**: For the excellent libraries and ecosystem
- **Open Source Contributors**: For making AI accessible to everyone

### Related Projects
- [Ollama Official](https://github.com/jmorganca/ollama) - Main Ollama repository
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [Hugging Face](https://github.com/huggingface) - Open source AI models

---

**Keywords**: AI Benchmarking, LLM Performance, GPU Computing, Model Comparison, Quantization Analysis, Docker Containers, Python Tools, Machine Learning, Artificial Intelligence, Local AI, Performance Testing, Hardware Optimization, VRAM Monitoring, Throughput Analysis, Latency Measurement
