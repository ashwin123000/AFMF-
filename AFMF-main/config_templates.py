"""
BENCHMARKING CONFIGURATION TEMPLATE
===================================

Copy this file and customize it for your specific benchmark runs.
Use it with: python benchmark_baseline.py --config your_config.py
"""

from benchmark_baseline import BenchmarkConfig


# ============================================================================
# CONFIGURATION TEMPLATE 1: Quick Benchmark
# ============================================================================

class QuickBenchmarkConfig(BenchmarkConfig):
    """Fast benchmark for quick testing"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 64
    NUM_ITERATIONS = 5              # Few iterations for speed
    WARMUP_ITERATIONS = 1
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/quick/"


# ============================================================================
# CONFIGURATION TEMPLATE 2: Thorough Benchmark
# ============================================================================

class ThoroughBenchmarkConfig(BenchmarkConfig):
    """Comprehensive benchmark with many measurements"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 32                 # Smaller batch for stability
    NUM_ITERATIONS = 50             # Many iterations for stability
    WARMUP_ITERATIONS = 5           # More warmup
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/thorough/"


# ============================================================================
# CONFIGURATION TEMPLATE 3: CPU Benchmark
# ============================================================================

class CPUBenchmarkConfig(BenchmarkConfig):
    """Benchmark on CPU only"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cpu"                  # Force CPU
    BATCH_SIZE = 16                 # Smaller batch for CPU
    NUM_ITERATIONS = 10
    WARMUP_ITERATIONS = 2
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/cpu/"


# ============================================================================
# CONFIGURATION TEMPLATE 4: GPU Benchmark
# ============================================================================

class GPUBenchmarkConfig(BenchmarkConfig):
    """Optimized for GPU execution"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 128                # Larger batch for GPU
    NUM_ITERATIONS = 20
    WARMUP_ITERATIONS = 3
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/gpu/"


# ============================================================================
# CONFIGURATION TEMPLATE 5: Model Comparison
# ============================================================================

class ModelComparisonConfig(BenchmarkConfig):
    """Consistent settings for comparing multiple models"""
    
    # Customize these
    MODEL_NAME = "MTAD_GAT"         # Change for each model
    
    # Keep same for all models
    DATASET_NAME = "SMD"
    DEVICE = "cuda"
    BATCH_SIZE = 32
    NUM_ITERATIONS = 10
    WARMUP_ITERATIONS = 2
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/comparison/"


# ============================================================================
# CONFIGURATION TEMPLATE 6: Different Dataset
# ============================================================================

class CustomDatasetConfig(BenchmarkConfig):
    """Template for different dataset"""
    
    DATASET_NAME = "MSL"            # Change dataset
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 32
    NUM_ITERATIONS = 10
    WARMUP_ITERATIONS = 2
    ANOMALY_THRESHOLD = 0.01
    OUTPUT_DIR = "./benchmark_results/msl_dataset/"


# ============================================================================
# CONFIGURATION TEMPLATE 6B: SWaT Benchmark
# ============================================================================

class SWaTBenchmarkConfig(BenchmarkConfig):
    """SWaT-specific benchmark configuration aligned with the paper."""

    DATASET_NAME = "SWaT"
    DATA_PROCESS = True
    PARTIAL_DATA = False
    WINDOW_SIZE = 720
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda" if BenchmarkConfig.DEVICE == "cuda" else "cpu"
    BATCH_SIZE = 128
    NUM_ITERATIONS = 10
    WARMUP_ITERATIONS = 2
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/swat/"


# ============================================================================
# CONFIGURATION TEMPLATE 7: Latency-Focused
# ============================================================================

class LatencyFocusedConfig(BenchmarkConfig):
    """Optimize measurements for latency analysis"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 1                  # Single item for per-request latency
    NUM_ITERATIONS = 100            # Many measurements
    WARMUP_ITERATIONS = 10
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/latency_focus/"


# ============================================================================
# CONFIGURATION TEMPLATE 8: Memory-Focused
# ============================================================================

class MemoryFocusedConfig(BenchmarkConfig):
    """Optimize measurements for memory analysis"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 256                # Large batch for memory testing
    NUM_ITERATIONS = 5
    WARMUP_ITERATIONS = 1
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/memory_focus/"


# ============================================================================
# CONFIGURATION TEMPLATE 9: Production Baseline
# ============================================================================

class ProductionBaselineConfig(BenchmarkConfig):
    """Realistic production settings"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 32                 # Typical production batch
    NUM_ITERATIONS = 20
    WARMUP_ITERATIONS = 5
    ANOMALY_THRESHOLD = 0.005
    OUTPUT_DIR = "./benchmark_results/production/"


# ============================================================================
# CONFIGURATION TEMPLATE 10: Custom Anomaly Threshold Testing
# ============================================================================

class ThresholdTestingConfig(BenchmarkConfig):
    """Test different anomaly detection thresholds"""
    
    DATASET_NAME = "SMD"
    MODEL_NAME = "MTAD_GAT"
    DEVICE = "cuda"
    BATCH_SIZE = 32
    NUM_ITERATIONS = 10
    WARMUP_ITERATIONS = 2
    ANOMALY_THRESHOLD = 0.01         # Adjust threshold ratio (0.1%-1%)
    OUTPUT_DIR = "./benchmark_results/threshold_test/"


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Use a predefined configuration

from benchmark_baseline import BaselineBenchmark
from config_templates import QuickBenchmarkConfig

config = QuickBenchmarkConfig()
benchmark = BaselineBenchmark(config)
results = benchmark.run_benchmark()
benchmark.save_results()
benchmark.print_summary()


Example 2: Create custom configuration

from benchmark_baseline import BenchmarkConfig, BaselineBenchmark

class MyCustomConfig(BenchmarkConfig):
    MODEL_NAME = "DLinear"
    DATASET_NAME = "SWaT"
    DEVICE = "cuda"
    BATCH_SIZE = 64
    NUM_ITERATIONS = 15
    ANOMALY_THRESHOLD = 0.6

config = MyCustomConfig()
benchmark = BaselineBenchmark(config)
results = benchmark.run_benchmark()
benchmark.save_results()


Example 3: Compare configurations

from benchmark_baseline import BaselineBenchmark

configs = [
    QuickBenchmarkConfig(),
    ThoroughBenchmarkConfig(),
    LatencyFocusedConfig()
]

for config in configs:
    print(f"\\nRunning: {config.__class__.__name__}")
    benchmark = BaselineBenchmark(config)
    results = benchmark.run_benchmark()
    benchmark.save_results()
"""

# ============================================================================
# PARAMETER GUIDE
# ============================================================================

"""
DATASET_NAME Options:
  - "SMD": Server Machine Dataset
  - "MSL": Mars Science Laboratory
  - "SMAP": Soil Moisture Active Passive
  - "PSM": Public dataset
  - "SWaT": Secure Water Treatment
  - "WADI": Water Distribution
  - "MBA": Multivariate Benchmark Array
  - "UCR": University of California Riverside
  - "NAB": Numenta Anomaly Benchmark
  - "MSDS": Microsoft Anomaly Dataset

MODEL_NAME Options:
  - "MTAD_GAT": Multi-Task Attention Detection with GAT
  - "DLinear": Deep Linear model
  - "Autoformer": Autoformer from transformers
  - "DeepAR": Deep Autoregressive model
  - "GTA": Gated Temporal Attention
  - "Informer": Informer transformer
  - "LSTNet": Long Short-Term Network
  - "MA": Moving Average baseline
  - "GDN": Graph Deviation Network
  - "RTNet": Recurrent Temporal Network

DEVICE Options:
  - "cuda": NVIDIA GPU (if available)
  - "cpu": CPU only

BATCH_SIZE Guidance:
  - CPU: 8-32 (lower is more stable)
  - GPU: 32-256 (higher for better throughput)
  - Latency testing: 1 (single item)
  - Memory testing: 128-512 (stress test)

NUM_ITERATIONS Guidance:
  - Quick test: 5
  - Standard: 10-20
  - Thorough: 50+
  - (Higher = more stable measurements but slower)

WARMUP_ITERATIONS Guidance:
  - Usually: 1-5 iterations
  - GPU: 3-5 (warmup CUDA)
  - CPU: 1-2 (less needed)

ANOMALY_THRESHOLD Guidance:
  - Conservative (high precision): 0.6-0.8
  - Balanced: 0.4-0.6
  - Sensitive (high recall): 0.1-0.4
  - Depends on your anomaly distribution

OUTPUT_DIR:
  - Use ./benchmark_results/ directory
  - Create subdirectories for organization
  - Ensure directory exists or will be created
"""

print(__doc__)
