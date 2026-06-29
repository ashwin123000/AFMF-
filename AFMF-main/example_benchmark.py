"""
EXAMPLE: Running Baseline Model Benchmarking
============================================

This script demonstrates how to quickly run benchmarking on your baseline model.
Execute this script directly: python example_benchmark.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_baseline import BenchmarkConfig, BaselineBenchmark


def example_1_simple_benchmark():
    """
    Example 1: Simplest usage with default configuration
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Benchmark with Defaults")
    print("="*80)
    
    # Create benchmark with default settings
    config = BenchmarkConfig()
    benchmark = BaselineBenchmark(config)
    
    # Run complete pipeline
    results = benchmark.run_benchmark()
    
    # Save and print results
    benchmark.save_results()
    benchmark.print_summary()
    
    return results


def example_2_custom_model_and_dataset():
    """
    Example 2: Benchmark specific model and dataset
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Model and Dataset")
    print("="*80)
    
    config = BenchmarkConfig()
    
    # Customize configuration
    config.MODEL_NAME = "DLinear"           # Changed model
    config.DATASET_NAME = "SMD"             # Changed dataset
    config.BATCH_SIZE = 64                  # Larger batch size
    config.NUM_ITERATIONS = 20              # More measurements
    config.DEVICE = "cuda"                  # Use GPU if available
    config.ANOMALY_THRESHOLD = 0.5          # Adjust threshold
    
    # Run benchmark
    benchmark = BaselineBenchmark(config)
    results = benchmark.run_benchmark()
    benchmark.save_results()
    benchmark.print_summary()
    
    return results


def example_3_compare_multiple_models():
    """
    Example 3: Compare multiple baseline models
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Multiple Models")
    print("="*80)
    
    models = ["MTAD_GAT", "DLinear", "Autoformer"]
    all_results = {}
    
    for model_name in models:
        print(f"\n--- Benchmarking {model_name} ---")
        
        config = BenchmarkConfig()
        config.MODEL_NAME = model_name
        config.DATASET_NAME = "SMD"
        config.BATCH_SIZE = 32
        
        benchmark = BaselineBenchmark(config)
        results = benchmark.run_benchmark()
        benchmark.save_results()
        
        all_results[model_name] = results
    
    # Print comparison summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Latency':<12} {'RAM':<10}")
    print("-"*74)
    
    for model_name, results in all_results.items():
        metrics = results['performance_metrics']
        latency = results['inference']['latency_per_batch_ms']['mean_latency_ms']
        ram_delta = results['memory']['ram_delta_mb']
        
        print(f"{model_name:<20} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{latency:<12.4f} "
              f"{ram_delta:<10.2f}")
    
    print("="*80)
    
    return all_results


def example_4_extract_metrics():
    """
    Example 4: Extract and analyze specific metrics
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Extract and Analyze Metrics")
    print("="*80)
    
    config = BenchmarkConfig()
    config.MODEL_NAME = "MTAD_GAT"
    
    benchmark = BaselineBenchmark(config)
    results = benchmark.run_benchmark()
    
    # Extract performance metrics
    print("\n[PERFORMANCE METRICS]")
    metrics = results['performance_metrics']
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    if metrics.get('auc_roc'):
        print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    
    # Extract inference metrics
    print("\n[INFERENCE METRICS]")
    inference = results['inference']
    latency = inference['latency_per_batch_ms']
    print(f"  Total Time:        {inference['total_inference_time_seconds']:.4f} seconds")
    print(f"  Mean Latency:      {latency['mean_latency_ms']:.4f} ms/batch")
    print(f"  Median Latency:    {latency['median_latency_ms']:.4f} ms/batch")
    print(f"  Std Latency:       {latency['std_latency_ms']:.4f} ms/batch")
    print(f"  Min Latency:       {latency['min_latency_ms']:.4f} ms/batch")
    print(f"  Max Latency:       {latency['max_latency_ms']:.4f} ms/batch")
    
    # Extract memory metrics
    print("\n[MEMORY METRICS]")
    memory = results['memory']
    print(f"  RAM Before:        {memory['ram_before_mb']:.2f} MB")
    print(f"  RAM After:         {memory['ram_after_mb']:.2f} MB")
    print(f"  RAM Delta:         {memory['ram_delta_mb']:+.2f} MB")
    print(f"  VRAM Before:       {memory['vram_before_mb']:.2f} MB")
    print(f"  VRAM After:        {memory['vram_after_mb']:.2f} MB")
    print(f"  VRAM Delta:        {memory['vram_delta_mb']:+.2f} MB")
    
    # Extract model info
    print("\n[MODEL INFO]")
    model_info = results['model_info']
    print(f"  Total Parameters:  {model_info['total_parameters']:,}")
    print(f"  Trainable Params:  {model_info['trainable_parameters']:,}")
    print(f"  Load Time:         {model_info['load_time_seconds']:.4f} seconds")
    
    # Extract data info
    print("\n[DATASET INFO]")
    data = results['data_loading']
    print(f"  Num Samples:       {data['num_samples']}")
    print(f"  Input Shape:       {data['input_shape']}")
    print(f"  Anomalies:         {data['num_anomalies']}")
    print(f"  Load Time:         {data['load_time_seconds']:.4f} seconds")
    
    benchmark.save_results()
    
    return results


def example_5_benchmark_with_checkpoints():
    """
    Example 5: Benchmark with pre-trained checkpoints
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Benchmark with Pre-trained Model")
    print("="*80)
    
    config = BenchmarkConfig()
    config.MODEL_NAME = "MTAD_GAT"
    config.DATASET_NAME = "SMD"
    
    # Set path to your pre-trained checkpoint
    # config.MODEL_CHECKPOINT = "./checkpoints/mtad_gat_best.pth"
    # For now, we'll skip checkpoint loading if it doesn't exist
    
    benchmark = BaselineBenchmark(config)
    results = benchmark.run_benchmark()
    benchmark.save_results()
    benchmark.print_summary()
    
    print("\n[INFO] To use pre-trained checkpoints:")
    print("  1. Set config.MODEL_CHECKPOINT = 'path/to/checkpoint.pth'")
    print("  2. Ensure checkpoint contains model weights")
    print("  3. Checkpoint will be loaded during model initialization")
    
    return results


def main():
    """
    Main function: Run all examples or select specific one
    """
    
    print("\n" + "="*80)
    print("BASELINE MODEL BENCHMARKING EXAMPLES")
    print("="*80)
    print("\nSelect an example to run:")
    print("  1. Simple benchmark (default configuration)")
    print("  2. Custom model and dataset")
    print("  3. Compare multiple models")
    print("  4. Extract and analyze specific metrics")
    print("  5. Benchmark with pre-trained checkpoints")
    print("  0. Run ALL examples")
    
    # Uncomment one of the examples below to run:
    
    # ===== OPTION A: Run all examples =====
    # example_1_simple_benchmark()
    # example_2_custom_model_and_dataset()
    # example_3_compare_multiple_models()
    # example_4_extract_metrics()
    # example_5_benchmark_with_checkpoints()
    
    # ===== OPTION B: Run specific example =====
    # Uncomment the example you want to run:
    
    # example_1_simple_benchmark()
    example_2_custom_model_and_dataset()
    # example_3_compare_multiple_models()
    # example_4_extract_metrics()
    # example_5_benchmark_with_checkpoints()
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETED")
    print("="*80)
    print("\nResults saved to: ./benchmark_results/")
    print("Check BENCHMARKING_GUIDE.md for more information")


if __name__ == "__main__":
    main()
