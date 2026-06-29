"""
Comprehensive Baseline Model Benchmarking Script
Phase 2: Benchmarking baseline models with performance metrics and computational overhead measurement

Author: ML Engineering Team
Date: 2024
"""

import os
import sys
import json
import csv
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, Tuple, List, Any
import warnings

from data.data_loader import Dataset_SMD, Dataset_SWaT

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THIS SECTION FOR YOUR SPECIFIC USE CASE
# ============================================================================

class BenchmarkConfig:
    """Configuration for benchmarking experiments"""
    
    # Dataset Configuration
    DATASET_NAME = "SMD"  # ["SMD", "MSL", "SMAP", "PSM", "SWaT", "WADI", "MBA", "UCR", "NAB", "MSDS"]
    DATA_PATH = "./data/"
    DATA_PROCESS = True
    PARTIAL_DATA = False
    WINDOW_SIZE = 720
    LABEL_LEN = 48
    
    # Model Configuration
    MODEL_NAME = "MTAD_GAT"  # ["RTNet", "DLinear", "Autoformer", "DeepAR", "GTA", "Informer", "LSTNet", "MA", "MTAD_GAT", "GDN"]
    MODEL_CHECKPOINT = None  # Path to pretrained model (if available)
    USE_LIN = True
    USE_PAM = False

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Benchmarking Configuration
    BATCH_SIZE = 32
    NUM_ITERATIONS = 10  # Number of iterations for averaging metrics
    WARMUP_ITERATIONS = 2  # Warmup runs before actual measurement
    
    # Output Configuration
    OUTPUT_DIR = "./benchmark_results/"
    SAVE_FORMAT = ["csv", "json"]  # Save formats for results

    # Metric Thresholds (for anomaly detection)
    ANOMALY_THRESHOLD = 0.005  # Anomaly ratio r used to derive the score threshold
    

# ============================================================================
# UTILITY FUNCTIONS - FOR MEMORY AND TIME MEASUREMENT
# ============================================================================

class PerformanceMonitor:
    """Monitor computational performance metrics"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        mem_info = self.process.memory_info()
        
        metrics = {
            "ram_mb": mem_info.rss / 1024 / 1024,
            "rss_mb": mem_info.rss / 1024 / 1024,
        }
        
        # VRAM usage if GPU is available
        if self.device == "cuda" and torch.cuda.is_available():
            metrics["vram_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            metrics["vram_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        else:
            metrics["vram_mb"] = 0.0
            metrics["vram_reserved_mb"] = 0.0
        
        return metrics
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage"""
        metrics = self.get_memory_usage()
        return metrics
    
    def reset_gpu_memory(self):
        """Reset GPU memory cache"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


class LatencyMeasurer:
    """Measure inference latency with precision"""
    
    def __init__(self, device: str = "cpu", warmup_runs: int = 2):
        self.device = device
        self.warmup_runs = warmup_runs
        self.latencies = []
    
    def warmup(self, model: nn.Module, sample_input: torch.Tensor):
        """Warm up the model to stabilize measurements"""
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(sample_input.clone())
    
    def measure(self, model: nn.Module, input_data: torch.Tensor, num_measurements: int = 10) -> Dict[str, float]:
        """
        Measure inference latency
        
        Returns:
            Dictionary with latency statistics (in milliseconds)
        """
        model.eval()
        self.latencies = []
        
        # Synchronize GPU if using CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_measurements):
                start_time = time.perf_counter()
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                _ = model(input_data.clone())
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                self.latencies.append(latency_ms)
        
        return {
            "mean_latency_ms": np.mean(self.latencies),
            "median_latency_ms": np.median(self.latencies),
            "std_latency_ms": np.std(self.latencies),
            "min_latency_ms": np.min(self.latencies),
            "max_latency_ms": np.max(self.latencies),
        }


# ============================================================================
# DATA LOADING SECTION - INSERT YOUR DATASET LOGIC HERE
# ============================================================================

def load_dataset(dataset_name: str, data_path: str = "./data/", config: BenchmarkConfig = None):
    """
    Load a dataset as a streaming window loader plus metadata.
    """

    if config is None:
        config = BenchmarkConfig()

    try:
        if dataset_name == "SMD":
            dataset_root = data_path if os.path.basename(os.path.normpath(data_path)) == "SMD" else os.path.join(data_path, "SMD")
            dataset = Dataset_SMD(
                flag="test",
                input_len=config.WINDOW_SIZE,
                data_path=dataset_root,
                data_process=config.DATA_PROCESS,
                LIN=config.USE_LIN,
                partial_train=config.PARTIAL_DATA,
            )
            meta = {
                "dataset_name": "SMD",
                "window_size": config.WINDOW_SIZE,
                "raw_variates": 38,
                "continuous_variates": 37,
                "discrete_variates": 0,
                "trivial_variates": 1,
                "num_input_features": int(dataset.test.shape[1]),
                "num_output_features": int(dataset.test.shape[1]),
                "test_points": int(len(dataset.get_label())),
            }

        elif dataset_name == "SWaT":
            dataset = Dataset_SWaT(
                flag="test",
                input_len=config.WINDOW_SIZE,
                data_path=data_path,
                data_process=True if config.DATA_PROCESS is None else config.DATA_PROCESS,
                LIN=config.USE_LIN,
            )
            meta = {
                "dataset_name": "SWaT",
                "window_size": config.WINDOW_SIZE,
                "raw_variates": 51,
                "continuous_variates": 25,
                "discrete_variates": 15,
                "trivial_variates": 11,
                "num_input_features": int(dataset.test.shape[1]),
                "num_output_features": 25,
                "test_points": int(len(dataset.get_label())),
            }

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        return {
            "dataset": dataset,
            "loader": loader,
            "labels": torch.LongTensor(dataset.get_label().reshape(-1)),
            "meta": meta,
        }

    except Exception as e:
        print(f"[ERROR] Failed to load dataset {dataset_name}: {str(e)}")
        raise


# ============================================================================
# MODEL LOADING SECTION - INSERT YOUR MODEL LOGIC HERE
# ============================================================================

def load_model(model_name: str, device: str = "cpu", checkpoint_path: str = None,
               dataset_meta: Dict[str, Any] = None, config: BenchmarkConfig = None) -> nn.Module:
    """
    Load baseline model for benchmarking
    
    *** INSERT YOUR MODEL LOADING LOGIC HERE ***
    
    Args:
        model_name: Name of the model
        device: Device to load model on ('cpu' or 'cuda')
        checkpoint_path: Path to pretrained weights
    
    Returns:
        PyTorch model in evaluation mode
    """
    
    try:
        if dataset_meta is None:
            dataset_meta = {
                "window_size": 720,
                "num_input_features": 37,
                "num_output_features": 37,
            }
        if config is None:
            config = BenchmarkConfig()

        seq_len = int(dataset_meta["window_size"])
        variate = int(dataset_meta["num_input_features"])
        out_variate = int(dataset_meta["num_output_features"])

        if model_name == "MTAD_GAT":
            from models.MTAD_GAT.MTAD_GAT import MTAD_GAT
            model = MTAD_GAT(
                variate=variate,
                out_variate=out_variate,
                input_len=seq_len,
                kernel_size=7,
                feat_gat_embed_dim=None,
                time_gat_embed_dim=None,
                use_gatv2=False,
                gru_n_layers=1,
                gru_hid_dim=150,
                forecast_n_layers=3,
                forecast_hid_dim=150,
                recon_n_layers=1,
                recon_hid_dim=150,
                dropout=0.2,
                alpha=0.2,
                LIN=config.USE_LIN,
            )

        elif model_name == "MA":
            from models.MA.MA import MA
            model = MA(out_variate=out_variate, input_len=seq_len, LIN=config.USE_LIN)

        elif model_name == "DLinear":
            from models.DLinear.DLinear import DLinear
            model = DLinear(variate=variate, out_variate=out_variate, input_len=seq_len, kernel=3, LIN=config.USE_LIN)

        elif model_name == "Autoformer":
            from models.Autoformer.Autoformer import Autoformer
            model = Autoformer(
                variate=variate,
                out_variate=out_variate,
                input_len=seq_len,
                label_len=config.LABEL_LEN,
                moving_avg=[24],
                d_model=512,
                dropout=0.05,
                factor=1,
                n_heads=8,
                activation='gelu',
                e_layers=2,
                d_layers=1,
                LIN=config.USE_LIN,
            )

        elif model_name == "GTA":
            from models.GTA.gta import GTA
            model = GTA(
                num_nodes=variate,
                c_out=out_variate,
                input_len=seq_len,
                label_len=config.LABEL_LEN,
                num_levels=3,
                factor=5,
                d_model=512,
                n_heads=8,
                e_layers=2,
                d_layers=1,
                dropout=0.1,
                activation='gelu',
                LIN=config.USE_LIN,
                device=torch.device(device),
            )

        # Load from checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model
    
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name}: {str(e)}")
        raise


# ============================================================================
# INFERENCE AND METRICS CALCULATION SECTION
# ============================================================================

def run_inference_batch(
    model: nn.Module,
    data: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 32
) -> np.ndarray:
    """
    Run inference on batches of data
    
    Args:
        model: PyTorch model
        data: Input tensor of shape (num_samples, seq_len, num_features)
        device: Device to run on
        batch_size: Batch size for inference
    
    Returns:
        Model predictions/anomaly scores
    """
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            
            try:
                # *** YOUR MODEL OUTPUT LOGIC HERE ***
                # Adjust based on your model's output format
                output = model(batch)
                
                # Convert to numpy for post-processing
                if isinstance(output, tuple):
                    # Some models return multiple outputs
                    output = output[0]
                
                predictions.append(output.cpu().numpy())
            
            except Exception as e:
                print(f"[ERROR] Inference failed for batch {i}: {str(e)}")
                raise
    
    return np.concatenate(predictions, axis=0)


def calculate_anomaly_scores(predictions: np.ndarray, method: str = "reconstruction_error") -> np.ndarray:
    """
    Calculate anomaly scores from model predictions
    
    *** CUSTOMIZE THIS BASED ON YOUR MODEL'S OUTPUT ***
    
    Args:
        predictions: Model output
        method: Method to calculate anomaly scores
    
    Returns:
        Anomaly scores (0-1 range)
    """
    
    if method == "reconstruction_error":
        # For autoencoder-like models
        scores = np.mean(predictions, axis=(1, 2))
    
    elif method == "direct_output":
        # If model directly outputs anomaly scores
        scores = predictions.flatten()
    
    else:
        # Default: use absolute values
        scores = np.abs(predictions).mean(axis=(1, 2))
    
    # Normalize to [0, 1]
    scores_min = scores.min()
    scores_max = scores.max()
    if scores_max > scores_min:
        scores = (scores - scores_min) / (scores_max - scores_min)
    else:
        scores = np.zeros_like(scores)
    
    return scores


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1-Score
    
    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly)
        y_pred: Predicted anomaly scores (0-1)
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    
    # Convert scores to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
    
    metrics = {
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred_binary),
    }
    
    # AUC-ROC if we have both positive and negative samples
    if len(np.unique(y_true)) > 1:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
        except:
            metrics["auc_roc"] = None
    
    return metrics


# ============================================================================
# BENCHMARKING ORCHESTRATION
# ============================================================================

class BaselineBenchmark:
    """Main benchmarking orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.create_output_dir()
    
    def create_output_dir(self):
        """Create output directory for results"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        print(f"[INFO] Output directory: {self.config.OUTPUT_DIR}")
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Execute complete benchmarking pipeline
        
        Returns:
            Dictionary with all benchmark results
        """
        
        print("=" * 80)
        print("BASELINE MODEL BENCHMARKING")
        print("=" * 80)
        print(f"[INFO] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[INFO] Device: {self.config.DEVICE}")
        print(f"[INFO] Model: {self.config.MODEL_NAME}")
        print(f"[INFO] Dataset: {self.config.DATASET_NAME}")
        print("=" * 80)
        
        # Step 1: Load dataset
        print("\n[STEP 1] Loading dataset...")
        start_data_load = time.time()
        dataset_bundle = load_dataset(self.config.DATASET_NAME, self.config.DATA_PATH, self.config)
        data_load_time = time.time() - start_data_load
        print(f"[SUCCESS] Dataset loaded in {data_load_time:.4f}s")
        test_loader = dataset_bundle["loader"]
        dataset_meta = dataset_bundle["meta"]
        y_test = dataset_bundle["labels"]
        print(f"  - Window size: {dataset_meta['window_size']}")
        print(f"  - Input features: {dataset_meta['num_input_features']}")
        print(f"  - Output features: {dataset_meta['num_output_features']}")
        print(f"  - Test labels: {y_test.shape[0]}")
        print(f"  - Anomalies: {int(y_test.sum().item())} / {len(y_test)}")
        component_state = {
            "LIN": bool(self.config.USE_LIN),
            "LF": dataset_meta["num_output_features"] < dataset_meta["num_input_features"],
            "PAM": bool(self.config.USE_PAM),
        }
        if component_state["LIN"] and component_state["LF"] and component_state["PAM"]:
            benchmark_variant = "AFMF-enhanced"
        elif component_state["LIN"] or component_state["LF"] or component_state["PAM"]:
            benchmark_variant = "AFMF-partial"
        else:
            benchmark_variant = "bare forecasting baseline"
        print(f"  - Benchmark variant: {benchmark_variant} (LIN={component_state['LIN']}, LF={component_state['LF']}, PAM={component_state['PAM']})")
        
        # Step 2: Load model
        print("\n[STEP 2] Loading baseline model...")
        start_model_load = time.time()
        model = load_model(
            self.config.MODEL_NAME,
            self.config.DEVICE,
            self.config.MODEL_CHECKPOINT,
            dataset_meta=dataset_meta,
            config=self.config,
        )
        model_load_time = time.time() - start_model_load
        print(f"[SUCCESS] Model loaded in {model_load_time:.4f}s")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Step 3: Measure memory overhead
        print("\n[STEP 3] Measuring memory overhead...")
        monitor = PerformanceMonitor(self.config.DEVICE)
        monitor.reset_gpu_memory()
        mem_before = monitor.get_memory_usage()
        print(f"  - RAM before: {mem_before['ram_mb']:.2f} MB")
        if self.config.DEVICE == "cuda":
            print(f"  - VRAM before: {mem_before['vram_mb']:.2f} MB")
        
        # Step 4: Run inference and measure latency
        print("\n[STEP 4] Running inference and measuring latency...")
        latency_measurer = LatencyMeasurer(self.config.DEVICE, self.config.WARMUP_ITERATIONS)
        
        # Get a sample batch for warmup
        sample_batch = next(iter(test_loader)).float().to(self.config.DEVICE).clone()
        latency_measurer.warmup(model, sample_batch)
        
        # Measure latency on the full test loader and collect anomaly scores
        start_inference = time.time()
        anomaly_scores = []
        model.eval()
        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x.float().to(self.config.DEVICE)
                output = model(batch_x)

                if isinstance(output, tuple):
                    if len(output) == 3:
                        pred, recon, true = output
                        mse = (pred - true) ** 2 + (recon - true) ** 2
                    elif len(output) == 2:
                        pred, true = output
                        mse = (pred - true) ** 2
                    else:
                        pred = output[0]
                        mse = pred.abs()
                else:
                    mse = output.abs()

                if mse.ndim == 1:
                    batch_scores = mse
                else:
                    reduce_dims = tuple(range(1, mse.ndim))
                    batch_scores = mse.mean(dim=reduce_dims)
                anomaly_scores.append(batch_scores.detach().cpu().numpy().reshape(-1))
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        total_inference_time = time.time() - start_inference
        
        # Measure latency per batch
        latency_stats = latency_measurer.measure(model, sample_batch, self.config.NUM_ITERATIONS)
        
        print(f"[SUCCESS] Inference completed in {total_inference_time:.4f}s")
        print(f"  - Mean latency per batch: {latency_stats['mean_latency_ms']:.4f} ms")
        print(f"  - Median latency: {latency_stats['median_latency_ms']:.4f} ms")
        print(f"  - Std latency: {latency_stats['std_latency_ms']:.4f} ms")
        
        # Step 5: Measure memory after inference
        print("\n[STEP 5] Measuring peak memory usage...")
        mem_after = monitor.get_memory_usage()
        print(f"  - RAM after: {mem_after['ram_mb']:.2f} MB")
        if self.config.DEVICE == "cuda":
            print(f"  - VRAM after: {mem_after['vram_mb']:.2f} MB")
        
        # Step 6: Calculate performance metrics
        print("\n[STEP 6] Calculating performance metrics...")
        score_threshold = np.percentile(anomaly_scores, 100 - (self.config.ANOMALY_THRESHOLD * 100))
        window_labels = y_test.cpu().numpy()[self.config.WINDOW_SIZE - 1:]
        min_len = min(len(window_labels), len(anomaly_scores))
        window_labels = window_labels[:min_len]
        anomaly_scores = anomaly_scores[:min_len]
        metrics = calculate_metrics(window_labels, (anomaly_scores >= score_threshold).astype(int), 0.5)
        
        print(f"[SUCCESS] Metrics calculated:")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1-Score: {metrics['f1_score']:.4f}")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        try:
            from sklearn.metrics import roc_auc_score
            auc_roc = roc_auc_score(window_labels, anomaly_scores)
            metrics["auc_roc"] = auc_roc
            print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
        except Exception:
            metrics["auc_roc"] = None
        
        # Compile results
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.config.MODEL_NAME,
                "dataset_name": self.config.DATASET_NAME,
                "device": self.config.DEVICE,
                "batch_size": self.config.BATCH_SIZE,
                "benchmark_variant": benchmark_variant,
                "afmf_components": component_state,
            },
            "data_loading": {
                "load_time_seconds": data_load_time,
                "num_samples": len(window_labels),
                "input_shape": f"{dataset_meta['window_size']}x{dataset_meta['num_input_features']}",
                "num_anomalies": int(window_labels.sum()),
            },
            "model_info": {
                "load_time_seconds": model_load_time,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            "inference": {
                "total_inference_time_seconds": total_inference_time,
                "latency_per_batch_ms": latency_stats,
                "score_threshold": float(score_threshold),
            },
            "memory": {
                "ram_before_mb": mem_before['ram_mb'],
                "ram_after_mb": mem_after['ram_mb'],
                "ram_delta_mb": mem_after['ram_mb'] - mem_before['ram_mb'],
                "vram_before_mb": mem_before.get('vram_mb', 0),
                "vram_after_mb": mem_after.get('vram_mb', 0),
                "vram_delta_mb": mem_after.get('vram_mb', 0) - mem_before.get('vram_mb', 0),
            },
            "performance_metrics": metrics,
            "anomaly_threshold_ratio": self.config.ANOMALY_THRESHOLD,
        }
        
        return self.results
    
    def save_results(self):
        """Save results to CSV and JSON formats"""
        
        print("\n[STEP 7] Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        if "json" in self.config.SAVE_FORMAT:
            json_path = os.path.join(
                self.config.OUTPUT_DIR,
                f"benchmark_{self.config.MODEL_NAME}_{timestamp}.json"
            )
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"[SUCCESS] JSON report saved: {json_path}")
        
        # Save CSV (flattened format)
        if "csv" in self.config.SAVE_FORMAT:
            csv_path = os.path.join(
                self.config.OUTPUT_DIR,
                f"benchmark_{self.config.MODEL_NAME}_{timestamp}.csv"
            )
            
            # Flatten results for CSV
            flattened_results = self._flatten_results(self.results)
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_results.keys())
                writer.writeheader()
                writer.writerow(flattened_results)
            
            print(f"[SUCCESS] CSV report saved: {csv_path}")
    
    def _flatten_results(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_results(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def print_summary(self):
        """Print summary of benchmarking results"""
        
        print("\n" + "=" * 80)
        print("BENCHMARKING SUMMARY")
        print("=" * 80)
        print(f"Model: {self.results['metadata']['model_name']}")
        print(f"Dataset: {self.results['metadata']['dataset_name']}")
        print(f"Device: {self.results['metadata']['device']}")
        print("-" * 80)
        print("PERFORMANCE METRICS:")
        metrics = self.results['performance_metrics']
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1-Score:     {metrics['f1_score']:.4f}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print("-" * 80)
        print("COMPUTATIONAL OVERHEAD:")
        inference = self.results['inference']
        print(f"  Total Inference Time: {inference['total_inference_time_seconds']:.4f}s")
        print(f"  Mean Latency:         {inference['latency_per_batch_ms']['mean_latency_ms']:.4f} ms/batch")
        print(f"  Median Latency:       {inference['latency_per_batch_ms']['median_latency_ms']:.4f} ms/batch")
        print("-" * 80)
        print("MEMORY USAGE:")
        memory = self.results['memory']
        print(f"  RAM Delta:  {memory['ram_delta_mb']:+.2f} MB")
        if self.results['metadata']['device'] == 'cuda':
            print(f"  VRAM Delta: {memory['vram_delta_mb']:+.2f} MB")
        print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    try:
        # Initialize configuration
        config = BenchmarkConfig()
        
        # Run benchmarking
        benchmark = BaselineBenchmark(config)
        benchmark.run_benchmark()
        
        # Save results
        benchmark.save_results()
        
        # Print summary
        benchmark.print_summary()
        
        print("[INFO] Benchmarking completed successfully!")
        return benchmark.results
    
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()
