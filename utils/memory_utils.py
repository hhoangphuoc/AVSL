"""
Utility functions for memory management and GPU optimization.
"""
import os
import gc
import torch
import numpy as np
from typing import Tuple, Dict, Optional, Union

def setup_memory_optimizations() -> None:
    """
    Configure PyTorch and CUDA for optimal memory usage.
    Call this at the start of your script.
    """
    # Critical CUDA memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable for better performance
    os.environ['TORCH_CUDNN_DISABLE'] = '0'  # Keep cuDNN enabled for performance
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.set_float32_matmul_precision('medium')  # Use Tensor Cores more efficiently

def clear_gpu_memory() -> None:
    """
    Aggressively clear GPU memory.
    Call this when transitioning between major operations.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory statistics for reporting.
    
    Returns:
        Dictionary containing memory statistics in GB
    """
    stats = {}
    if torch.cuda.is_available():
        stats['allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        stats['reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        stats['max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        stats['max_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)
    
    # Add system memory stats if needed
    try:
        import psutil
        mem = psutil.virtual_memory()
        stats['system_total_gb'] = mem.total / (1024**3)
        stats['system_available_gb'] = mem.available / (1024**3)
        stats['system_used_percent'] = mem.percent
    except ImportError:
        pass
        
    return stats

def log_memory_stats(prefix: str = "") -> None:
    """
    Log current memory statistics.
    
    Args:
        prefix: Optional prefix for the log message
    """
    stats = get_memory_stats()
    prefix = f"{prefix} ---------------- " if prefix else ""
    
    if torch.cuda.is_available():
        print(f"{prefix}GPU Memory: "
              f"Allocated: {stats['allocated_gb']:.2f} GB, "
              f"Reserved: {stats['reserved_gb']:.2f} GB, "
              f"Max Allocated: {stats['max_allocated_gb']:.2f} GB, "
              f"Max Reserved: {stats['max_reserved_gb']:.2f} GB")
    
    if 'system_total_gb' in stats:
        print(f"{prefix}System Memory: "
              f"Used: {stats['system_used_percent']:.1f}%, "
              f"Available: {stats['system_available_gb']:.2f} GB / {stats['system_total_gb']:.2f} GB")

def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Enable gradient checkpointing for supported PyTorch modules.
    
    Args:
        model: PyTorch model
    """
    # Enable checkpointing for transformers components if they exist
    for submodule in model.modules():
        if hasattr(submodule, 'gradient_checkpointing_enable'):
            submodule.gradient_checkpointing_enable()
    
    print("✓ Gradient checkpointing enabled for supported modules")

def limit_batch_size_to_memory(base_batch_size: int, 
                              required_memory_per_item: float,
                              available_memory: Optional[float] = None,
                              safety_factor: float = 0.8) -> int:
    """
    Adaptively limit batch size based on available GPU memory.
    
    Args:
        base_batch_size: The desired batch size
        required_memory_per_item: Memory required per item in GB
        available_memory: Available GPU memory in GB (auto-detected if None)
        safety_factor: Factor to multiply available memory (0.0-1.0) as safety buffer
        
    Returns:
        Adjusted batch size
    """
    if not torch.cuda.is_available():
        return base_batch_size
        
    if available_memory is None:
        stats = get_memory_stats()
        reserved = stats.get('reserved_gb', 0)
        max_reserved = stats.get('max_reserved_gb', 0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory = total - max(reserved, max_reserved)
    
    # Apply safety factor
    available_memory *= safety_factor
    
    # Calculate max batch size
    max_batch_size = max(1, int(available_memory / required_memory_per_item))
    
    # Return smaller of desired and maximum safe batch size
    result = min(base_batch_size, max_batch_size)
    print(f"✓ Adjusted batch size from {base_batch_size} to {result} based on memory constraints")
    return result

def measure_model_memory_usage(model: torch.nn.Module, 
                              input_shape: Union[Tuple, Dict],
                              dtype: torch.dtype = torch.float32) -> float:
    """
    Estimate memory usage of a model for a given input shape.
    
    Args:
        model: The PyTorch model
        input_shape: Input shape (tuple) or dict mapping input names to shapes
        dtype: Data type for estimation
        
    Returns:
        Estimated memory usage in GB
    """
    # Move model to CPU for measurement to avoid OOM errors
    model = model.cpu()
    clear_gpu_memory()
    
    # Prepare dummy inputs
    if isinstance(input_shape, tuple):
        input_tensor = torch.rand(input_shape, dtype=dtype)
        dummy_input = input_tensor
    elif isinstance(input_shape, dict):
        dummy_input = {}
        for name, shape in input_shape.items():
            if isinstance(shape, tuple):
                dummy_input[name] = torch.rand(shape, dtype=dtype)
            else:
                dummy_input[name] = shape  # Assume it's already a tensor
    else:
        raise ValueError("input_shape must be a tuple or dict mapping input names to shapes")
    
    # Create a trace
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Get graph and parameters
        graph = traced_model.inlined_graph
        parameters = dict(model.named_parameters())
        
        # Estimate activation memory
        activation_memory = 0
        for node in graph.nodes():
            for output in node.outputs():
                if output.type().kind() == 'TensorType':
                    size = 1
                    for dim in output.type().sizes():
                        if dim is not None and dim > 0:
                            size *= dim
                    activation_memory += size * dtype.itemsize
        
        # Calculate parameter memory
        parameter_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Calculate total memory in GB
        total_memory_gb = (activation_memory + parameter_memory) / (1024**3)
        
        return total_memory_gb
        
    except Exception as e:
        print(f"Warning: Failed to trace model for memory estimation: {e}")
        # Fallback: just estimate parameter size
        parameter_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        return parameter_memory / (1024**3)