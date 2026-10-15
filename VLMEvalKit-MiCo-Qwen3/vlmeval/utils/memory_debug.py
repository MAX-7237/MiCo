"""
Memory debugging utilities for tracking CUDA memory usage and identifying OOM issues.
"""

import torch
import gc
import traceback
from datetime import datetime
from typing import Optional, Dict, Any


def format_bytes(size_bytes: float) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def log_memory_stats(location: str, show_tensors: bool = False, verbose: bool = True) -> Dict[str, float]:
    """
    Log current CUDA memory statistics.

    Args:
        location: Description of where this is being called from
        show_tensors: If True, show information about large tensors
        verbose: If True, print detailed information

    Returns:
        Dictionary with memory statistics in GB
    """
    stats = {}

    if not torch.cuda.is_available():
        if verbose:
            print(f"[{location}] CUDA not available")
        return stats

    try:
        # Get memory statistics in GB
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        free = reserved - allocated

        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        available = total_memory - reserved

        stats = {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'free_in_reserved': free,
            'total_gpu': total_memory,
            'available': available
        }

        if verbose:
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"\n{'='*70}")
            print(f"[{timestamp}] Memory Stats @ {location}")
            print(f"{'='*70}")
            print(f"  GPU Total:        {total_memory:.2f} GB")
            print(f"  Allocated:        {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
            print(f"  Reserved:         {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
            print(f"  Max Allocated:    {max_allocated:.2f} GB")
            print(f"  Free (Reserved):  {free:.2f} GB")
            print(f"  Available:        {available:.2f} GB")

        if show_tensors and verbose:
            print("\n  Large Tensors (>10MB):")
            tensor_count = 0
            total_tensor_memory = 0
            tensors_by_size = []

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        size_bytes = obj.element_size() * obj.nelement()
                        size_gb = size_bytes / 1024**3
                        total_tensor_memory += size_gb

                        if size_bytes > 10 * 1024 * 1024:  # > 10MB
                            tensors_by_size.append((size_bytes, obj.shape, obj.dtype))
                            tensor_count += 1
                except:
                    pass

            # Sort by size and show top 10
            tensors_by_size.sort(reverse=True)
            for i, (size_bytes, shape, dtype) in enumerate(tensors_by_size[:10]):
                print(f"    {i+1:2d}. {shape} ({dtype}) -> {format_bytes(size_bytes)}")

            if tensor_count > 10:
                print(f"    ... and {tensor_count - 10} more large tensors")
            print(f"  Total tensor memory: {total_tensor_memory:.2f} GB")

        if verbose:
            print(f"{'='*70}\n")

    except Exception as e:
        print(f"Error getting memory stats at {location}: {e}")

    return stats


def memory_profiler(func):
    """
    Decorator to profile memory usage before and after a function call.
    """
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        # Log before
        before_stats = log_memory_stats(f"Before {func_name}", verbose=True)

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Log after
            after_stats = log_memory_stats(f"After {func_name}", verbose=True)

            # Calculate delta
            if before_stats and after_stats:
                delta = after_stats['allocated'] - before_stats['allocated']
                if abs(delta) > 0.1:  # Only show if change > 100MB
                    print(f"  Memory change in {func_name}: {delta:+.2f} GB")

            return result

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n{'!'*70}")
            print(f"!!! CUDA OOM in {func_name} !!!")
            print(f"{'!'*70}")
            log_memory_stats(f"OOM in {func_name}", show_tensors=True, verbose=True)

            # Print stack trace
            print("\nStack trace:")
            traceback.print_exc()
            raise e

    return wrapper


def validate_attention_memory(q_shape, k_shape, num_heads: Optional[int] = None,
                             dtype: torch.dtype = torch.float32, location: str = "") -> float:
    """
    Validate and estimate memory required for attention computation.

    Args:
        q_shape: Shape of query tensor
        k_shape: Shape of key tensor
        num_heads: Number of attention heads
        dtype: Data type of tensors
        location: Where this computation happens

    Returns:
        Estimated memory in GB
    """
    # Determine sequence lengths
    if len(q_shape) == 3:  # [batch/heads, seq_len, dim]
        seq_len_q = q_shape[1]
        heads = q_shape[0] if num_heads is None else num_heads
    else:  # [batch, heads, seq_len, dim]
        seq_len_q = q_shape[2]
        heads = q_shape[1] if num_heads is None else num_heads

    seq_len_k = k_shape[1] if len(k_shape) == 3 else k_shape[2]

    # Calculate bytes per element
    if dtype in [torch.float16, torch.bfloat16]:
        bytes_per_element = 2
    elif dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float64:
        bytes_per_element = 8
    else:
        bytes_per_element = 4  # default

    # Calculate memory for attention weights
    attention_elements = heads * seq_len_q * seq_len_k
    attention_memory_bytes = attention_elements * bytes_per_element
    attention_memory_gb = attention_memory_bytes / 1024**3

    print(f"\n  Attention Memory Estimate @ {location}:")
    print(f"    Q shape: {q_shape}, K shape: {k_shape}")
    print(f"    Attention shape: [{heads}, {seq_len_q}, {seq_len_k}]")
    print(f"    Elements: {attention_elements:,}")
    print(f"    Dtype: {dtype}, Bytes per element: {bytes_per_element}")
    print(f"    Estimated memory: {attention_memory_gb:.3f} GB")

    # Warning for large matrices
    if attention_memory_gb > 1.0:
        print(f"    ⚠️  WARNING: Large attention matrix! May cause OOM.")

    if seq_len_q > 2048 or seq_len_k > 2048:
        print(f"    ⚠️  WARNING: Very long sequence! Consider using Flash Attention.")

    return attention_memory_gb


def log_tensor_info(tensor: torch.Tensor, name: str = "Tensor", show_stats: bool = False):
    """
    Log detailed information about a tensor.

    Args:
        tensor: The tensor to analyze
        name: Name/description of the tensor
        show_stats: If True, show min/max/mean statistics
    """
    if not isinstance(tensor, torch.Tensor):
        print(f"  {name}: Not a tensor (type: {type(tensor)})")
        return

    memory_bytes = tensor.element_size() * tensor.nelement()
    memory_str = format_bytes(memory_bytes)

    device = str(tensor.device)

    print(f"  {name}:")
    print(f"    Shape: {list(tensor.shape)}")
    print(f"    Dtype: {tensor.dtype}")
    print(f"    Device: {device}")
    print(f"    Memory: {memory_str}")

    if show_stats and tensor.numel() > 0:
        try:
            if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                print(f"    Stats: min={tensor.min().item():.3f}, "
                      f"max={tensor.max().item():.3f}, "
                      f"mean={tensor.mean().item():.3f}")
        except:
            pass


def track_memory_during_forward(layer_name: str, input_tensor: torch.Tensor,
                               output_tensor: Optional[torch.Tensor] = None):
    """
    Track memory during a forward pass through a layer.

    Args:
        layer_name: Name of the layer
        input_tensor: Input to the layer
        output_tensor: Output from the layer (if available)
    """
    print(f"\n--- Layer: {layer_name} ---")

    # Log current memory
    stats = log_memory_stats(layer_name, verbose=False)
    if stats:
        print(f"  Memory: Allocated={stats['allocated']:.2f}GB, Available={stats['available']:.2f}GB")

    # Log input tensor
    log_tensor_info(input_tensor, "Input")

    # Log output tensor if provided
    if output_tensor is not None:
        log_tensor_info(output_tensor, "Output")

        # Calculate memory delta
        if isinstance(input_tensor, torch.Tensor) and isinstance(output_tensor, torch.Tensor):
            input_memory = input_tensor.element_size() * input_tensor.nelement() / 1024**3
            output_memory = output_tensor.element_size() * output_tensor.nelement() / 1024**3
            delta = output_memory - input_memory
            if abs(delta) > 0.001:  # Only show if > 1MB
                print(f"    Memory delta: {delta:+.3f} GB")


def generate_memory_report(log_file: Optional[str] = None):
    """
    Generate a comprehensive memory usage report.

    Args:
        log_file: Optional file to save the report to
    """
    report_lines = []
    report_lines.append("\n" + "="*70)
    report_lines.append("CUDA MEMORY USAGE REPORT")
    report_lines.append("="*70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not torch.cuda.is_available():
        report_lines.append("CUDA not available")
    else:
        # Device information
        device_props = torch.cuda.get_device_properties(0)
        report_lines.append(f"\nDevice: {device_props.name}")
        report_lines.append(f"Total Memory: {device_props.total_memory / 1024**3:.2f} GB")

        # Current memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        report_lines.append(f"\nCurrent Status:")
        report_lines.append(f"  Allocated: {allocated:.2f} GB")
        report_lines.append(f"  Reserved: {reserved:.2f} GB")
        report_lines.append(f"  Peak Allocated: {max_allocated:.2f} GB")

        # Memory stats history
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats()
            report_lines.append(f"\nMemory Statistics:")
            important_stats = [
                'allocated_bytes.all.current',
                'allocated_bytes.all.peak',
                'reserved_bytes.all.current',
                'reserved_bytes.all.peak',
                'active_bytes.all.current',
                'inactive_split_bytes.all.current'
            ]

            for stat_name in important_stats:
                if stat_name in stats:
                    value_gb = stats[stat_name] / 1024**3
                    report_lines.append(f"  {stat_name}: {value_gb:.3f} GB")

        # Memory summary
        torch.cuda.memory_summary()

    report_lines.append("="*70 + "\n")

    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file if requested
    if log_file:
        with open(log_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {log_file}")

    return report_text