#!/usr/bin/env python3
"""
GPU Memory Monitor Script

This script monitors GPU memory usage in real-time and can be run in parallel
with training to track memory consumption patterns.
"""

import os
import time
import torch
import subprocess
import signal
import sys
from datetime import datetime

class GPUMonitor:
    def __init__(self, interval=5, log_file=None):
        """
        Initialize GPU monitor.
        
        Args:
            interval: Monitoring interval in seconds
            log_file: Optional log file path
        """
        self.interval = interval
        self.log_file = log_file
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutting down GPU monitor...")
        self.running = False
        
    def get_gpu_info(self):
        """Get current GPU memory information."""
        info = {}
        
        if torch.cuda.is_available():
            # PyTorch memory info
            info['allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            info['reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            info['max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            info['max_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['total_gb'] = total_memory
            info['free_gb'] = total_memory - info['reserved_gb']
            info['utilization_percent'] = (info['reserved_gb'] / total_memory) * 100
            
            # Try to get additional info from nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_util, temp, power = result.stdout.strip().split(', ')
                    info['gpu_utilization_percent'] = float(gpu_util)
                    info['temperature_c'] = float(temp)
                    info['power_watts'] = float(power)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                pass
                
        return info
    
    def format_info(self, info):
        """Format GPU info for display."""
        if not info:
            return "GPU not available"
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            f"[{timestamp}] GPU Memory Status:",
            f"  Allocated: {info['allocated_gb']:.2f} GB",
            f"  Reserved:  {info['reserved_gb']:.2f} GB", 
            f"  Free:      {info['free_gb']:.2f} GB",
            f"  Total:     {info['total_gb']:.2f} GB",
            f"  Usage:     {info['utilization_percent']:.1f}%",
            f"  Max Alloc: {info['max_allocated_gb']:.2f} GB",
            f"  Max Reserv:{info['max_reserved_gb']:.2f} GB"
        ]
        
        if 'gpu_utilization_percent' in info:
            lines.append(f"  GPU Util:  {info['gpu_utilization_percent']:.1f}%")
        if 'temperature_c' in info:
            lines.append(f"  Temp:      {info['temperature_c']:.1f}°C")
        if 'power_watts' in info:
            lines.append(f"  Power:     {info['power_watts']:.1f}W")
            
        return "\n".join(lines)
    
    def log_info(self, info_str):
        """Log information to file and/or console."""
        print(info_str)
        print("-" * 50)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(info_str + "\n" + "-" * 50 + "\n")
            except IOError as e:
                print(f"Warning: Could not write to log file: {e}")
    
    def monitor(self):
        """Main monitoring loop."""
        print(f"Starting GPU memory monitoring (interval: {self.interval}s)")
        if self.log_file:
            print(f"Logging to: {self.log_file}")
        print("Press Ctrl+C to stop\n")
        
        while self.running:
            try:
                info = self.get_gpu_info()
                info_str = self.format_info(info)
                self.log_info(info_str)
                
                # Check for potential OOM conditions
                if info and info.get('utilization_percent', 0) > 95:
                    print("⚠️  WARNING: GPU memory usage > 95%! OOM risk!")
                    
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(self.interval)
        
        print("GPU monitoring stopped.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor GPU memory usage")
    parser.add_argument('--interval', type=int, default=5, 
                       help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--log-file', type=str, 
                       help='Optional log file path')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously (default: run once)')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(interval=args.interval, log_file=args.log_file)
    
    if args.continuous:
        monitor.monitor()
    else:
        # Run once
        info = monitor.get_gpu_info()
        info_str = monitor.format_info(info)
        monitor.log_info(info_str)

if __name__ == "__main__":
    main() 