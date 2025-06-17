import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from auto_follow.evaluation.evaluation_methods import IBVSEvaluator, StudentEvaluator

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False



def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory():
    """Get GPU memory usage in MB"""
    if not GPU_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except:
        return None


def benchmark_evaluators(input_directory: str):
    """Simple benchmark comparing StudentEvaluator vs IBVSEvaluator"""

    input_path = Path(input_directory)
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    frames = []

    for img_path in sorted(image_files):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            frames.append(frame)

    print(f"Loaded {len(frames)} frames")

    print("\n=== Testing StudentEvaluator ===")
    gc.collect()

    initial_memory = get_memory_usage()
    initial_gpu_memory = get_gpu_memory()

    student_evaluator = StudentEvaluator()
    student_times = []

    model_memory = get_memory_usage()
    model_gpu_memory = get_gpu_memory()

    for frame in frames:
        start = time.perf_counter()
        result = student_evaluator.predict_command_on_frame(frame)
        end = time.perf_counter()
        student_times.append((end - start) * 1000)

    # Measure peak memory
    peak_memory = get_memory_usage()
    peak_gpu_memory = get_gpu_memory()

    student_memory_usage = {
        'model_load': model_memory - initial_memory,
        'peak_usage': peak_memory - initial_memory,
        'gpu_model': (model_gpu_memory - initial_gpu_memory) if initial_gpu_memory else None,
        'gpu_peak': (peak_gpu_memory - initial_gpu_memory) if initial_gpu_memory else None
    }

    del student_evaluator
    gc.collect()

    print("\n=== Testing IBVSEvaluator ===")

    initial_memory = get_memory_usage()
    initial_gpu_memory = get_gpu_memory()

    ibvs_evaluator = IBVSEvaluator()
    ibvs_times = []

    model_memory = get_memory_usage()
    model_gpu_memory = get_gpu_memory()

    for frame in frames:
        start = time.perf_counter()
        result = ibvs_evaluator.predict_command_on_frame(frame)
        end = time.perf_counter()
        ibvs_times.append((end - start) * 1000)

    peak_memory = get_memory_usage()
    peak_gpu_memory = get_gpu_memory()

    ibvs_memory_usage = {
        'model_load': model_memory - initial_memory,
        'peak_usage': peak_memory - initial_memory,
        'gpu_model': (model_gpu_memory - initial_gpu_memory) if initial_gpu_memory else None,
        'gpu_peak': (peak_gpu_memory - initial_gpu_memory) if initial_gpu_memory else None
    }

    del ibvs_evaluator
    gc.collect()

    print("\n=== TIMING RESULTS ===")
    print(f"StudentEvaluator:")
    print(f"  Average: {np.mean(student_times):.2f} ms")
    print(f"  Min: {np.min(student_times):.2f} ms")
    print(f"  Max: {np.max(student_times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(student_times):.1f}")

    print(f"\nIBVSEvaluator:")
    print(f"  Average: {np.mean(ibvs_times):.2f} ms")
    print(f"  Min: {np.min(ibvs_times):.2f} ms")
    print(f"  Max: {np.max(ibvs_times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(ibvs_times):.1f}")

    print("\n=== MEMORY RESULTS ===")
    print(f"StudentEvaluator:")
    print(f"  Model Load: {student_memory_usage['model_load']:.1f} MB RAM")
    print(f"  Peak Usage: {student_memory_usage['peak_usage']:.1f} MB RAM")
    if student_memory_usage['gpu_model']:
        print(f"  Model Load: {student_memory_usage['gpu_model']:.1f} MB GPU")
        print(f"  Peak Usage: {student_memory_usage['gpu_peak']:.1f} MB GPU")

    print(f"\nIBVSEvaluator:")
    print(f"  Model Load: {ibvs_memory_usage['model_load']:.1f} MB RAM")
    print(f"  Peak Usage: {ibvs_memory_usage['peak_usage']:.1f} MB RAM")
    if ibvs_memory_usage['gpu_model']:
        print(f"  Model Load: {ibvs_memory_usage['gpu_model']:.1f} MB GPU")
        print(f"  Peak Usage: {ibvs_memory_usage['gpu_peak']:.1f} MB GPU")

    faster = "StudentEvaluator" if np.mean(student_times) < np.mean(ibvs_times) else "IBVSEvaluator"
    speedup = max(np.mean(student_times), np.mean(ibvs_times)) / min(np.mean(student_times), np.mean(ibvs_times))
    print(f"\n{faster} is {speedup:.2f}x faster")


def main():
    path = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_02-01-23/frames"
    benchmark_evaluators(path)


if __name__ == '__main__':
    main()
