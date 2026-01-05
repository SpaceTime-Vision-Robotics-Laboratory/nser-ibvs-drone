import time
import gc
import json
import csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from auto_follow.evaluation.evaluation_methods import StudentEvaluator, IBVSEvaluator
from auto_follow.utils.path_manager import Paths


def benchmark_evaluators(input_directory: str):
    """Benchmark comparing StudentEvaluator vs IBVSEvaluator"""

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

    student_evaluator = StudentEvaluator()
    student_times = []

    for frame in frames:
        start = time.perf_counter()
        _ = student_evaluator.predict_command_on_frame(frame)
        end = time.perf_counter()
        student_times.append((end - start) * 1000)  # ms

    del student_evaluator
    gc.collect()

    print("\n=== Testing StudentEvaluator with Segmentation ===")
    gc.collect()

    student_evaluator_seg = StudentEvaluator(segmentation_model_path=Paths.SIM_CAR_IBVS_YOLO_PATH)
    student_times_seg = []

    for frame in frames:
        start = time.perf_counter()
        _ = student_evaluator_seg.predict_with_segmentation(frame)
        end = time.perf_counter()
        student_times_seg.append((end - start) * 1000)  # ms

    del student_evaluator_seg
    gc.collect()

    print("\n=== Testing IBVSEvaluator ===")

    ibvs_evaluator = IBVSEvaluator()
    ibvs_times = []

    for frame in frames:
        start = time.perf_counter()
        _ = ibvs_evaluator.predict_command_on_frame(frame)
        end = time.perf_counter()
        ibvs_times.append((end - start) * 1000)  # ms

    del ibvs_evaluator
    gc.collect()

    print("\n=== RESULTS ===")
    print("StudentEvaluator:")
    print(f"  Average: {np.mean(student_times):.2f} ms")
    print(f"  Median: {np.median(student_times):.2f} ms")
    print(f"  Min: {np.min(student_times):.2f} ms")
    print(f"  Max: {np.max(student_times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(student_times):.1f}")

    print("StudentEvaluator Segmentation:")
    print(f"  Average: {np.mean(student_times_seg):.2f} ms")
    print(f"  Median: {np.median(student_times_seg):.2f} ms")
    print(f"  Min: {np.min(student_times_seg):.2f} ms")
    print(f"  Max: {np.max(student_times_seg):.2f} ms")
    print(f"  FPS: {1000 / np.mean(student_times_seg):.1f}")

    print("\nIBVSEvaluator:")
    print(f"  Average: {np.mean(ibvs_times):.2f} ms")
    print(f"  Median: {np.median(ibvs_times):.2f} ms")
    print(f"  Min: {np.min(ibvs_times):.2f} ms")
    print(f"  Max: {np.max(ibvs_times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(ibvs_times):.1f}")

    faster = "StudentEvaluator" if np.mean(student_times) < np.mean(ibvs_times) else "IBVSEvaluator"
    faster_seg = "StudentEvaluatorSegmentation" if np.mean(student_times_seg) < np.mean(ibvs_times) else "IBVSEvaluator"
    speedup = max(np.mean(student_times), np.mean(ibvs_times)) / min(np.mean(student_times), np.mean(ibvs_times))
    speedup_seg = max(np.mean(student_times_seg), np.mean(ibvs_times)) / min(np.mean(student_times_seg),
                                                                             np.mean(ibvs_times))
    print(f"\n{faster} is {speedup:.2f}x faster")
    print(f"\n{faster_seg} is {speedup_seg:.2f}x faster")

    return student_times, student_times_seg, ibvs_times


def calculate_stats(times):
    """Calculate comprehensive statistics for timing data"""
    return {
        'mean': float(np.mean(times)),
        'median': float(np.median(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'std': float(np.std(times)),
        'fps': float(1000 / np.mean(times))
    }


def save_results(all_results, output_dir="benchmark_results"):
    """Save benchmark results in multiple formats"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    student_all = [time for run in all_results for time in run['student_times']]
    student_seg_all = [time for run in all_results for time in run['student_seg_times']]
    ibvs_all = [time for run in all_results for time in run['ibvs_times']]

    aggregate_stats = {
        'student_evaluator': calculate_stats(student_all),
        'student_evaluator_segmentation': calculate_stats(student_seg_all),
        'ibvs_evaluator': calculate_stats(ibvs_all),
        'total_runs': len(all_results),
        'frames_per_run': len(all_results[0]['student_times']) if all_results else 0
    }

    student_vs_ibvs_speedup = max(aggregate_stats['student_evaluator']['mean'],
                                  aggregate_stats['ibvs_evaluator']['mean']) / \
                              min(aggregate_stats['student_evaluator']['mean'],
                                  aggregate_stats['ibvs_evaluator']['mean'])

    student_seg_vs_ibvs_speedup = max(aggregate_stats['student_evaluator_segmentation']['mean'],
                                      aggregate_stats['ibvs_evaluator']['mean']) / \
                                  min(aggregate_stats['student_evaluator_segmentation']['mean'],
                                      aggregate_stats['ibvs_evaluator']['mean'])

    aggregate_stats['speedups'] = {
        'student_vs_ibvs': float(student_vs_ibvs_speedup),
        'student_seg_vs_ibvs': float(student_seg_vs_ibvs_speedup),
        'faster_method_student_vs_ibvs': 'StudentEvaluator' if aggregate_stats['student_evaluator']['mean'] <
                                                               aggregate_stats['ibvs_evaluator'][
                                                                   'mean'] else 'IBVSEvaluator',
        'faster_method_student_seg_vs_ibvs': 'StudentEvaluatorSegmentation' if
        aggregate_stats['student_evaluator_segmentation']['mean'] < aggregate_stats['ibvs_evaluator'][
            'mean'] else 'IBVSEvaluator'
    }

    json_path = output_path / f"benchmark_detailed_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'aggregate_statistics': aggregate_stats,
            'individual_runs': all_results,
            'timestamp': timestamp
        }, f, indent=2)

    csv_path = output_path / f"benchmark_summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Mean(ms)', 'Median(ms)', 'Min(ms)', 'Max(ms)', 'Std(ms)', 'FPS'])

        for method, stats in [
            ('StudentEvaluator', aggregate_stats['student_evaluator']),
            ('StudentEvaluator_Segmentation', aggregate_stats['student_evaluator_segmentation']),
            ('IBVSEvaluator', aggregate_stats['ibvs_evaluator'])
        ]:
            writer.writerow([
                method,
                f"{stats['mean']:.2f}",
                f"{stats['median']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['fps']:.1f}"
            ])

    per_run_csv_path = output_path / f"benchmark_per_run_{timestamp}.csv"
    with open(per_run_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Run',
            'Student_Mean(ms)', 'Student_Median(ms)', 'Student_Min(ms)', 'Student_Max(ms)', 'Student_FPS',
            'StudentSeg_Mean(ms)', 'StudentSeg_Median(ms)', 'StudentSeg_Min(ms)', 'StudentSeg_Max(ms)',
            'StudentSeg_FPS',
            'IBVS_Mean(ms)', 'IBVS_Median(ms)', 'IBVS_Min(ms)', 'IBVS_Max(ms)', 'IBVS_FPS'
        ])

        for i, run in enumerate(all_results):
            student_stats = calculate_stats(run['student_times'])
            student_seg_stats = calculate_stats(run['student_seg_times'])
            ibvs_stats = calculate_stats(run['ibvs_times'])

            writer.writerow([
                i + 1,
                f"{student_stats['mean']:.2f}", f"{student_stats['median']:.2f}",
                f"{student_stats['min']:.2f}", f"{student_stats['max']:.2f}", f"{student_stats['fps']:.1f}",
                f"{student_seg_stats['mean']:.2f}", f"{student_seg_stats['median']:.2f}",
                f"{student_seg_stats['min']:.2f}", f"{student_seg_stats['max']:.2f}", f"{student_seg_stats['fps']:.1f}",
                f"{ibvs_stats['mean']:.2f}", f"{ibvs_stats['median']:.2f}",
                f"{ibvs_stats['min']:.2f}", f"{ibvs_stats['max']:.2f}", f"{ibvs_stats['fps']:.1f}"
            ])

    print("\n=== SAVED RESULTS ===")
    print(f"Detailed results: {json_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Per-run CSV: {per_run_csv_path}")

    print(f"\n=== AGGREGATE STATISTICS ({len(all_results)} runs) ===")
    for method, stats in [
        ('StudentEvaluator', aggregate_stats['student_evaluator']),
        ('StudentEvaluator Segmentation', aggregate_stats['student_evaluator_segmentation']),
        ('IBVSEvaluator', aggregate_stats['ibvs_evaluator'])
    ]:
        print(f"{method}:")
        print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f} ms")
        print(f"  Median: {stats['median']:.2f} ms")
        print(f"  Min: {stats['min']:.2f} ms")
        print(f"  Max: {stats['max']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print()

    print("Speedups:")
    print(
        f"  {aggregate_stats['speedups']['faster_method_student_vs_ibvs']} is {aggregate_stats['speedups']['student_vs_ibvs']:.2f}x faster than the other")
    print(
        f"  {aggregate_stats['speedups']['faster_method_student_seg_vs_ibvs']} is {aggregate_stats['speedups']['student_seg_vs_ibvs']:.2f}x faster than IBVSEvaluator")


if __name__ == '__main__':
    path = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_02-01-23/frames"

    print("Starting comprehensive benchmark...")

    all_results = []

    print("\nStarting measured iterations...")
    for i in range(30):
        print(f"\nRun {i + 1}/30")
        s_times, s_seg_times, i_times = benchmark_evaluators(path)

        all_results.append({
            'run_number': i + 1,
            'student_times': s_times,
            'student_seg_times': s_seg_times,
            'ibvs_times': i_times
        })

    save_results(all_results)
