import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from auto_follow.evaluation.evaluate_inference_time import benchmark_evaluators


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


def save_results(all_results, output_dir: str | Path = "benchmark_results"):
    """Save benchmark results in multiple formats"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

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


def run_detailed_benchmark_multi_eval(path_to_frames: str | Path, output_dir: str | Path, trials: int = 30):
    print("Starting comprehensive benchmark...")

    all_results = []

    print("\nStarting measured iterations...")
    for i in range(trials):
        print(f"\nRun {i + 1}/30")
        s_times, s_seg_times, i_times = benchmark_evaluators(path_to_frames)

        all_results.append({
            'run_number': i + 1,
            'student_times': s_times,
            'student_seg_times': s_seg_times,
            'ibvs_times': i_times
        })

    save_results(all_results, output_dir=output_dir)


if __name__ == '__main__':
    path = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_02-01-23/frames"

    run_detailed_benchmark_multi_eval(path_to_frames=path, output_dir="./", trials=30)
