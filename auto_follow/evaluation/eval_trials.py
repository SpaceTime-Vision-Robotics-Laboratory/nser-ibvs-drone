import gc
import platform
import time
from pathlib import Path

import cv2
import pandas as pd

from auto_follow.evaluation.evaluation_methods import StudentEvaluator, IBVSEvaluator
from auto_follow.utils.path_manager import Paths


def run_single_evaluation(evaluator, frames, predict_fn, warmup=3):
    # Warm-up
    for frame in frames[:warmup]:
        _ = predict_fn(evaluator, frame)

    times = []
    for frame in frames:
        start = time.perf_counter()
        _ = predict_fn(evaluator, frame)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    return times


def predict_student(evaluator, frame):
    return evaluator.predict_command_on_frame(frame)


def predict_student_seg(evaluator, frame):
    return evaluator.predict_with_segmentation(frame)


def predict_ibvs(evaluator, frame):
    return evaluator.predict_command_on_frame(frame)


def benchmark_evaluators(input_directory: str, trials: int = 10, output_csv: str = "benchmark_results.csv"):
    input_path = Path(input_directory)
    image_files = sorted(list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')))
    frames = [cv2.imread(str(p)) for p in image_files if cv2.imread(str(p)) is not None]

    print(f"Loaded {len(frames)} frames")

    all_results = []

    for trial in range(trials):
        print(f"\n=== Trial {trial + 1}/{trials} ===")

        gc.collect()
        evaluator = StudentEvaluator()
        student_times = run_single_evaluation(evaluator, frames, predict_student)
        del evaluator
        gc.collect()

        evaluator_seg = StudentEvaluator(segmentation_model_path=Paths.SIM_CAR_IBVS_YOLO_PATH)
        student_times_seg = run_single_evaluation(evaluator_seg, frames, predict_student_seg)
        del evaluator_seg
        gc.collect()

        evaluator_ibvs = IBVSEvaluator()
        ibvs_times = run_single_evaluation(evaluator_ibvs, frames, predict_ibvs)
        del evaluator_ibvs
        gc.collect()

        all_results.extend([
            {"Trial": trial + 1, "Evaluator": "Student", "Times": t} for t in student_times
        ])
        all_results.extend([
            {"Trial": trial + 1, "Evaluator": "Student+Segmentation", "Times": t} for t in student_times_seg
        ])
        all_results.extend([
            {"Trial": trial + 1, "Evaluator": "IBVS", "Times": t} for t in ibvs_times
        ])

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved raw data to {output_csv}")

    # Compute statistics
    summary = df.groupby("Evaluator")["Times"].agg(
        Average="mean",
        StdDev="std",
        Median="median",
        Min="min",
        Max="max",
        FPS=lambda x: 1000 / x.mean()
    ).reset_index()

    print("\n=== Summary Table ===")
    print(summary.to_string(index=False, float_format="%.2f"))

    print("\n=== LaTeX Table (for paper) ===")
    print(summary.to_latex(index=False, float_format="%.2f", caption="Timing Comparison of Evaluators",
                           label="tab:timing_eval"))

    with open("benchmark_table.tex", "w") as f:
        f.write(summary.to_latex(index=False, float_format="%.2f", caption="Timing Comparison of Evaluators",
                                 label="tab:timing_eval"))

    print("\nSaved LaTeX table to benchmark_table.tex")

    print("\n=== System Info ===")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor: {platform.processor()}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


if __name__ == '__main__':
    path = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_02-01-23/frames"
    benchmark_evaluators(path, trials=30)
