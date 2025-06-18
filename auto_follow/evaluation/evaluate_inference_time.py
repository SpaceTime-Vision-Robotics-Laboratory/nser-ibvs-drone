import time
import gc
from pathlib import Path
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
        # result = student_evaluator.predict_with_segmentation(frame)
        result = student_evaluator.predict_command_on_frame(frame)
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
        result = student_evaluator_seg.predict_with_segmentation(frame)
        end = time.perf_counter()
        student_times_seg.append((end - start) * 1000)  # ms

    del student_evaluator_seg
    gc.collect()

    print("\n=== Testing IBVSEvaluator ===")

    ibvs_evaluator = IBVSEvaluator()
    ibvs_times = []

    for frame in frames:
        start = time.perf_counter()
        result = ibvs_evaluator.predict_command_on_frame(frame)
        end = time.perf_counter()
        ibvs_times.append((end - start) * 1000)  # ms

    del ibvs_evaluator
    gc.collect()

    print("\n=== RESULTS ===")
    print(f"StudentEvaluator:")
    print(f"  Average: {np.mean(student_times):.2f} ms")
    print(f"  Min: {np.min(student_times):.2f} ms")
    print(f"  Max: {np.max(student_times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(student_times):.1f}")

    print(f"StudentEvaluator Segmentation:")
    print(f"  Average: {np.mean(student_times_seg):.2f} ms")
    print(f"  Min: {np.min(student_times_seg):.2f} ms")
    print(f"  Max: {np.max(student_times_seg):.2f} ms")
    print(f"  FPS: {1000 / np.mean(student_times_seg):.1f}")

    print(f"\nIBVSEvaluator:")
    print(f"  Average: {np.mean(ibvs_times):.2f} ms")
    print(f"  Min: {np.min(ibvs_times):.2f} ms")
    print(f"  Max: {np.max(ibvs_times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(ibvs_times):.1f}")

    faster = "StudentEvaluator" if np.mean(student_times) < np.mean(ibvs_times) else "IBVSEvaluator"
    faster_seg = "StudentEvaluatorSegmentation" if np.mean(student_times_seg) < np.mean(ibvs_times) else "IBVSEvaluator"
    speedup = max(np.mean(student_times), np.mean(ibvs_times)) / min(np.mean(student_times), np.mean(ibvs_times))
    speedup_seg = max(np.mean(student_times_seg), np.mean(ibvs_times)) / min(np.mean(student_times_seg), np.mean(ibvs_times))
    print(f"\n{faster} is {speedup:.2f}x faster")
    print(f"\n{faster_seg} is {speedup_seg:.2f}x faster")


if __name__ == '__main__':
    path = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_02-01-23/frames"

    benchmark_evaluators(path)
