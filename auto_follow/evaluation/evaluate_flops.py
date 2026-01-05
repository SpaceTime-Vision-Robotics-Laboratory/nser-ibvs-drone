import copy
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

try:
    from thop import profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' library not found. TFLOPs will not be calculated.")
    print("Please install it using: pip install thop")

from auto_follow.evaluation.evaluation_methods import StudentEvaluator, IBVSEvaluator
from auto_follow.utils.path_manager import Paths


def extract_pytorch_model(obj):
    """
    Recursively attempts to find the underlying torch.nn.Module from wrappers.
    """
    if obj is None:
        return None

    if isinstance(obj, nn.Module):
        return obj

    if type(obj).__name__ == 'YOLO' and hasattr(obj, 'model'):
        return obj.model

    # Case: StudentEvaluator -> StudentEngine
    if hasattr(obj, 'student_engine'):
        return extract_pytorch_model(obj.student_engine)

    # Case: StudentEngine -> has .model
    if hasattr(obj, 'model'):
        if isinstance(obj.model, nn.Module):
            return obj.model
        return extract_pytorch_model(obj.model)

    # Case: IBVSEvaluator -> detector (MaskSplitterEngineIBVS)
    if hasattr(obj, 'detector'):
        return extract_pytorch_model(obj.detector)

    return None


def measure_complexity(model, input_size: tuple = (1, 3, 224, 224)):
    """
    Calculates FLOPs and Parameters for a PyTorch model.
    Uses a deepcopy to avoid modifying the actual model with thop hooks.
    """
    if model is None:
        return 0, 0

    model_class_name = type(model).__name__

    if hasattr(model, 'info') or model_class_name in {'DetectionModel', 'SegmentationModel', 'WorldModel'}:
        try:
            info_result = model.info(verbose=True)  # (layers, params, gradients, gflops)
            gflops = info_result[3]
            params = info_result[1]
            flops = gflops * 1e9
            return flops, params
        except Exception as e:
            print(f"  [Ultralytics profiling failed: {e}]")
            return 0, 0

    if not THOP_AVAILABLE:
        return 0, 0

    try:
        model_clone = copy.deepcopy(model)
        model_clone.eval()
    except Exception as e:
        print(f"  [Warning: Could not copy model for profiling: {e}. Skipping FLOPs count.]")
        return 0, 0

    try:
        device = next(model_clone.parameters()).device
    except StopIteration:
        return 0, 0

    dummy_input = torch.randn(*input_size).to(device)

    try:
        macs, params = profile(model_clone, inputs=(dummy_input,), verbose=False)
        flops = macs * 2
        del model_clone
        torch.cuda.empty_cache()

        return flops, params
    except Exception as e:
        print(f"  [Error calculating FLOPs for {type(model).__name__}: {e}]")
        return 0, 0


def benchmark_evaluators(input_directory: str):
    """Benchmark comparing StudentEvaluator vs IBVSEvaluator with TFLOPs"""
    input_path = Path(input_directory)
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    frames = []
    for img_path in sorted(image_files):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            frames.append(frame)

    if not frames:
        print("No frames found!")
        return [], [], []

    print(f"Loaded {len(frames)} frames")

    # --- StudentEvaluator ---
    print("\n=== Testing StudentEvaluator ===")
    gc.collect()
    student_evaluator = StudentEvaluator()

    student_model = extract_pytorch_model(student_evaluator)
    st_input_size = (1, 3, 224, 224)
    if hasattr(student_evaluator, 'student_engine') and hasattr(student_evaluator.student_engine, 'image_size'):
        h, w = student_evaluator.student_engine.image_size
        st_input_size = (1, 3, h, w)
    st_flops, st_params = measure_complexity(student_model, st_input_size)

    student_times = []
    for frame in frames:
        start = time.perf_counter()
        _ = student_evaluator.predict_command_on_frame(frame)
        end = time.perf_counter()
        student_times.append((end - start) * 1000)  # ms
    del student_evaluator
    gc.collect()

    # --- StudentEvaluator with Segmentation ---
    print("\n=== Testing StudentEvaluator with Segmentation ===")
    gc.collect()
    student_evaluator_seg = StudentEvaluator(segmentation_model_path=Paths.SIM_CAR_IBVS_YOLO_PATH)

    # This one has TWO models: The Student Policy + The YOLO Detector
    # A. Student Model
    pol_model = extract_pytorch_model(student_evaluator_seg.student_engine)
    pol_flops, pol_params = measure_complexity(pol_model, st_input_size)
    # B. YOLO Model (inside detector)
    det_model = extract_pytorch_model(student_evaluator_seg.detector)
    yolo_input_size = (1, 3, 640, 640)
    det_flops, det_params = measure_complexity(det_model, yolo_input_size)

    seg_flops = pol_flops + det_flops
    seg_params = pol_params + det_params

    student_times_seg = []
    for frame in frames:
        start = time.perf_counter()
        _ = student_evaluator_seg.predict_with_segmentation(frame)
        end = time.perf_counter()
        student_times_seg.append((end - start) * 1000)  # ms
    del student_evaluator_seg
    gc.collect()

    # --- IBVSEvaluator ---
    print("\n=== Testing IBVSEvaluator ===")
    ibvs_evaluator = IBVSEvaluator()

    # IBVS uses MaskSplitterEngineIBVS -> YOLO seg model + Splitter Network
    # Note: IBVS logic itself is negligible in FLOPs compared to the NN.
    # A. YOLO seg model
    ibvs_model = extract_pytorch_model(ibvs_evaluator)
    ibvs_flops, ibvs_params = measure_complexity(ibvs_model, yolo_input_size)

    # B. Mask Splitter model
    split_image_size = (1, 4, 360, 640)
    ibvs_split_model = extract_pytorch_model(ibvs_evaluator.detector.splitter_model)
    ibvs_split_flops, ibvs_split_params = measure_complexity(ibvs_split_model, split_image_size)

    ibvs_flops = ibvs_flops + ibvs_split_flops
    ibvs_params = ibvs_params + ibvs_split_params

    ibvs_times = []
    for frame in frames:
        start = time.perf_counter()
        _ = ibvs_evaluator.predict_command_on_frame(frame)
        end = time.perf_counter()
        ibvs_times.append((end - start) * 1000)  # ms
    del ibvs_evaluator
    gc.collect()

    print("\n=== RESULTS ===")

    def print_stats(name, times, flops, params):
        avg_time = np.mean(times)
        fps = 1000 / avg_time

        print(f"{name}:")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  FPS:     {fps:.1f}")

        if flops > 0:
            tflops = flops / 1e12
            gflops = flops / 1e9
            print(f"  GFLOPs:  {gflops:.4f}")
            print(f"  TFLOPs:  {tflops:.6f}")
            print(f"  Params:  {params / 1e6:.2f} M")
        else:
            print("  TFLOPs:  N/A (Model not found or error in profiling)")

    print_stats("StudentEvaluator (Policy Only)", student_times, st_flops, st_params)
    print_stats("StudentEvaluator + Segmentation", student_times_seg, seg_flops, seg_params)
    print_stats("IBVSEvaluator (Segmentation + Control)", ibvs_times, ibvs_flops, ibvs_params)

    return student_times, student_times_seg, ibvs_times


if __name__ == '__main__':
    path = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output/bunker-online-4k-config-test-front-small-offset-right/results/2025-11-25_15-33-27/frames"

    benchmark_evaluators(path)
