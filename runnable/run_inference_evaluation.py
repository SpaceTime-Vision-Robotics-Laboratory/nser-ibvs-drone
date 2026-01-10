import argparse

from auto_follow.evaluation.eval_inference_multi_trials import benchmark_infer_multi_evaluators
from auto_follow.evaluation.eval_inference_multi_trials_detailed import run_detailed_benchmark_multi_eval
from auto_follow.evaluation.evaluate_flops import benchmark_flops_evaluators
from auto_follow.evaluation.evaluation_memory import benchmark_memory_evaluators


def main():
    parser = argparse.ArgumentParser(description="Run simulator experiments suite on 8 available scenes.")
    parser.add_argument(
        "--frames_dir", type=str,
        help="Path to where frames are stored on your machine."
    )
    parser.add_argument(
        "--target_runs", type=int, default=30,
        help="Number of times to run an evaluation on a scene."
    )
    parser.add_argument(
        "--save_dir", type=str, default="./",
        help="Path to where to save the evaluation results."
    )
    args = parser.parse_args()
    benchmark_infer_multi_evaluators(args.frames_dir, trials=args.target_runs, output_dir=args.save_dir)
    run_detailed_benchmark_multi_eval(args.frames_dir, output_dir=args.save_dir, trials=args.target_runs)
    benchmark_flops_evaluators(args.frames_dir, output_dir=args.save_dir)
    benchmark_memory_evaluators(args.frames_dir, output_dir=args.save_dir)

if __name__ == '__main__':
    main()