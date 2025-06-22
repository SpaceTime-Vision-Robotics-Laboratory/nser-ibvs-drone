import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import re

class Scene:
    def __init__(self, scene_path: Path):
        self.scene_path = scene_path
        self.scene_name = scene_path.name
        
        # Updated to use parquet-logs directory
        parquet_log_path = self.scene_path / "parquet-logs"
        frames_path = self.scene_path / "results"
        
        self.runs_data = {}
        
        # Get all timestamp folders
        timestamp_folders_parquet = [folder for folder in parquet_log_path.iterdir() if folder.is_dir()]
        timestamp_folders_frames = [folder for folder in frames_path.iterdir() if folder.is_dir()]
        
        # Match timestamp folders
        for timestamp_folder_parquet in timestamp_folders_parquet:
            timestamp = timestamp_folder_parquet.name
            
            # Find corresponding frames folder
            timestamp_folder_frames = frames_path / timestamp
            if not timestamp_folder_frames.exists():
                print(f"Warning: No frames folder found for timestamp {timestamp}")
                continue
                
            parquet_file = timestamp_folder_parquet / "logs.parquet"
            frames_folder = timestamp_folder_frames / "frames"
            
            if parquet_file.exists() and frames_folder.exists():
                # Read parquet data
                try:
                    df = pd.read_parquet(parquet_file)
                    frames = list(frames_folder.glob("*.jpg"))
                    
                    self.runs_data[timestamp] = {
                        'dataframe': df,
                        'frames_folder': frames_folder,
                        'frames': frames
                    }
                    
                    print(f"Timestamp {timestamp}: {len(df)} data points, {len(frames)} frames")
                except Exception as e:
                    print(f"Error processing {timestamp}: {e}")
            
        print(f"Scene {self.scene_name} has {len(self.runs_data)} valid runs")
    
    def process_to_output(self, output_train_path: Path, output_val_path: Path):
        """Process scene data to create the desired output structure."""
        
        # Sort timestamps to ensure consistent splitting
        sorted_timestamps = sorted(self.runs_data.keys())
        
        # Split timestamps into train and validation
        train_timestamps = sorted_timestamps[:-2]  # All except last 2
        val_timestamps = sorted_timestamps[-2:]    # Last 2
        
        # Process train runs
        for timestamp in tqdm(train_timestamps, desc=f"Processing {self.scene_name} - Train"):
            scene_train_path = output_train_path / self.scene_name
            self._process_single_run(timestamp, self.runs_data[timestamp], scene_train_path)
        
        # Process validation runs
        for timestamp in tqdm(val_timestamps, desc=f"Processing {self.scene_name} - Validation"):
            scene_val_path = output_val_path / self.scene_name
            self._process_single_run(timestamp, self.runs_data[timestamp], scene_val_path)

    def _process_single_run(self, timestamp: str, run_data: dict, output_base_path: Path):
        """Helper method to process a single run."""
        df = run_data['dataframe']
        frames_folder = run_data['frames_folder']
        
        # Create run output directory
        run_output_path = output_base_path / f"RUN_{timestamp}"
        images_path = run_output_path / "images"
        labels_path = run_output_path / "labels"
        
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        
        # Create mapping of frame indices to frame files
        frame_mapping = {}
        for frame_file in frames_folder.glob("*.jpg"):
            match = re.search(r'frame_(\d+)_', frame_file.name)
            if match:
                frame_idx = int(match.group(1))
                frame_mapping[frame_idx] = frame_file
        
        processed_count = 0
        
        # Process each row in the dataframe
        for _, row in df.iterrows():
            frame_idx = row['frame_idx']
            
            # Check if we have the corresponding frame
            if frame_idx in frame_mapping:
                source_frame = frame_mapping[frame_idx]
                
                # Copy image with new name
                target_image = images_path / f"image_frame_{frame_idx:06d}.jpg"
                shutil.copy2(source_frame, target_image)
                
                # Create txt file with command data
                target_label = labels_path / f"image_frame_{frame_idx:06d}.txt"
                with open(target_label, 'w') as f:
                    f.write(f"{row['x_cmd']} {row['y_cmd']} {row['rot_cmd']}\n")
                
                processed_count += 1
            else:
                print(f"Warning: Frame {frame_idx} not found in files for timestamp {timestamp}")
        
        print(f"  Processed {processed_count} frame-data pairs for run {timestamp}")


def main(args):
    raw_data_path = Path(args.raw_data_path)
    output_path = Path(args.output_path)
    
    # Create train and validation directories
    train_path = output_path / "train"
    val_path = output_path / "validation"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Each folder in the raw_data folder is a scene that needs to be processed
    for scene_path in raw_data_path.glob("*"):
        if scene_path.is_dir():
            scene_name = scene_path.name
            print(f"Processing scene: {scene_name}")
            
            # Create scene object and process it
            scene = Scene(scene_path)
            
            # Process scene to create desired output structure
            scene.process_to_output(train_path, val_path)
            
            print(f"Completed processing scene: {scene_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform drone dataset to desired structure")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw data directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()
    main(args)