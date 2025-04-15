from pathlib import Path
import sys
import time
# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

import numpy as np
import cv2
from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController
from drone_base.main.stream.base_video_processor import BaseVideoProcessor
from ultralytics import YOLO

# Camera Parameters
# camera_fx = 465.6
# camera_fy = 465.6
camera_fx = 465
camera_fy = 348
camera_cx = 320.0
camera_cy = 180.0

lambda_gain = 0.1

# Target Parameters
X1Y1 = (220, 50)
X2Y2 = (400, 310)
X1Y2 = (220, 310)
X2Y1 = (400, 50)

class IBVSController(BaseStreamingController):
    def __init__(self, lambda_gain:float = lambda_gain, **kwargs):
        super().__init__(**kwargs)
        self.lambda_gain = lambda_gain
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = camera_cx
        self.cy = camera_cy
        
        
    def initialize_position(self, takeoff_height=2.0, gimbal_angle=-45, back_distance=-2.0, left_distance=0.0):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
            
        print("Taking off...")
        
        self.drone_commander.tilt_camera(pitch_deg=gimbal_angle, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        
        self.drone_commander.take_off()
        time.sleep(1)
        
        print(f"Moving to initial position...")
        self.drone_commander.move_by(forward=back_distance, right=0, down=-takeoff_height, rotation=0)
        time.sleep(1)
        
        print(f"Starting Program...")

class IBVSProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str = "/home/sebnae/shared_drive/ws/drone_ws/auto-follow/models/yolov11n_best_car_simulator.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = YOLO(model_path)
        
    def estimate_depth(self, current_features:list[np.ndarray], target_features:list[np.ndarray]) -> float:
        return 2.0
    
    def calculate_interaction_matrix(self, u, v, Z = 6.5):
        x = (u - camera_cx) / camera_fx
        y = (v - camera_cy) / camera_fy
        
        return np.array([
           [-1/Z, 0, x/Z, x*y, -(1+x**2), y],
           [0, -1/Z, y/Z, 1+y**2, -x*y, -x],
        ])
    
    def compute_control_law(self, error_vector:np.ndarray, interaction_matrix:np.ndarray) -> np.ndarray:
        # u = -lambda * J+^T * e
        # J+ = (J^T * J)^-1 * J^T
        L = interaction_matrix
        L_pseudo_inv = np.linalg.inv(L.T @ L) @ L.T
        u = -lambda_gain * L_pseudo_inv @ error_vector
        return u
        

    def _process_frame(self, frame: np.ndarray) -> list[np.ndarray, list[np.ndarray]]:
        results = self.detector.predict(frame, stream=True, verbose=False)
        first_result = next(results, None)
        plotted_frame = first_result.plot()
        
        x1, y1, x2, y2 = 0, 0, 0, 0
        # Extract bounding boxes and draw center dot
        if first_result.boxes and first_result.boxes.xyxy is not None:
            for box in first_result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(plotted_frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Green dot
        
        # On the frame draw a circle for the center of the image
        cv2.circle(plotted_frame, (plotted_frame.shape[1] // 2, plotted_frame.shape[0] // 2), 5, (255, 0, 0), -1)
        
        # On the frame draw a rectangle for the target
        cv2.rectangle(plotted_frame, X1Y1, X2Y2, (0, 0, 255), 2)
        
        # Get the current features which are the corners of the bounding box from the first result
        current_features = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        
        # Get the target features which are the corners of the bounding box
        target_features = np.array([X1Y1, X2Y1, X2Y2, X1Y2], dtype=np.float32)
        # print(f"Current Features: {current_features}")
        # print(f"Target Features: {target_features}")
        
        # Get the error vector
        error_vector = current_features.flatten().T - target_features.flatten().T
        # print(f"Error Vector: {error_vector}")
        
        L_full = np.zeros((len(current_features) * 2, 6))
        
        for i in range(len(current_features)):
            x, y = current_features[i]
            L = self.calculate_interaction_matrix(x, y, Z=2.0)
            L_full[i*2:i*2+2, :] = L
            
        # print(f"L_full: {L_full}")
        
        # Compute the control law
        control_law_camera_frame = self.compute_control_law(error_vector, L_full)
        # print(f"Control Law: {control_law}")
        
        # Convert the control law to the drone frame
        # Def rotation matrix is with respect to the camera frame
        R_matrix = np.array([
            [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
            [0, 1, 0],
            [1/np.sqrt(2), 0, -1/np.sqrt(2)]
        ])
        # control_law_drone_frame = R_matrix @ control_law_camera_frame[3:]
        
        vx, vy, vz, wx, wy, wz = control_law_camera_frame
        
        # Linear scaler
        k_v = 1
        # Angular scaler
        k_w = 1
        
        # print(f"Control Law: Vx: {vx}, Vy: {vy}, Vz: {vz}, Wx: {wx}, Wy: {wy}, Wz: {wz}")
        MAX_COMMAND = 30
        
        roll = int(np.clip(vx, -MAX_COMMAND, MAX_COMMAND))
        pitch = int(np.clip(vy, -MAX_COMMAND, MAX_COMMAND))
        gaz = int(np.clip(vz, -MAX_COMMAND, MAX_COMMAND))
        yaw_rate = int(np.clip(wx, -MAX_COMMAND, MAX_COMMAND))
        
        # print(f"Command: SIDE: {vx}, FWD: {vy}, UP: {vz}, ROT: {wx}")
        print(f'Command: ROLL: {roll}, PITCH: {pitch}, GAZ: {gaz}, YAW_RATE: {yaw_rate}')
        self.drone_commander.piloting(x=0, y=-pitch, z=0, z_rot=yaw_rate, dt=0.1)
        
        # print(f"Command: SIDE: {vx}, FWD: {vy}, UP: {vz}, ROT: {wz}")
        
        # self.drone_commander.piloting(x=vx, y=-vy, z=vz, z_rot=wz, dt=0.1)
        
        # Write the control law on the frame
        # cv2.putText(plotted_frame, f"Control Law: {control_law}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return plotted_frame
    
    
    
if __name__ == "__main__":
    controller = IBVSController(
        ip=DroneIp.SIMULATED,
        processor_class=IBVSProcessor,
        speed=35
    )
    
    controller.initialize_position()
    controller.run()
    