from pathlib import Path
import sys
import time
import argparse
import numpy as np
import cv2
import ultralytics

# Add parent directory to the path to make modules visible
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController
from drone_base.main.stream.base_video_processor import BaseVideoProcessor
from collections import deque


class PoseIBVSController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_position(self, takeoff_height=2.0, gimbal_angle=-45, back_distance=3.0, left_distance=0.0):
        """
        Initialize the drone position to start following the car
        - Takes off
        - Moves to the specified height
        - Positions the drone behind the car
        - Sets the gimbal angle for optimal viewing
        """
        # Connect to the drone if not already connected
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()

        # Take off
        print("Taking off...")
        self.drone_commander.take_off()
        time.sleep(5)  # Wait for stable takeoff

        # Tilt gimbal to the proper angle for car tracking
        print(f"Tilting gimbal to {gimbal_angle} degrees...")
        self.drone_commander.tilt_camera(
            pitch_deg=gimbal_angle,
            control_mode=GimbalType.MODE_POSITION,
            reference_type=GimbalType.REF_ABSOLUTE
        )
        time.sleep(2)  # Wait for gimbal to stabilize
        
        # Position drone at the right height and distance behind car
        print(f"Moving to initial tracking position: height={takeoff_height}m, behind={back_distance}m, left={left_distance}m")
        self.drone_commander.move_by(
            forward=-back_distance,  # Negative = move backward
            right=-left_distance,    # Negative = move left 
            down=-takeoff_height,    # Negative = move up
            rotation=0
        )
        time.sleep(4)  # Wait for movement to complete
        
        # Clear frame queue to ensure fresh frames for tracking
        self.frame_processor.frame_queue.empty()
        print("Initialization complete. Ready for tracking.")


class PoseIBVSProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | None = "models/yolov11n_best_car_simulator.pt", **kwargs):
        """
        Initializes the IBVS processor for car pose-based following
        
        Args:
            model_path: Path to the detection model (defaults to car detection model)
        """
        # Load the detector model for finding the car
        self.detector = ultralytics.YOLO(model_path)
        
        # Number of features to use for control (5 for bbox: corners + center)
        self.num_features = 5
        
        # Frame counter for selecting the reference frame
        self.detection_frame_count = 0
        self.target_reference_frame = 60  # Capture reference at the 60th detection
        
        # Target settings
        self.target_width_ratio = 0.3  # Target width as ratio of frame width
        self.target_features = None  # Will store the desired feature points (bbox corners + center)
        
        # Feature tracking variables
        self.reference_features = None
        self.reference_image = None
        self.reference_box = None
        self.last_valid_features = None
        
        # Camera parameters (will be set based on frame dimensions)
        self.fx = 465.60298  # Focal length x (default, will be updated)
        self.fy = 465.60298  # Focal length y (default, will be updated)
        self.cx = 320.0   # Principal point x (default, will be updated)
        self.cy = 180.0   # Principal point y (default, will be updated)
        
        # IBVS parameters
        self.lambda_gain = 0.5  # Control gain for IBVS
        self.depth_estimate = 3.0  # Initial depth estimate in meters
        self.depth_history = deque(maxlen=10)  # For smoothing depth estimates
        
        # Reference capture flag
        self.reference_captured = False
        
        # Initialize the time tracking
        self.last_command_time = time.time()
        
        # Initialize with parent class
        super().__init__(**kwargs)

    def _extract_reference(self, frame, box):
        """Extract and store reference information from bounding box coordinates"""
        x1, y1, x2, y2 = map(int, box)
        
        # Store reference box dimensions
        self.reference_box = box
        self.reference_box_width = x2 - x1
        self.reference_box_height = y2 - y1
        self.reference_center_x = (x1 + x2) / 2
        self.reference_center_y = (y1 + y2) / 2
        
        # Calculate and store reference aspect ratio (width/height)
        self.reference_aspect_ratio = self.reference_box_width / self.reference_box_height
        print(f"Reference aspect ratio: {self.reference_aspect_ratio:.3f} (width/height)")
        
        # Extract reference image (only for visualization)
        self.reference_image = frame.copy()
        
        # Extract the car region as reference (only for visualization)
        self.reference_roi = frame[y1:y2, x1:x2].copy()
        
        # Use bounding box corners and center as features
        self.reference_features = self._get_features_from_box(box)
        
        # Set target features to be the same as reference (we want to maintain these positions)
        self.target_features = self.reference_features.copy()
        self.reference_captured = True
        
        print(f"Reference captured at frame {self.detection_frame_count} using bounding box features")
        return True
    
    def _process_frame(self, frame: np.ndarray) -> list:
        """Process frame to detect car and extract pose features"""
        # Update camera parameters based on frame size if not set
        if self.cx == 320.0:  # Default value, meaning not initialized yet
            frame_height, frame_width = frame.shape[:2]
            self.cx = frame_width / 2
            self.cy = frame_height / 2
            self.fx = 465.60298
            self.fy = 465.60298
            print(f"Camera parameters set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        
        # Run detection
        results = self.detector.predict(frame, stream=False, verbose=False)
        processed_results = results[0]
        
        # If we see a car, increment the detection frame counter
        if processed_results.boxes and len(processed_results.boxes.conf) > 0:
            # Get the most confident detection
            best_conf_idx = processed_results.boxes.conf.argmax().item()
            coords = processed_results.boxes.xyxy[best_conf_idx].cpu().numpy()
            best_box = tuple(coords)  # (x1, y1, x2, y2)
            
            # Increment detection counter
            self.detection_frame_count += 1
            
            # Extract reference features at the target frame count
            if self.detection_frame_count == self.target_reference_frame and not self.reference_captured:
                print(f"Reached frame {self.detection_frame_count} - capturing reference image")
                self._extract_reference(frame, best_box)
            
            # Display counter if reference not yet captured
            if not self.reference_captured and self.detection_frame_count % 10 == 0:
                print(f"Detection frame count: {self.detection_frame_count}/{self.target_reference_frame}")
        
        return [frame, processed_results]

    def _get_features_from_box(self, box_coords):
        """Extract feature points from a bounding box (corners and center)"""
        x1, y1, x2, y2 = box_coords
        # Corners: Top-left, Top-right, Bottom-right, Bottom-left
        corners = np.array([
            [x1, y1],  # Top-left
            [x2, y1],  # Top-right
            [x2, y2],  # Bottom-right
            [x1, y2]   # Bottom-left
        ], dtype=np.float32)
        # Center
        center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
        # Stack corners and center (creates a 5x2 array)
        features = np.vstack((corners, center))
        return features
    
    def _compute_interaction_matrix(self, features_points, Z):
        """Compute the interaction matrix (Jacobian) for IBVS control"""
        if Z <= 0:  # Depth must be positive
            print("Warning: Assumed depth Z must be positive.")
            return None
        
        num_points = features_points.shape[0]  # Number of points (5 or more)
        L_s = np.zeros((2 * num_points, 6))  # Will be 10x6 or larger
        
        for i in range(num_points):
            u_pix = features_points[i, 0]  # Pixel coordinate u
            v_pix = features_points[i, 1]  # Pixel coordinate v
            
            # Rows of interaction matrix for this point
            L_point = np.array([
                [-self.fx / Z, 0, u_pix / Z, (u_pix * v_pix) / self.fx, -(self.fx**2 + u_pix**2) / self.fx, v_pix],
                [0, -self.fy / Z, v_pix / Z, (self.fy**2 + v_pix**2) / self.fy, -(u_pix * v_pix) / self.fy, -u_pix]
            ])
            L_s[2*i:2*i+2, :] = L_point
            
        return L_s
    
    def _estimate_depth(self, bbox_coords):
        """Estimate depth based on bounding box size and known parameters"""
        x1, y1, x2, y2 = bbox_coords
        box_width = x2 - x1
        
        # Approximate depth using the inverse relationship with bounding box width
        # Assuming the car has roughly known width in meters
        car_width_meters = 1.8  # Average car width
        depth_estimate = (car_width_meters * self.fx) / box_width
        
        # Apply temporal smoothing
        self.depth_history.append(depth_estimate)
        smoothed_depth = sum(self.depth_history) / len(self.depth_history)
        
        return smoothed_depth
    
    def _generate_follow_command(self, frame_dimensions, detection_results):
        """Generate drone control commands based on IBVS to maintain object pose"""
        # Initialize variables
        target_lost = True
        current_features = None
        frame_width, frame_height = frame_dimensions
        
        # Default commands (hover)
        x_cmd, y_cmd, z_cmd, yaw_cmd = 0, 0, 0, 0
        status = "Hovering"
        
        # Motion tracking variables
        current_time = time.time()
        dt = current_time - self.last_command_time
        self.last_command_time = current_time
        
        # Process detection results
        if detection_results and detection_results.boxes and len(detection_results.boxes.conf) > 0:
            # Find the most confident car detection
            best_conf_idx = -1
            best_conf = 0
            confidence_threshold = 0.4
            
            for i, conf in enumerate(detection_results.boxes.conf):
                conf_val = conf.item() if hasattr(conf, 'item') else conf
                if conf_val > confidence_threshold and conf_val > best_conf:
                    best_conf = conf_val
                    best_conf_idx = i
            
            # Process the best detection if found
            if best_conf_idx >= 0:
                target_lost = False
                coords = detection_results.boxes.xyxy[best_conf_idx].cpu().numpy()
                best_box = tuple(coords)  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = best_box
                
                # Calculate box center and dimensions
                current_center_x = (x1 + x2) / 2
                current_center_y = (y1 + y2) / 2
                current_width = x2 - x1
                current_height = y2 - y1
                
                # Calculate horizontal offset from center (for debugging)
                x_offset = (current_center_x - self.cx) / (frame_width / 2)
                
                # Track motion if we have a previous position
                if hasattr(self, 'prev_center_x') and dt > 0:
                    # Calculate velocity in pixels per second
                    vx = (current_center_x - self.prev_center_x) / dt
                    vy = (current_center_y - self.prev_center_y) / dt
                    self.pixel_vx = vx
                    self.pixel_vy = vy
                else:
                    # Initialize velocity
                    self.pixel_vx = 0
                    self.pixel_vy = 0
                
                # Store current position for next frame
                self.prev_center_x = current_center_x
                self.prev_center_y = current_center_y
                
                # Extract current features from bounding box
                current_features = self._get_features_from_box(best_box)
                
                # Update depth estimate based on box size
                self.depth_estimate = self._estimate_depth(best_box)
                
                # Store last valid features for display
                self.last_valid_features = current_features
        
        # If we have a reference and current features, apply IBVS control
        if self.reference_captured and not target_lost and current_features is not None:
            status = "IBVS Control"
            
            # Calculate feature error (current - reference)
            current_features_flat = current_features.flatten()
            reference_features_flat = self.reference_features.flatten()
            
            # Make sure the arrays have the same dimensions
            min_len = min(len(current_features_flat), len(reference_features_flat))
            error = current_features_flat[:min_len] - reference_features_flat[:min_len]
            
            # Compute interaction matrix
            L_s = self._compute_interaction_matrix(current_features, 4.5)
            
            if L_s is not None:
                try:
                    # Compute pseudoinverse with damping
                    damping = 0.01
                    L_s_pinv = np.linalg.pinv(L_s, rcond=damping)
                    
                    # Compute camera velocity: vc = -lambda * L_s^+ * e
                    gain = 0.4  # Control gain
                    camera_velocity = -gain * L_s_pinv @ error
                    
                    # Extract components
                    v_x, v_y, v_z, w_x, w_y, w_z = camera_velocity
                    
                    # Convert camera velocities to drone commands
                    # Scale factors to convert velocity to command values
                    lateral_gain = 20.0
                    forward_gain = 20.0
                    yaw_gain = 20.0
                    
                    # Apply the IBVS control directly
                    x_cmd = int(lateral_gain * v_y)  # v_y maps to lateral drone movement (x)
                    y_cmd = int(forward_gain * v_x)  # v_x maps to forward drone movement (y)
                    z_cmd = 0  # Keep altitude fixed
                    yaw_cmd = int(yaw_gain * w_z)  # w_z maps to yaw rotation
                    
                    status = "Pure IBVS Control"
                    
                except np.linalg.LinAlgError:
                    print("Warning: Matrix inverse calculation failed. Using simple centering.")
                    # Fallback to simple centering if matrix inverse fails
                    x_offset = (current_center_x - self.cx) / (frame_width / 2)
                    yaw_cmd = int(-30.0 * x_offset)
                    x_cmd = int(15.0 * x_offset)
            else:
                print("Warning: Invalid interaction matrix.")
                status = "Matrix Error"
        
        # Apply deadzone to prevent micro-movements
        deadzone = 1  # Low deadzone for responsive movement
        if abs(y_cmd) < deadzone:
            y_cmd = 0
        if abs(yaw_cmd) < deadzone:
            yaw_cmd = 0
        if abs(x_cmd) < deadzone:
            x_cmd = 0
            
        # Clip commands to safe ranges
        x_cmd = np.clip(x_cmd, -30, 30)
        y_cmd = np.clip(y_cmd, -30, 30)
        z_cmd = 0
        yaw_cmd = np.clip(yaw_cmd, -40, 40)
            
        # Send command to the drone
        if hasattr(self, 'drone_commander'):
            self.drone_commander.piloting(
                x=0,        # Right/Left
                y=y_cmd,    # Forward/Backward
                z=0,        # Up/Down
                z_rot=x_cmd,  # Yaw rotation
                dt=0.1      # Control interval
            )
        
        # Display status
        print(f"Status: {status} | Commands: X={x_cmd}, Y={y_cmd}, Z={z_cmd}, Yaw={yaw_cmd} | Depth: {self.depth_estimate:.2f}m")
        if hasattr(self, 'pixel_vx') and hasattr(self, 'pixel_vy'):
            print(f"Motion: Vx={self.pixel_vx:.1f} px/s, Vy={self.pixel_vy:.1f} px/s")
        
        return [x_cmd, y_cmd, z_cmd, yaw_cmd]
        
    def _display_frame(self, frame_data: list) -> None:
        """Display the processed frame with visualization of tracking and features"""
        original_frame = frame_data[0]
        results = frame_data[1]
        
        # Make a copy for drawing
        display_frame = original_frame.copy()
        frame_height, frame_width = display_frame.shape[:2]
        frame_center = (int(self.cx), int(self.cy))
        frame_dimensions = (frame_width, frame_height)
        
        # Draw the frame center
        cv2.circle(display_frame, frame_center, 5, (0, 255, 0), -1)
        
        # Variables to track if target was found
        best_box = None
        target_lost = True
        
        # Process detection results if available
        if results and results.boxes and len(results.boxes.conf) > 0:
            # Find the best detection
            best_conf_idx = -1
            best_conf = 0
            confidence_threshold = 0.4
            
            for i, conf in enumerate(results.boxes.conf):
                conf_val = conf.item() if hasattr(conf, 'item') else conf
                if conf_val > confidence_threshold and conf_val > best_conf:
                    best_conf = conf_val
                    best_conf_idx = i
            
            # Draw the best detection if found
            if best_conf_idx >= 0:
                target_lost = False
                coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
                best_box = tuple(map(int, coords))  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = best_box
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Get center of the box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                box_center = (center_x, center_y)
                
                # Calculate aspect ratio
                current_width = x2 - x1
                current_height = y2 - y1
                current_aspect_ratio = current_width / current_height
                
                # Draw line from frame center to box center
                cv2.line(display_frame, frame_center, box_center, (0, 255, 0), 2)
                
                # Draw the current box features
                current_features = self._get_features_from_box(best_box)
                for i, point in enumerate(current_features):
                    px, py = int(point[0]), int(point[1])
                    cv2.circle(display_frame, (px, py), 4, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"{i+1}", (px+5, py+5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw reference features if available
                if self.reference_features is not None:
                    for i, point in enumerate(self.reference_features):
                        px, py = int(point[0]), int(point[1])
                        cv2.circle(display_frame, (px, py), 4, (255, 0, 0), -1)
                        cv2.putText(display_frame, f"{i+1}", (px+5, py+5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                        # Draw line between reference and current feature if they exist
                        if i < len(current_features):
                            curr_px, curr_py = int(current_features[i][0]), int(current_features[i][1])
                            cv2.line(display_frame, (px, py), (curr_px, curr_py), (0, 255, 255), 1)
                
                # If we have the reference image, draw that too in a corner
                if self.reference_roi is not None:
                    h, w = self.reference_roi.shape[:2]
                    max_display_size = 150
                    scale = min(max_display_size / w, max_display_size / h)
                    ref_small = cv2.resize(self.reference_roi, (int(w * scale), int(h * scale)))
                    
                    # Draw in top right corner
                    h_small, w_small = ref_small.shape[:2]
                    roi = display_frame[10:10+h_small, frame_width-w_small-10:frame_width-10]
                    
                    if roi.shape == ref_small.shape:
                        # Create a blended overlay
                        alpha = 0.7
                        cv2.addWeighted(ref_small, alpha, roi, 1-alpha, 0, roi)
                        
                        # Add border
                        cv2.rectangle(display_frame, 
                                    (frame_width-w_small-10, 10), 
                                    (frame_width-10, 10+h_small), 
                                    (0, 255, 255), 2)
                        
                        # Add label
                        cv2.putText(display_frame, "Reference", 
                                   (frame_width-w_small-10, 8), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display depth estimate
                cv2.putText(display_frame, f"Depth: {self.depth_estimate:.2f}m", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display error info
                if self.reference_captured:
                    x_offset = (center_x - frame_center[0]) / (frame_width / 2)
                    y_offset = (center_y - frame_center[1]) / (frame_height / 2)
                    cv2.putText(display_frame, f"Offset X: {x_offset:.2f}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Offset Y: {y_offset:.2f}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add feature count display
                    cv2.putText(display_frame, f"Features: {self.num_features} (bbox)", 
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Add aspect ratio information
                    if hasattr(self, 'reference_aspect_ratio'):
                        aspect_ratio_bias = current_aspect_ratio - self.reference_aspect_ratio
                        ar_color = (0, 255, 0) if abs(aspect_ratio_bias) < 0.1 else (0, 165, 255)
                        cv2.putText(display_frame, f"Aspect: {current_aspect_ratio:.2f}/{self.reference_aspect_ratio:.2f}", 
                                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ar_color, 2)
                        
                        # Visualize optimal position zone
                        # Draw an arrow indicating which way to move for optimal position
                        if abs(aspect_ratio_bias) >= 0.1:
                            arrow_length = 50
                            arrow_start = (center_x, center_y - 40)  # Above the car
                            
                            # Determine arrow direction based on aspect ratio bias and x_offset
                            direction = 1 if aspect_ratio_bias > 0 else -1
                            side_factor = 1 if x_offset > 0 else -1
                            
                            # If aspect_ratio_bias is positive (too much width), move further to side
                            # If aspect_ratio_bias is negative (too little width), move more behind
                            arrow_end = (center_x + direction * side_factor * arrow_length, center_y - 40)
                            
                            # Draw the arrow
                            cv2.arrowedLine(display_frame, arrow_start, arrow_end, ar_color, 2, tipLength=0.3)
                    
                    # Calculate average error between reference and current features
                    if self.reference_features is not None and current_features is not None:
                        ref_array = np.array(self.reference_features)
                        curr_array = np.array(current_features)
                        
                        if ref_array.shape == curr_array.shape and curr_array.size > 0:
                            mean_error = np.mean(np.linalg.norm(curr_array - ref_array, axis=1))
                            cv2.putText(display_frame, f"Avg Error: {mean_error:.1f}px", 
                                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    # Display frame counter if reference not yet captured
                    cv2.putText(display_frame, f"Waiting for reference: {self.detection_frame_count}/{self.target_reference_frame}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display detection confidence
                cv2.putText(display_frame, f"Conf: {best_conf:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # If target is lost, show notification
        if target_lost:
            cv2.putText(display_frame, "NO TARGET DETECTED", 
                        (int(frame_width * 0.1), int(frame_height * 0.5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Generate follow command based on detection results
        if not target_lost and best_box is not None and self.reference_captured:
            self._generate_follow_command(frame_dimensions, results)
        else:
            # If target lost or reference not yet captured, send zero commands (hover)
            if hasattr(self, 'drone_commander'):
                self.drone_commander.piloting(x=0, y=0, z=0, z_rot=0, dt=0.15)
            if target_lost:
                print("Target lost - Hovering")
            elif not self.reference_captured:
                print("Waiting for reference frame - Hovering")
        
        # Add title
        cv2.putText(display_frame, "Bbox IBVS Tracking", 
                    (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Drone View", display_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run pose-based IBVS to follow behind a car")
    parser.add_argument("--ip", type=str, default=DroneIp.SIMULATED, help="Drone IP address")
    parser.add_argument("--speed", type=int, default=35, help="Maximum control speed")
    parser.add_argument("--simulated", action="store_true", help="Running in simulation mode")
    args = parser.parse_args()
    
    # Create controller with the PoseIBVSProcessor
    controller = PoseIBVSController(
        ip=args.ip,
        processor_class=PoseIBVSProcessor,
        speed=args.speed
    )
    
    # Initialize drone position before starting tracking
    if args.simulated:
        controller.initialize_position(takeoff_height=3.0, gimbal_angle=-45, back_distance=3.0, left_distance=0.0)
    
    # Start tracking
    controller.run()
