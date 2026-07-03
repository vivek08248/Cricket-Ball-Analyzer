import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque
import math

class CricketBallTracker:
    def __init__(self, model_path, conf_threshold=0.5, max_trail_length=50):
        """
        Initialize the Cricket Ball Tracker
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.trajectory = deque(maxlen=max_trail_length)
        self.kalman_filter = self.setup_kalman_filter()
        self.last_position = None
        self.frame_count = 0
        
    def setup_kalman_filter(self):
        """Initialize Kalman filter for smooth trajectory prediction"""
        kalman = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (x, y, vx, vy)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        return kalman
    
    def detect_ball(self, frame):
        """Detect cricket ball in the frame using YOLOv8"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if getattr(result, 'boxes', None) is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # confidence
                    confidence = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                    if confidence > self.conf_threshold:
                        # safer extraction of xyxy
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        if xyxy.size >= 4:
                            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        else:
                            # fallback - skip this box if coords are unexpected
                            continue
                        class_id = int(box.cls.item()) if hasattr(box, 'cls') else -1
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        })
        
        return detections
    
    def update_trajectory(self, current_position):
        """Update ball trajectory with Kalman filter smoothing"""
        if current_position is None:
            return None
            
        # Convert to measurement (2x1 column)
        measurement = np.array([[np.float32(current_position[0])], 
                               [np.float32(current_position[1])]], dtype=np.float32)
        
        # Kalman prediction (not used directly but advances internal state)
        _ = self.kalman_filter.predict()
        
        # Kalman correction - returns a state (4x1) matrix
        estimated = self.kalman_filter.correct(measurement)
        
        # Extract x,y safely from estimated (may be shape (4,1))
        try:
            est_x = float(estimated[0, 0])
            est_y = float(estimated[1, 0])
        except Exception:
            # fallback if shape is (4,) or other
            est_x = float(np.array(estimated).flatten()[0])
            est_y = float(np.array(estimated).flatten()[1])
        
        smoothed_position = (int(round(est_x)), int(round(est_y)))
        self.trajectory.append(smoothed_position)
        
        return smoothed_position

    def calculate_bsi(self, positions):
        """
        Calculate Ball Swing Intensity (BSI).We fit a straight line to the early segment (first 20% or at least 3 points),then compute perpendicular distances of all points to that line and take the mean.Returns BSI in pixels (average perpendicular deviation)."""
        if len(positions) < 3:
            return 0.0

        pts = np.array(positions, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]

        # choose reference segment: first 20% of points (at least 3)
        n_ref = max(3, len(pts) // 5)
        x_ref = x[:n_ref]
        y_ref = y[:n_ref]

        # Fit line y = m*x + c (least squares). If vertical-ish, fit x = ay + b instead.
        # Use exception handling for degenerate cases.
        try:
            m, c = np.polyfit(x_ref, y_ref, 1)  # slope m and intercept c
            # perpendicular distance formula for line y = m x + c
            denom = math.sqrt(m*m + 1.0)
            dists = np.abs(m * x - y + c) / denom
        except Exception:
            # fallback: if polyfit fails (near-vertical), fit x = a*y + b
            a, b = np.polyfit(y_ref, x_ref, 1)
            denom = math.sqrt(a*a + 1.0)
            dists = np.abs(a * y - x + b) / denom

        bsi_value = float(np.mean(dists))
        return round(bsi_value, 3)

    def calculate_trajectory_metrics(self):
        """Calculate trajectory metrics including BSI"""
        if len(self.trajectory) < 2:
            return None
            
        positions = list(self.trajectory)
        
        # Calculate speeds between consecutive points (pixels per frame)
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = math.hypot(dx, dy)
            speeds.append(distance)
        
        # Overall direction angle (from start to end)
        if len(positions) >= 2:
            start_pos = positions[0]
            end_pos = positions[-1]
            dx_total = end_pos[0] - start_pos[0]
            dy_total = end_pos[1] - start_pos[1]
            angle = math.degrees(math.atan2(dy_total, dx_total))
        else:
            angle = 0.0

        # BSI (Ball Swing Intensity) in pixels
        bsi_value = self.calculate_bsi(positions)
        
        metrics = {
            'total_points': len(positions),
            'total_distance': float(sum(speeds)) if speeds else 0.0,
            'average_speed': float(np.mean(speeds)) if speeds else 0.0,
            'max_speed': float(np.max(speeds)) if speeds else 0.0,
            'start_point': positions[0] if positions else None,
            'end_point': positions[-1] if positions else None,
            'direction_angle': float(angle),
            'current_velocity': float(speeds[-1]) if speeds else 0.0,
            'bsi': bsi_value
        }
        
        return metrics
    
    def draw_trajectory(self, frame):
        """Draw ball trajectory on the frame"""
        if len(self.trajectory) < 2:
            return frame
            
        # Draw trajectory path with gradient color
        for i in range(1, len(self.trajectory)):
            intensity = int(255 * (i / max(1, len(self.trajectory)-1)))
            color = (0, intensity, 255 - intensity)  # Blue to Red gradient
            thickness = max(2, int(3 * (i / max(1, len(self.trajectory)-1))))
            cv2.line(frame, self.trajectory[i-1], self.trajectory[i], color, thickness)
        
        # Draw current position
        if self.trajectory:
            current_pos = self.trajectory[-1]
            cv2.circle(frame, current_pos, 8, (0, 255, 0), -1)  # Green dot
            cv2.circle(frame, current_pos, 8, (255, 255, 255), 2)  # White border
        
        return frame
    
    def draw_detection_info(self, frame, detection, metrics):
        """Draw detection information and metrics on frame"""
        if detection:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw confidence label
            label = f"Ball: {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw trajectory metrics
        if metrics:
            y_offset = 30
            metrics_text = [
                f"Points: {metrics.get('total_points', 0)}",
                f"Distance: {metrics.get('total_distance', 0.0):.1f}px",
                f"Avg Speed: {metrics.get('average_speed', 0.0):.1f}px/frame",
                f"Angle: {metrics.get('direction_angle', 0.0):.1f}°",
                f"BSI: {metrics.get('bsi', 0.0):.3f}px"
            ]
            
            for text in metrics_text:
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            return frame
    
    def process_frame(self, frame):
        """Process a single frame for ball detection and trajectory update"""
        self.frame_count += 1
        
        # Detect ball in current frame
        detections = self.detect_ball(frame)
        
        current_position = None
        best_detection = None
        
        if detections:
            # Use detection with highest confidence
            best_detection = max(detections, key=lambda x: x['confidence'])
            current_position = best_detection['center']
        
        # Update trajectory with Kalman filter
        smoothed_position = self.update_trajectory(current_position)
        
        # Calculate trajectory metrics
        metrics = self.calculate_trajectory_metrics()
        
        # Draw trajectory and information
        frame = self.draw_trajectory(frame)
        frame = self.draw_detection_info(frame, best_detection, metrics)
        
        return frame, metrics, best_detection

def process_video(input_path, output_path, model_path, conf_threshold=0.5):
    """Process complete video for ball trajectory tracking"""
    
    # Initialize tracker
    tracker = CricketBallTracker(model_path, conf_threshold)
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
    
    frame_number = 0
    all_metrics = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, metrics, detection = tracker.process_frame(frame)
        
        if metrics:
            all_metrics.append(metrics)
        
        # Write frame to output
        out.write(processed_frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed frame {frame_number}/{total_frames}")
        
        # Removed cv2.imshow() to avoid GUI issues
        # You can uncomment the following lines if you want to check for early termination
        # but without displaying the video
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    cap.release()
    out.release()
    # Removed cv2.destroyAllWindows() since we're not displaying anything
    
    print(f"Processing completed! Output saved to: {output_path}")
    return all_metrics, tracker.trajectory

def plot_trajectory_analysis(trajectory, metrics_history, output_path):
    """Create detailed trajectory analysis plots"""
    if not trajectory:
        print("No trajectory data to plot")
        return
    
    positions = list(trajectory)
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Trajectory path
    ax1.plot(x_coords, y_coords, linewidth=2, alpha=0.7, label='Trajectory')
    ax1.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='viridis', s=50)
    ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Cricket Ball Trajectory Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert y-axis to match image coordinates
    
    # Plot 2: Speed analysis
    if len(metrics_history) > 1:
        speeds = [m['average_speed'] for m in metrics_history if m]
        frames = list(range(len(speeds)))
        ax2.plot(frames, speeds, linewidth=2)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Speed (pixels/frame)')
        ax2.set_title('Ball Speed Over Time')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance covered
    if len(metrics_history) > 1:
        distances = [m['total_distance'] for m in metrics_history if m]
        frames = list(range(len(distances)))
        
        ax3.plot(frames, distances, linewidth=2)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Total Distance (pixels)')
        ax3.set_title('Cumulative Distance Covered')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory statistics (including BSI)
    if metrics_history and metrics_history[-1]:
        last_metrics = metrics_history[-1]
        stats_labels = ['Total Points', 'Total Distance', 'Avg Speed', 'Direction Angle', 'BSI (px)']
        stats_values = [
            last_metrics.get('total_points', 0),
            last_metrics.get('total_distance', 0.0),
            last_metrics.get('average_speed', 0.0),
            last_metrics.get('direction_angle', 0.0),
            last_metrics.get('bsi', 0.0)  
        ]
        
        bars = ax4.bar(stats_labels, stats_values)
        ax4.set_title('Trajectory Statistics')
        ax4.set_ylabel('Values')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory analysis plot saved to: {output_path}")

def main():
    MODEL_PATH = r"C:\Users\vivek\Documents\Cricket Ball Analyzer\runs\detect\train26\weights\best.pt"  # Path to your trained YOLOv8 model
    INPUT_VIDEO = r"C:\Users\vivek\Documents\Cricket Ball Analyzer\Cricket-Ball-Trajectory-Prediction\videos\test3.mp4"  # Input video path
    OUTPUT_VIDEO = r"C:\Users\vivek\Documents\Cricket Ball Analyzer\Cricket-Ball-Trajectory-Prediction\output\output.mp4"  # Output video path
    PLOT_PATH = r"C:\Users\vivek\Documents\Cricket Ball Analyzer\Cricket-Ball-Trajectory-Prediction\output\output.png"  # Analysis plot path
    CONFIDENCE_THRESHOLD = 0.5
    
    try:
        # Process video and get trajectory data
        metrics_history, trajectory = process_video(
            INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH, CONFIDENCE_THRESHOLD
        )
        
        # Generate trajectory analysis plot
        plot_trajectory_analysis(trajectory, metrics_history, PLOT_PATH)
        
        # Print final statistics
        if metrics_history and metrics_history[-1]:
            final_metrics = metrics_history[-1]
            print("\n=== FINAL TRAJECTORY STATISTICS ===")
            print(f"Total trajectory points: {final_metrics['total_points']}")
            print(f"Total distance covered: {final_metrics['total_distance']:.2f} pixels")
            print(f"Average speed: {final_metrics['average_speed']:.2f} pixels/frame")
            print(f"Direction angle: {final_metrics['direction_angle']:.1f}°")
            print(f"BSI: {final_metrics['bsi']:.3f} px (average perpendicular deviation)")
            print(f"Start point: {final_metrics['start_point']}")
            print(f"End point: {final_metrics['end_point']}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

# Single frame processing example
def process_single_image(image_path, model_path, output_path):
    tracker = CricketBallTracker(model_path)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    processed_image, metrics, detection = tracker.process_frame(image)
    
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to: {output_path}")
    
    if detection:
        print(f"Ball detected with confidence: {detection['confidence']:.2f}")
    
    return processed_image, metrics

if __name__ == "__main__":
    main()