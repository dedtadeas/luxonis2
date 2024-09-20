import os
import cv2
import timeit
import time
from tqdm import tqdm
from object_tracker import ObjectTracker
from object_detector import ObjectDetector
from plotly_result_viewer import PlotlyResultViewer


class TrackingApplication:
    def __init__(
        self,
        video_source: str,
        animation_speed: float = 0,
        show_processing_frame: bool = True,
    ):
        self.video_source = video_source
        self.tracker = ObjectTracker()
        self.detector = ObjectDetector()
        self.animation_speed = animation_speed
        self.show_processing_frame = show_processing_frame
        self.cap = None
        self.frame_id = 0

    def initialize(self):
        # Check if the video source is a file and if it exists
        if isinstance(self.video_source, str) and not os.path.exists(self.video_source):
            raise FileNotFoundError(
                f"Error: The file '{self.video_source}' does not exist."
            )

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video source.")

    def run(self) -> dict:
        try:
            self.initialize()

            total_time = 0
            total_detection_time = 0
            total_tracking_time = 0
            frame_count = 0

            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with tqdm(total=total_frames, desc="Processing Frames") as pbar:
                while True:
                    ret, frame = self.cap.read()

                    if not ret:
                        break

                    # Initialize the track_frame once we know the frame size
                    if self.frame_id == 0:
                        self.tracker.initialize_track_frame(frame.shape)

                    # Start timing for detection
                    detection_start_time = timeit.default_timer()

                    # Detect objects in the frame
                    detected_objects = self.detector.detect_objects(frame)

                    # End timing for detection
                    detection_end_time = timeit.default_timer()

                    # Start timing for tracking
                    tracking_start_time = timeit.default_timer()

                    # Update the tracker with the detected objects
                    updated_objects = self.tracker.update_tracked_objects(
                        detected_objects, self.frame_id
                    )

                    # End timing for tracking
                    tracking_end_time = timeit.default_timer()

                    # Accumulate times
                    total_detection_time += detection_end_time - detection_start_time
                    total_tracking_time += tracking_end_time - tracking_start_time
                    total_time += (
                        tracking_end_time - detection_start_time
                    )  # Total time for the frame
                    frame_count += 1

                    # Display the combined frame
                    if self.show_processing_frame:
                        self.tracker.update_track_frame(updated_objects)

                        # Combine the original frame with the tracking frame
                        combined_frame = self.tracker.combine_frames(frame)
                        cv2.imshow("Tracking objects", combined_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    self.frame_id += 1

                    time.sleep(self.animation_speed)

                    # Update the progress bar
                    pbar.update(1)

            # Calculate and print the average times
            if frame_count > 0:
                average_time_per_frame = total_time / frame_count
                average_detection_time = total_detection_time / frame_count
                average_tracking_time = total_tracking_time / frame_count

                print(
                    f"Average processing time per frame: {average_time_per_frame:.6f} seconds"
                )
                print(
                    f"Average detection time per frame: {average_detection_time:.6f} seconds"
                )
                print(
                    f"Average tracking time per frame: {average_tracking_time:.6f} seconds"
                )

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            self.cleanup()

        return self.tracker.tracked_objects

    def cleanup(self):
        # Ensure resources are released properly
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    app = TrackingApplication("video/luxonis_task_video.mp4")
    tracked_objects = app.run()

    # Generate the HTML report with Plotly
    viewer = PlotlyResultViewer(tracked_objects, "video/luxonis_task_video.mp4")
    viewer.generate_html("output/tracking_results.html")
