
# Real-Time Object Tracking Simulation

## Overview

This project simulates real-time object tracking using OpenCV for object detection and tracking, and Plotly for visualization. The application processes a video feed, identifies circles and rectangles, tracks their movement across frames, and generates an interactive HTML report visualizing the tracked paths.

## Project Structure

- **`main.py`**: The main script that runs the object detection and tracking simulation.
- **`object_tracker.py`**: Contains the `ObjectTracker` class, responsible for tracking detected objects across video frames. Emphasizes vectorized operations for efficiency.
- **`object_detector.py`**: Contains the `ObjectDetector` class, which detects circles and rectangles in each frame using OpenCV.
- **`plotly_result_viewer.py`**: Contains the `PlotlyResultViewer` class, which generates the HTML visualization of the tracked paths using Plotly.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.

## Prerequisites

- Python 3.7+
- Install the required Python packages using the following command:

```sh
pip install -r requirements.txt
```

## How to Use

### 1. Prepare the Video Source

Place the video file you want to process in the `video/` directory. The provided example video is `luxonis_task_video.mp4`.

### 2. Run the Application

Execute the main script to start the object tracking simulation:

```sh
python main.py
```

The script will process the video, track the detected objects, and generate an HTML report with the tracked paths.

### 3. View the Results

The HTML report will be saved as `output/tracking_results.html`. Open this file in your web browser to interactively explore the tracked paths.

### Configuration
- **Animation Speed**: You can control the animation speed by adjusting the `animation_speed` parameter in `TrackingApplication`.

## Key Classes and Configuration

### `ObjectDetector` (object_detector.py)

**Purpose**: 
Detects circles and rectangles in each video frame using contour detection and classification techniques.

**Key Parameters**:
- **RECTANGLE_CONTOUR_APPROX**: Approximation parameter for rectangle contours. Default is 0.02.
- **CIRCULARITY_THRESHOLD**: Threshold for determining if a contour is circular. Default is 0.7.

**Methods**:
- **detect_shapes(frame)**: Detects shapes (rectangles and circles) in the input frame.
- **classify_and_extract_shape(contour, frame)**: Classifies the shape of a contour and extracts its properties, such as coordinates, size, and color.

### How to Use `ObjectDetector`

**Input**:
- **frame** (np.ndarray): The current video frame to be processed.

**Output**:
- **shapes** (np.ndarray): Array of detected shapes with their properties.

```python
detector = ObjectDetector()
detected_shapes = detector.detect_objects(frame)
```
### `ObjectTracker` (object_tracker.py)

**Purpose**: 
Tracks detected objects across frames, handling cases where objects disappear and reappear in concurrent frames. It uses vectorized operations to find matching objects efficiently.

**Key Parameters**:
- **SIMILAR_OBJECT_DISTANCE_THRESHOLD**: Threshold for matching objects based on position, in pixels. Default is 50.
- **MISSING_OBJECT_FRAME_STEPS_THRESHOLD**: Threshold for missing object before removing it from tracking, in frames. Default is 2.
- **COLOR_THRESHOLD**: Threshold for color difference between objects. Default is 20.
- **SCALE_FACTOR**: Scale factor for displaying the combined frame. Default is 0.5.

**Methods**:
- **initialize_track_frame(frame_shape)**: Initializes the track frame with the same size as the input frame.
- **update_tracked_objects(detected_objects, frame_id)**: Updates the list of tracked objects based on newly detected objects using vectorized operations.
- **combine_frames(original_frame)**: Combines the original frame with the tracking frame to display the tracked paths.

### How to Use `ObjectTracker`

**Input**:
- **detected_objects** (np.ndarray): Array of detected objects with their properties.
- **frame_id** (int): The current frame number.

**Output**:
- **updated_objects** (list): List of objects that were updated or newly added to the tracking list.

```python
tracker = ObjectTracker()
tracker.initialize_track_frame(frame.shape)
updated_objects = tracker.update_tracked_objects(detected_shapes, frame_id)
combined_frame = tracker.combine_frames(frame)
```

### `PlotlyResultViewer` (plotly_result_viewer.py)

**Purpose**: 
Generates an HTML report visualizing the tracked paths of objects across frames.

**Methods**:
- **generate_html(output_file=None)**: Generates the HTML report, saving it to the specified output file.

### How to Use `PlotlyResultViewer`

**Input**:
- **tracked_objects** (dict): The dictionary of tracked objects and their paths.
- **video_path** (str): Path to the video file to extract dimensions.

**Output**:
- An HTML file containing the visualization of the tracked paths.

```python
viewer = PlotlyResultViewer(tracked_objects, "video/luxonis_task_video.mp4")
viewer.generate_html("tracking_results.html")
```

## Example

To run the provided example:

```sh
python main.py
```

This will process `luxonis_task_video.mp4` and generate an HTML file with the results.

## Notes

- Ensure your video source file exists before running the application.
- The visualization is saved as an HTML file for easy sharing and review.

## License
.....

## TODOs:
- Improve image recognition. Try different algorithms such as the watershed algorithm.
- Extract the global parameters into a config YAML file.
