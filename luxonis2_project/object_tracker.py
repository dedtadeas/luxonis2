import cv2
import numpy as np

SIMILAR_OBJECT_DISTANCE_THRESHOLD = 50    # Threshold for matching based on position, in pixels
MISSING_OBJECT_FRAME_STEPS_THRESHOLD = 2  # Threshold for number of concurrent frames as missing frame steps allowance, in frames
COLOR_THRESHOLD = 20                      # Threshold for color difference between objects
SCALE_FACTOR = 0.5                        # Scale factor for displaying the combined frame


class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}  # Dictionary to store tracked objects
        self.next_object_id = 0
        self.track_frame = None  # We'll initialize this when we know the frame size

        # Numpy array for storing last positions [x, y, frame_id, object_id, shape, color_r, color_g, color_b]
        self.last_positions = np.empty((0, 8), dtype=np.float32)

    def update_tracked_objects(
        self, detected_objects: np.ndarray, frame_id: int
    ) -> list:
        """
        Update the tracked objects with the detected objects in the current frame
        Args:
            detected_objects (np.ndarray): Detected objects in the format (shape, x, y, w/h or r, None for rectangles, color_r, color_g, color_b)
            frame_id (int): Frame number

        Returns:
            list: List of updated objects in the format (object_id, (x, y))
        """
        updated_objects = []

        if self.last_positions.size > 0:
            matches = self._get_matches(detected_objects)

            for i in range(matches.shape[1]):
                if matches[:, i].any():
                    matched_idx = np.where(matches[:, i])[0][0]
                    matched_object_id = int(self.last_positions[matched_idx, 3])
                    self._update_existing_object(
                        matched_object_id, detected_objects[i], frame_id
                    )
                    updated_objects.append(
                        (matched_object_id, detected_objects[i][1:3])
                    )
                else:
                    new_object_id = self._add_new_object(detected_objects[i], frame_id)
                    updated_objects.append((new_object_id, detected_objects[i][1:3]))
        else:
            # If no tracked objects, add all detected as new
            for obj in detected_objects:
                new_object_id = self._add_new_object(obj, frame_id)
                updated_objects.append((new_object_id, obj[1:3]))

        # Remove old objects that have been missing for too many frames
        self._remove_old_objects(frame_id)

        return updated_objects 

    def _get_matches(self, detected_objects):
        """
        Get matches between the last positions and the detected objects. Uses vectorized operations for efficiency.
        """
        detected_positions = detected_objects[:, 1:3].astype(np.float32)
        detected_shapes = detected_objects[:, 0]
        detected_colors = detected_objects[:, 5:8]

            # Vectorized operation to find matching objects
        distances = np.linalg.norm(self.last_positions[:, :2, np.newaxis] - detected_positions.T, axis=1)
        close_enough = distances < SIMILAR_OBJECT_DISTANCE_THRESHOLD
        shape_matches = self.last_positions[:, 4, np.newaxis] == detected_shapes
        color_diff = np.abs(self.last_positions[:, 5:8, np.newaxis] - detected_colors.T).sum(axis=1)
        color_matches = color_diff < COLOR_THRESHOLD

        matches = close_enough & shape_matches & color_matches
        return matches # Return a list of objects that were updated

    def _update_existing_object(
        self, object_id: int, obj: np.ndarray, frame_id: int
    ) -> None:
        # Update the tracked object with the new position and frame_id
        self.tracked_objects[object_id]["last_frame_id"] = frame_id
        self.tracked_objects[object_id]["path"].append((frame_id, obj[1:3]))

        idx = np.where(self.last_positions[:, 3] == object_id)[0]
        self.last_positions[idx, :3] = [obj[1], obj[2], frame_id]

    def _add_new_object(self, obj: np.ndarray, frame_id: int) -> int:
        new_object_id = self.next_object_id
        self.tracked_objects[new_object_id] = {
            "shape": obj[0],
            "color": obj[5:8],
            "last_frame_id": frame_id,
            "path": [(frame_id, obj[1:3])],
        }

        new_position = np.array([[obj[1], obj[2], frame_id, new_object_id, obj[0], *obj[5:8]]],
            dtype=np.float32,
        )
        self.last_positions = np.vstack([self.last_positions, new_position])

        self.next_object_id += 1
        return new_object_id

    def _remove_old_objects(self, frame_id: int) -> None:
        # Filter out objects that have been missing for too many frames
        recent_positions = self.last_positions[:, 2] > (
            frame_id - MISSING_OBJECT_FRAME_STEPS_THRESHOLD
        )
        self.last_positions = self.last_positions[recent_positions]

    ### Methods for real time visualization ###
    def initialize_track_frame(self, frame_shape: tuple) -> None:
        self.track_frame = np.zeros(frame_shape, dtype=np.uint8)

    def update_track_frame(self, updated_objects: list) -> None:
        for object_id, coordinates in updated_objects:
            path = self.tracked_objects[object_id]["path"]
            color = tuple(int(c) for c in self.tracked_objects[object_id]["color"])

            if len(path) > 1:
                start_point = tuple(int(coord) for coord in path[-2][1])
                end_point = tuple(int(coord) for coord in path[-1][1])
                cv2.line(self.track_frame, start_point, end_point, color, 2)

    def get_track_frame(self):
        return self.track_frame

    def combine_frames(self, original_frame: np.ndarray) -> np.ndarray:
        h, w = original_frame.shape[:2]
        combined_frame = np.zeros((h, 2 * w, 3), dtype=np.uint8)

        combined_frame[:, :w] = original_frame
        combined_frame[:, w:] = self.track_frame
        combined_frame = cv2.resize(
            combined_frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR
        )

        return combined_frame
