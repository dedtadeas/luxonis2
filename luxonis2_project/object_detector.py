import cv2
import numpy as np

# Global Variables for Parameterization
RECTANGLE_CONTOUR_APPROX = 0.02    # Approximation parameter for rectangle contours
CIRCULARITY_THRESHOLD    = 0.7     # Threshold for determining if a contour is circular


class ObjectDetector:
    def __init__(self):
        pass

    def detect_shapes(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect rectangles and circles in a frame using contour detection
        Args:
            frame (np.ndarray): Input frame
        Returns:
            np.ndarray: Detected shapes in the format (shape_id, x, y, w/h or r, None for rectangles, color_r, color_g, color_b)
        """

        # Convert to grayscale and use adaptive thresholding
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred_frame,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        shapes = []
        for cnt in contours:
            shape_type, x, y, w, h, color = self.classify_and_extract_shape(cnt, frame)
            if shape_type == "circle":
                shapes.append(
                    (1, x, y, w // 2, None, *color)
                )  # Circle format: (1, x, y, r, None, color)
            elif shape_type == "rectangle":
                shapes.append(
                    (2, x, y, w, h, *color)
                )  # Rectangle format: (2, x, y, w, h, color)

        if len(shapes) == 0:
            return np.empty((0, 8))  # Empty array if no shapes detected

        return np.array(shapes)

    def classify_and_extract_shape(self, contour: np.ndarray, frame: np.ndarray)->str:
        """
        Classify the shape of a contour and extract its properties
        
        Args:
            contour (np.ndarray): Contour to classify
            frame (np.ndarray): Frame to sample the color from the center of the shape
            
        Returns:
            str: Shape classification
            int: x-coordinate of the shape
            int: y-coordinate of the shape
            int: width of the shape (for rectangles)
            int: height of the shape (for rectangles)
            tuple: RGB color of the shape
        """
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, RECTANGLE_CONTOUR_APPROX * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        color = frame[y + h // 2, x + w // 2]  # Sample the color at the center

        # Skip degenerate contours
        if peri == 0:
            return "unknown", 0, 0, 0, 0, (0, 0, 0)

        if len(approx) == 4:
            return "rectangle", x, y, w, h, color
        else:
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * (area / (peri**2))
            if circularity > CIRCULARITY_THRESHOLD:
                return "circle", x + w // 2, y + h // 2, w, h, color
            else:
                return "unknown", 0, 0, 0, 0, (0, 0, 0)   # Ignore non-recognized shapes

    def detect_objects(self, frame: np.ndarray) -> np.ndarray:
        shapes = self.detect_shapes(frame)
        return shapes
