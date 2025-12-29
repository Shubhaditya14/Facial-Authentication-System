"""Face detection and cropping utilities.

Provides face detection using MediaPipe with support for multi-modal
aligned image cropping.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class FaceDetector:
    """Face detector using MediaPipe or other backends.

    Provides face detection and cropping with support for aligned
    multi-modal images (RGB, depth, IR).

    Example:
        >>> detector = FaceDetector(backend="mediapipe")
        >>> image = cv2.imread("face.jpg")
        >>> faces = detector.detect(image)
        >>> cropped = detector.crop_face(image, margin=0.2)
    """

    def __init__(
        self,
        backend: str = "mediapipe",
        min_confidence: float = 0.5,
        model_selection: int = 0,
    ):
        """Initialize face detector.

        Args:
            backend: Detection backend ("mediapipe" or "opencv").
            min_confidence: Minimum detection confidence threshold.
            model_selection: MediaPipe model selection (0=short range, 1=full range).

        Raises:
            ValueError: If backend is not supported.
            ImportError: If required backend is not installed.
        """
        self.backend = backend.lower()
        self.min_confidence = min_confidence
        self._detector = None

        if self.backend == "mediapipe":
            self._init_mediapipe(model_selection)
        elif self.backend == "opencv":
            self._init_opencv()
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'mediapipe' or 'opencv'")

    def _init_mediapipe(self, model_selection: int) -> None:
        """Initialize MediaPipe face detection.

        Args:
            model_selection: Model selection (0 or 1).
        """
        try:
            import mediapipe as mp

            # Try new API first (mediapipe >= 0.10.8)
            if hasattr(mp, "tasks"):
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                # Use the new FaceDetector API
                base_options = mp_python.BaseOptions(
                    model_asset_path=self._get_model_path()
                )
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=self.min_confidence,
                )
                self._detector = vision.FaceDetector.create_from_options(options)
                self._use_new_api = True
            else:
                # Fall back to legacy API
                self._mp_face_detection = mp.solutions.face_detection
                self._detector = self._mp_face_detection.FaceDetection(
                    min_detection_confidence=self.min_confidence,
                    model_selection=model_selection,
                )
                self._use_new_api = False
        except (ImportError, AttributeError, Exception) as e:
            # If MediaPipe fails, fall back to OpenCV
            import warnings
            warnings.warn(f"MediaPipe initialization failed: {e}. Falling back to OpenCV.")
            self.backend = "opencv"
            self._init_opencv()

    def _get_model_path(self) -> str:
        """Get the path to MediaPipe face detection model."""
        import mediapipe as mp
        import os

        # Try to find the model in the mediapipe package
        mp_path = os.path.dirname(mp.__file__)
        model_paths = [
            os.path.join(mp_path, "modules", "face_detection", "face_detection_short_range.tflite"),
            os.path.join(mp_path, "modules", "face_detection", "face_detection_full_range.tflite"),
        ]

        for path in model_paths:
            if os.path.exists(path):
                return path

        # Return empty string to let MediaPipe use its default
        return ""

    def _init_opencv(self) -> None:
        """Initialize OpenCV Haar Cascade face detection."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)

        if self._detector.empty():
            raise RuntimeError("Failed to load OpenCV face cascade classifier")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in an image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR or RGB format.

        Returns:
            List of detections, each containing:
            - bbox: (x, y, w, h) bounding box in pixel coordinates
            - confidence: Detection confidence score
        """
        if image is None or image.size == 0:
            return []

        if self.backend == "mediapipe":
            return self._detect_mediapipe(image)
        else:
            return self._detect_opencv(image)

    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe.

        Args:
            image: Input image (BGR or RGB).

        Returns:
            List of face detections.
        """
        # MediaPipe expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        h, w = image.shape[:2]
        detections = []

        if getattr(self, "_use_new_api", False):
            # New MediaPipe Tasks API
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = self._detector.detect(mp_image)

            for detection in results.detections:
                bbox = detection.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                box_w = bbox.width
                box_h = bbox.height

                # Clamp to image bounds
                x = max(0, x)
                y = max(0, y)
                box_w = min(box_w, w - x)
                box_h = min(box_h, h - y)

                confidence = detection.categories[0].score if detection.categories else 0.0

                detections.append({
                    "bbox": (int(x), int(y), int(box_w), int(box_h)),
                    "confidence": confidence,
                })
        else:
            # Legacy API
            results = self._detector.process(rgb_image)

            if results.detections:
                for detection in results.detections:
                    bbox_rel = detection.location_data.relative_bounding_box

                    # Convert relative to absolute coordinates
                    x = int(bbox_rel.xmin * w)
                    y = int(bbox_rel.ymin * h)
                    box_w = int(bbox_rel.width * w)
                    box_h = int(bbox_rel.height * h)

                    # Clamp to image bounds
                    x = max(0, x)
                    y = max(0, y)
                    box_w = min(box_w, w - x)
                    box_h = min(box_h, h - y)

                    detections.append({
                        "bbox": (x, y, box_w, box_h),
                        "confidence": detection.score[0] if detection.score else 0.0,
                    })

        return detections

    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade.

        Args:
            image: Input image.

        Returns:
            List of face detections.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        detections = []
        for x, y, w, h in faces:
            detections.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": 1.0,  # Haar doesn't provide confidence
            })

        return detections

    def crop_face(
        self,
        image: np.ndarray,
        margin: float = 0.2,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Detect and crop the largest face with margin.

        If no face is detected, returns the original image (optionally resized).

        Args:
            image: Input image (H, W, C).
            margin: Margin around face as fraction of face size.
            output_size: Optional (width, height) to resize cropped face.

        Returns:
            Cropped face image, or original image if no face detected.
        """
        if image is None or image.size == 0:
            return image

        detections = self.detect(image)

        if not detections:
            # No face detected, return original (optionally resized)
            if output_size:
                return cv2.resize(image, output_size)
            return image

        # Get largest face by area
        largest = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
        x, y, w, h = largest["bbox"]

        # Calculate margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        # Expand bbox with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        # Crop
        cropped = image[y1:y2, x1:x2]

        # Resize if needed
        if output_size:
            cropped = cv2.resize(cropped, output_size)

        return cropped

    def crop_aligned(
        self,
        images: Dict[str, np.ndarray],
        margin: float = 0.2,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Crop multiple aligned images (RGB, depth, IR) with the same bbox.

        Detects face on RGB image and applies same crop to all modalities.

        Args:
            images: Dictionary of modality name to image.
                Must contain "rgb" key for face detection.
            margin: Margin around face as fraction of face size.
            output_size: Optional (width, height) to resize all crops.

        Returns:
            Dictionary of cropped images with same keys.
        """
        if "rgb" not in images:
            raise ValueError("images dict must contain 'rgb' key for face detection")

        rgb_image = images["rgb"]
        detections = self.detect(rgb_image)

        result = {}

        if not detections:
            # No face detected, return originals (optionally resized)
            for key, img in images.items():
                if img is not None:
                    if output_size:
                        result[key] = cv2.resize(img, output_size)
                    else:
                        result[key] = img
                else:
                    result[key] = None
            return result

        # Get largest face
        largest = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
        x, y, w, h = largest["bbox"]

        # Calculate margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        # Calculate crop coordinates (relative to RGB)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(rgb_image.shape[1], x + w + margin_x)
        y2 = min(rgb_image.shape[0], y + h + margin_y)

        # Apply same crop to all modalities
        for key, img in images.items():
            if img is None:
                result[key] = None
                continue

            # Handle different image sizes
            img_h, img_w = img.shape[:2]
            rgb_h, rgb_w = rgb_image.shape[:2]

            # Scale coordinates if image sizes differ
            if img_h != rgb_h or img_w != rgb_w:
                scale_x = img_w / rgb_w
                scale_y = img_h / rgb_h
                img_x1 = int(x1 * scale_x)
                img_y1 = int(y1 * scale_y)
                img_x2 = int(x2 * scale_x)
                img_y2 = int(y2 * scale_y)
            else:
                img_x1, img_y1, img_x2, img_y2 = x1, y1, x2, y2

            # Clamp to image bounds
            img_x1 = max(0, img_x1)
            img_y1 = max(0, img_y1)
            img_x2 = min(img_w, img_x2)
            img_y2 = min(img_h, img_y2)

            # Crop
            cropped = img[img_y1:img_y2, img_x1:img_x2]

            # Resize if needed
            if output_size:
                cropped = cv2.resize(cropped, output_size)

            result[key] = cropped

        return result

    def get_face_bbox(
        self,
        image: np.ndarray,
        margin: float = 0.0,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get the bounding box of the largest detected face.

        Args:
            image: Input image.
            margin: Margin to add around the face.

        Returns:
            Tuple (x1, y1, x2, y2) or None if no face detected.
        """
        detections = self.detect(image)

        if not detections:
            return None

        # Get largest face
        largest = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
        x, y, w, h = largest["bbox"]

        # Calculate margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        # Expand and clamp
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        return (x1, y1, x2, y2)

    def close(self) -> None:
        """Release detector resources."""
        if self.backend == "mediapipe" and self._detector:
            if hasattr(self._detector, "close"):
                try:
                    self._detector.close()
                except Exception:
                    pass
        self._detector = None

    def __enter__(self) -> "FaceDetector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
