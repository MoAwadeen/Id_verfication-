import asyncio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
from pathlib import Path

from paddleocr import TextRecognition
from ultralytics import YOLO

from config import Config


@dataclass
class IDResult:
    first_name: str
    first_name_confidence: float
    full_name: str
    full_name_confidence: float
    id_number: Optional[str]
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    birth_date: Optional[str] = None
    governorate: Optional[str] = None
    gender: Optional[str] = None
    status: Optional[str] = None


class Id_Image:
    def __init__(
        self,
        image: Union[str, np.ndarray],
        cfg: Config,
        paddle_rec: TextRecognition,
        yolo_model: YOLO,
        executor: ThreadPoolExecutor,
    ):
        self.cfg = cfg
        self.image = self._read_image(image)
        self.paddle_rec = paddle_rec
        self.yolo_model = yolo_model
        self._executor = executor

    async def process(self) -> IDResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_process)

    def _sync_process(self) -> IDResult:
        transformed = self._perspective_transform(self.image)
        crops = self._crop_fields(transformed)

        first_name, fn_conf = self._ocr_name(crops["first_name"])
        full_name, ln_conf = self._ocr_name(crops["full_name"])

        id_number = self._ocr_id(transformed)
        id_info = self._decode_id(id_number) if id_number else {"status": "id is empty"}

        return IDResult(
            first_name=first_name,
            first_name_confidence=fn_conf,
            full_name=full_name,
            full_name_confidence=ln_conf,
            id_number=id_number,
            year=id_info.get("year"),
            month=id_info.get("month"),
            day=id_info.get("day"),
            birth_date=id_info.get("Birth Date"),
            governorate=id_info.get("Governorate"),
            gender=id_info.get("Gender"),
            status=id_info.get("status"),
        )

    def _read_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not load image at path: {image}")
            return img
        return image

    def _perspective_transform(self, image: np.ndarray) -> np.ndarray:
        resize_ratio = 500.0 / image.shape[0]
        small_img = cv2.resize(
            image,
            (int(image.shape[1] * resize_ratio), 500),
        )

        points = self._get_smart_corners(small_img)
        points = np.float32(points) / resize_ratio

        top_left, top_right, bottom_right, bottom_left = points
        max_width, max_height = self._extract_dimensions(
            top_left, top_right, bottom_right, bottom_left, self.cfg.aspect_ratio
        )

        input_points = np.float32([top_left, top_right, bottom_right, bottom_left])
        converted_points = np.float32([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ])

        matrix = cv2.getPerspectiveTransform(input_points, converted_points)
        img_output = cv2.warpPerspective(image, matrix, (max_width, max_height))

        h_out, w_out = img_output.shape[:2]
        if h_out > w_out:
            img_output = cv2.rotate(img_output, cv2.ROTATE_90_CLOCKWISE)

        img_output = cv2.resize(img_output, (self.cfg.target_width, self.cfg.target_height))
        return img_output

    def _extract_contours(self, image: np.ndarray, apply_clahe: bool = False) -> List:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_gray = clahe.apply(img_gray)

        img_smoothed = cv2.bilateralFilter(img_gray, 11, 17, 17)
        edges = cv2.Canny(img_smoothed, 30, 70)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges_dilated = cv2.dilate(edges_closed, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    def _extract_corner_points(self, biggest_contour: np.ndarray) -> Tuple:
        points = biggest_contour.reshape((4, 2))
        input_points = np.zeros((4, 2), dtype="float32")

        points_sum = points.sum(axis=1)
        input_points[0] = points[np.argmin(points_sum)]
        input_points[3] = points[np.argmax(points_sum)]

        points_diff = np.diff(points, axis=1)
        input_points[1] = points[np.argmin(points_diff)]
        input_points[2] = points[np.argmax(points_diff)]

        return (
            input_points[0],
            input_points[1],
            input_points[3],
            input_points[2],
        )

    def _get_valid_card_box(self, image: np.ndarray, contours: List) -> Optional[Tuple]:
        image_area = image.shape[0] * image.shape[1]
        best_box = None
        max_area = 0

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.float32(box)

            width, height = rect[1][0], rect[1][1]
            if width == 0 or height == 0:
                continue

            box_area = width * height
            ratio = max(width, height) / min(width, height)

            if (image_area * self.cfg.contour_min_area_ratio) < box_area < (image_area * self.cfg.contour_max_area_ratio):
                if self.cfg.contour_min_aspect_ratio < ratio < self.cfg.contour_max_aspect_ratio:
                    if box_area > max_area:
                        best_box = box
                        max_area = box_area

        if best_box is not None:
            return self._extract_corner_points(best_box)
        return None

    def _get_smart_corners(self, image: np.ndarray) -> Tuple:
        contours_standard = self._extract_contours(image, apply_clahe=False)
        corners = self._get_valid_card_box(image, contours_standard)

        if corners is not None:
            return corners

        contours_clahe = self._extract_contours(image, apply_clahe=True)
        corners = self._get_valid_card_box(image, contours_clahe)

        if corners is not None:
            return corners

        h, w = image.shape[:2]
        return (
            np.array([0, 0], dtype="float32"),
            np.array([w - 1, 0], dtype="float32"),
            np.array([w - 1, h - 1], dtype="float32"),
            np.array([0, h - 1], dtype="float32"),
        )

    def _extract_dimensions(
        self,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_right: np.ndarray,
        bottom_left: np.ndarray,
        aspect_ratio: float,
    ) -> Tuple[int, int]:
        bottom_width = np.linalg.norm(bottom_right - bottom_left)
        top_width = np.linalg.norm(top_right - top_left)
        max_width = max(int(bottom_width), int(top_width))
        max_height = int(max_width / aspect_ratio)
        return max_width, max_height

    def _crop_fields(self, img: np.ndarray) -> dict:
        fn = img[
            self.cfg.crop_first_name_y1:self.cfg.crop_first_name_y2,
            self.cfg.crop_first_name_x1:self.cfg.crop_first_name_x2,
        ]
        fln = img[
            self.cfg.crop_full_name_y1:self.cfg.crop_full_name_y2,
            self.cfg.crop_full_name_x1:self.cfg.crop_full_name_x2,
        ]
        return {
            "first_name": fn,
            "full_name": fln,
        }

    def _preprocess_field(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image is None")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

        ys, xs = np.where(mask > 0)
        if len(xs) > 20:
            x_min = int(np.percentile(xs, 2))
            x_max = int(np.percentile(xs, 98))
            y_min = int(np.percentile(ys, 5))
            y_max = int(np.percentile(ys, 95))

            margin_x = max(5, int((x_max - x_min) * 0.5))
            margin_y = max(5, int((y_max - y_min) * 0.20))

            x_min = max(0, x_min - margin_x)
            x_max = min(gray.shape[1], x_max + margin_x)
            y_min = max(0, y_min - margin_y)
            y_max = min(gray.shape[0], y_max + margin_y)

            cropped = image[y_min:y_max, x_min:x_max]
        else:
            cropped = image.copy()

        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, deskew_mask = cv2.threshold(gray_crop, 120, 255, cv2.THRESH_BINARY_INV)

        coords = np.column_stack(np.where(deskew_mask > 0))
        if len(coords) > 20:
            rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
            angle = rect[-1]

            if angle < -45:
                angle = 90 + angle

            if abs(angle) < 10:
                h, w = cropped.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                cropped = cv2.warpAffine(
                    cropped,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )

        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ys, _ = np.where(gray_crop < 120)
        if len(ys) > 20:
            y_min = int(np.percentile(ys, 5))
            y_max = int(np.percentile(ys, 95))
            text_height = y_max - y_min

            target_text_height = 64
            scale = target_text_height / max(text_height, 1)
            scale = np.clip(scale, 1.0, 4.0)

            new_w = int(cropped.shape[1] * scale)
            new_h = int(cropped.shape[0] * scale)
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _ocr_name(self, image: np.ndarray) -> Tuple[str, float]:
        processed = self._preprocess_field(image)
        results = self.paddle_rec.predict(input=processed, batch_size=1)

        if not results:
            return "", 0.0

        # PaddleOCR 3.x: result.json returns dict directly
        res = results[0]
        data = res.json

        if "res" in data:
            data = data["res"]

        text = data.get("rec_text", "")
        confidence = float(data.get("rec_score", 0.0))
        return text, confidence

    def _ocr_id(self, image: np.ndarray) -> Optional[str]:
        result = self.yolo_model.predict(
            source=image,
            conf=self.cfg.yolo_conf_threshold,
            verbose=False,
        )[0]

        sorted_boxes = sorted(result.boxes, key=lambda b: b.xyxy[0][0].item())
        detected_numbers = []

        for box in sorted_boxes:
            cls_id = int(box.cls[0].item())
            label = result.names[cls_id]
            if label.isdigit():
                detected_numbers.append(label)

        return "".join(detected_numbers) if detected_numbers else None

    def _decode_id(self, id_number: str) -> dict:
        governorates = {
            '01': 'Cairo', '02': 'Alexandria', '03': 'Port Said', '04': 'Suez',
            '11': 'Damietta', '12': 'Dakahlia', '13': 'EL-Sharqia', '14': 'Kaliobeya',
            '15': 'Kafr El-Sheikh', '16': 'Gharbia', '17': 'Monoufia', '18': 'El-Beheira',
            '19': 'Ismailia', '21': 'Giza', '22': 'Beni Suef', '23': 'Fayoum',
            '24': 'El-Menia', '25': 'Assiut', '26': 'Sohag', '27': 'Qena',
            '28': 'Aswan', '29': 'Luxor', '31': 'Red Sea', '32': 'New Valley',
            '33': 'Matrouh', '34': 'North Sinai', '35': 'South Sinai', '88': 'Foreign',
        }

        if not id_number or len(id_number) != 14:
            return {"status": "invalid id"}

        century_digit = int(id_number[0])
        year = int(id_number[1:3])
        month = int(id_number[3:5])
        day = int(id_number[5:7])
        governorate_code = id_number[7:9]
        gender_code = int(id_number[12:13])

        if century_digit == 2:
            full_year = 1900 + year
        elif century_digit == 3:
            full_year = 2000 + year
        else:
            return {"status": "invalid century digit"}

        gender = "Male" if gender_code % 2 != 0 else "Female"
        governorate = governorates.get(governorate_code, "Unknown")
        birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"

        return {
            "year": year,
            "month": month,
            "day": day,
            "Birth Date": birth_date,
            "Governorate": governorate,
            "Gender": gender,
        }