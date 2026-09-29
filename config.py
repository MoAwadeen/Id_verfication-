from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os


@dataclass
class Config:
    target_width: int
    target_height: int
    aspect_ratio: float

    crop_first_name_x1: int
    crop_first_name_y1: int
    crop_first_name_x2: int
    crop_first_name_y2: int

    crop_full_name_x1: int
    crop_full_name_y1: int
    crop_full_name_x2: int
    crop_full_name_y2: int

    crop_id_x1: int
    crop_id_y1: int
    crop_id_x2: int
    crop_id_y2: int

    yolo_model_path: str
    paddle_model_name: str

    yolo_conf_threshold: float
    contour_min_area_ratio: float
    contour_max_area_ratio: float
    contour_min_aspect_ratio: float
    contour_max_aspect_ratio: float

    supabase_url: str
    supabase_key: str
    thread_pool_workers: int


def load_config(env_path: str = None) -> Config:
    if env_path is None:
        base_dir = Path(__file__).parent
        env_path = base_dir / ".env"

    print(f"DEBUG: Loading config from {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)
    
    min_area = os.getenv("CONTOUR_MIN_AREA_RATIO", "0.05")
    max_area = os.getenv("CONTOUR_MAX_AREA_RATIO", "0.95")
    min_aspect = os.getenv("CONTOUR_MIN_ASPECT_RATIO", "1.2")
    max_aspect = os.getenv("CONTOUR_MAX_ASPECT_RATIO", "2.0")
    print(f"DEBUG: Env values - min_area={min_area}, max_area={max_area}, min_aspect={min_aspect}, max_aspect={max_aspect}")

    return Config(
        target_width=int(os.getenv("TARGET_WIDTH", "750")),
        target_height=int(os.getenv("TARGET_HEIGHT", "600")),
        aspect_ratio=float(os.getenv("ASPECT_RATIO", "1.586")),

        crop_first_name_x1=int(os.getenv("CROP_FIRST_NAME_X1", "520")),
        crop_first_name_y1=int(os.getenv("CROP_FIRST_NAME_Y1", "120")),
        crop_first_name_x2=int(os.getenv("CROP_FIRST_NAME_X2", "720")),
        crop_first_name_y2=int(os.getenv("CROP_FIRST_NAME_Y2", "210")),

        crop_full_name_x1=int(os.getenv("CROP_FULL_NAME_X1", "230")),
        crop_full_name_y1=int(os.getenv("CROP_FULL_NAME_Y1", "190")),
        crop_full_name_x2=int(os.getenv("CROP_FULL_NAME_X2", "720")),
        crop_full_name_y2=int(os.getenv("CROP_FULL_NAME_Y2", "270")),

        crop_id_x1=int(os.getenv("CROP_ID_X1", "300")),
        crop_id_y1=int(os.getenv("CROP_ID_Y1", "420")),
        crop_id_x2=int(os.getenv("CROP_ID_X2", "740")),
        crop_id_y2=int(os.getenv("CROP_ID_Y2", "540")),

        yolo_model_path=os.getenv("YOLO_MODEL_PATH", "weights/best.pt"),
        paddle_model_name=os.getenv("PADDLE_MODEL_NAME", "arabic_PP-OCRv5_mobile_rec"),

        yolo_conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", "0.3")),
        contour_min_area_ratio=float(os.getenv("CONTOUR_MIN_AREA_RATIO", "0.05")),
        contour_max_area_ratio=float(os.getenv("CONTOUR_MAX_AREA_RATIO", "0.95")),
        contour_min_aspect_ratio=float(os.getenv("CONTOUR_MIN_ASPECT_RATIO", "1.2")),
        contour_max_aspect_ratio=float(os.getenv("CONTOUR_MAX_ASPECT_RATIO", "2.0")),

        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_key=os.getenv("SUPABASE_KEY", ""),
        thread_pool_workers=int(os.getenv("THREAD_POOL_WORKERS", "4")),
    )


config = load_config()