from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from database import SupabaseClient
    from concurrent.futures import ThreadPoolExecutor
    from paddleocr import TextRecognition
    from ultralytics import YOLO
else:
    # Runtime stubs - will be overridden in main.py
    Config = object
    SupabaseClient = object
    ThreadPoolExecutor = object
    TextRecognition = object
    YOLO = object


def get_config() -> Config: ...
def get_db() -> SupabaseClient | None: ...
def get_executor() -> ThreadPoolExecutor: ...
def get_paddle_rec() -> TextRecognition: ...
def get_yolo_model() -> YOLO: ...