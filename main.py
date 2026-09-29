import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from config import load_config, Config
from database import SupabaseClient
from id_image import Id_Image
from routes.id_verification import router as verification_router
from dependencies import (
    get_config, get_db, get_executor, get_paddle_rec, get_yolo_model
)
from paddleocr import TextRecognition
from ultralytics import YOLO

_app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DEBUG: Starting lifespan, loading config...")
    cfg = load_config()
    print(f"DEBUG: Config loaded: supabase_url={cfg.supabase_url[:30] if cfg.supabase_url else 'None'}..., thread_pool_workers={cfg.thread_pool_workers}")
    print(f"DEBUG: Config contour params: min_area_ratio={cfg.contour_min_area_ratio}, max_area_ratio={cfg.contour_max_area_ratio}, min_aspect={cfg.contour_min_aspect_ratio}, max_aspect={cfg.contour_max_aspect_ratio}")
    _app_state["config"] = cfg

    executor = ThreadPoolExecutor(max_workers=cfg.thread_pool_workers)
    _app_state["executor"] = executor

    try:
        paddle_rec = TextRecognition(
            model_name=cfg.paddle_model_name,
            device="cpu",
        )
        _app_state["paddle_rec"] = paddle_rec
        print("PaddleOCR model loaded")
    except Exception as e:
        print(f"FATAL: Failed to load PaddleOCR model: {e}")
        raise

    try:
        yolo_model = YOLO(cfg.yolo_model_path)
        _app_state["yolo_model"] = yolo_model
        print("YOLO model loaded")
    except Exception as e:
        print(f"FATAL: Failed to load YOLO model: {e}")
        raise

    db = None
    if cfg.supabase_url and cfg.supabase_key:
        try:
            db = SupabaseClient(cfg.supabase_url, cfg.supabase_key, executor)
            _app_state["db"] = db
            print("Database: Connected")
        except Exception as e:
            print(f"FATAL: Failed to connect to Supabase: {e}")
            raise

    yield

    executor.shutdown(wait=True)
    _app_state.clear()


app = FastAPI(title="ID Verification API", lifespan=lifespan)


# Override the imported dependency functions (from dependencies module)
def _get_config() -> Config:
    cfg = _app_state.get("config")
    print(f"DEBUG: get_config called, config in state: {cfg is not None}")
    if cfg:
        print(f"DEBUG: get_config returning config with contour_min_area_ratio={cfg.contour_min_area_ratio}")
    else:
        print("DEBUG: get_config returning None!")
    return cfg


def _get_db() -> SupabaseClient | None:
    return _app_state.get("db")


def _get_executor() -> ThreadPoolExecutor:
    return _app_state["executor"]


def _get_paddle_rec() -> TextRecognition:
    return _app_state["paddle_rec"]


def _get_yolo_model() -> YOLO:
    return _app_state["yolo_model"]


app.dependency_overrides[get_config] = _get_config
app.dependency_overrides[get_db] = _get_db
app.dependency_overrides[get_executor] = _get_executor
app.dependency_overrides[get_paddle_rec] = _get_paddle_rec
app.dependency_overrides[get_yolo_model] = _get_yolo_model

app.include_router(verification_router)


if __name__ == "__main__":
    import uvicorn
    print("Starting ID Verification API on http://0.0.0.0:8000")
    print("Test with: curl -X POST http://localhost:8000/api/verify -H 'Content-Type: application/json' -d '{\"image_name\": \"file1.png\"}'")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)