from fastapi import APIRouter, Depends
from routes.schemas.id_verification import VerifyRequest, VerifyResponse
from id_image import Id_Image
from database import SupabaseClient
from config import Config
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from paddleocr import TextRecognition
from ultralytics import YOLO
from dependencies import (
    get_config, get_db, get_executor, get_paddle_rec, get_yolo_model
)

router = APIRouter(prefix="/api", tags=["verification"])


@router.post("/verify", response_model=VerifyResponse)
async def verify_id(
    request: VerifyRequest,
    cfg: Config = Depends(get_config),
    db: SupabaseClient | None = Depends(get_db),
    executor: ThreadPoolExecutor = Depends(get_executor),
    paddle_rec: TextRecognition = Depends(get_paddle_rec),
    yolo_model: YOLO = Depends(get_yolo_model),
):
    print(f"DEBUG: verify_id called, cfg is None: {cfg is None}")
    if cfg:
        print(f"DEBUG: cfg.contour_min_area_ratio = {cfg.contour_min_area_ratio}")
    
    IMAGES_DIR = Path(__file__).parent.parent / "images"
    image_path = IMAGES_DIR / request.image_name

    if not image_path.exists():
        return VerifyResponse(status="rejected", reason="not_found")

    id_img = Id_Image(str(image_path), cfg, paddle_rec, yolo_model, executor)
    result = await id_img.process()

    if result.status is not None:
        return VerifyResponse(
            status="retry",
            reason="invalid_id",
            detail=result.status
        )

    if db is None:
        return VerifyResponse(
            status="rejected",
            reason="problem with DB"
        )

    success, msg = await db.save_id_result(result)
    if success:
        return VerifyResponse(status="verified", id_number=result.id_number)
    if "already exists" in msg:
        return VerifyResponse(status="rejected", reason="duplicate", id_number=result.id_number)

    return VerifyResponse(status="retry", reason="db_error", detail=msg)