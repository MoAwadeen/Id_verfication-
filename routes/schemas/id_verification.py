from pydantic import BaseModel
from typing import Optional


class VerifyRequest(BaseModel):
    image_name: str


class VerifyResponse(BaseModel):
    status: str
    id_number: Optional[str] = None
    reason: Optional[str] = None
    detail: Optional[str] = None