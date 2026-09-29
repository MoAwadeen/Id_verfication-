import pytest
import numpy as np
from pathlib import Path

from id_image import Id_Image, IDResult
from config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_image():
    img = np.zeros((800, 600, 3), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(img, (100, 100), (500, 400), (255, 255, 255), -1)
    return img


def test_config_loads(config):
    assert config.target_width == 750
    assert config.target_height == 600
    assert config.aspect_ratio == 1.586
    assert config.crop_first_name_x1 == 520


def test_id_image_init_with_array(config, sample_image):
    id_img = Id_Image(sample_image, config)
    assert id_img.image is not None
    assert id_img.image.shape == (800, 600, 3)


def test_id_image_init_with_path(config, tmp_path):
    cv2 = pytest.importorskip("cv2")
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    id_img = Id_Image(str(img_path), config)
    assert id_img.image is not None


def test_decode_id_valid(config):
    id_img = Id_Image(np.zeros((100, 100, 3), dtype=np.uint8), config)
    result = id_img._decode_id("29611091301456")

    assert result["Birth Date"] == "1996-11-09"
    assert result["Governorate"] == "EL-Sharqia"
    assert result["Gender"] == "Male"


def test_decode_id_invalid_length(config):
    id_img = Id_Image(np.zeros((100, 100, 3), dtype=np.uint8), config)
    result = id_img._decode_id("12345")
    assert result["status"] == "invalid id"


def test_decode_id_empty(config):
    id_img = Id_Image(np.zeros((100, 100, 3), dtype=np.uint8), config)
    result = id_img._decode_id("")
    assert result["status"] == "invalid id"


def test_idresult_dataclass():
    result = IDResult(
        first_name="Test",
        first_name_confidence=0.9,
        full_name="Test User",
        full_name_confidence=0.85,
        id_number="12345678901234",
        year=1990,
        month=1,
        day=1,
        birth_date="1990-01-01",
        governorate="Cairo",
        gender="Male",
        status=None,
    )
    assert result.first_name == "Test"
    assert result.id_number == "12345678901234"
    assert result.birth_date == "1990-01-01"
    assert result.governorate == "Cairo"
    assert result.gender == "Male"