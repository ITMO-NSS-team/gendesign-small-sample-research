import logging
import re
from pathlib import Path
from re import match

import cv2
import numpy as np
from paddleocr import PaddleOCR


def bbox_center(coords: list[tuple[float, float]]):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return (min(xs) + (max(xs) - min(xs)), min(ys) + (max(ys) - min(ys)))


def crop_four(img: np.ndarray):
    midy = img.shape[0] // 2
    midx = img.shape[1] // 2

    crops = []
    crops.append(img[0 : midy + 100, 0 : midx + 100, :])
    crops.append(img[0 : midy + 100, midx - 100 :, :])
    crops.append(img[midy - 100 :, 0 : midx + 100, :])
    crops.append(img[midy - 100 :, midx - 100 :, :])

    return crops


def get_closest_horiz_line_width(image: np.ndarray, bbox: tuple[float, float, float, float]):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 3, 0)
    edges_mid = cv2.Canny(image=img_blur, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(
        edges_mid, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )
    horiz_lines = []
    for i in range(lines.shape[0]):
        line = (lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
        if abs(line[1] - line[3]) == 0:
            horiz_lines.append(line)

    line_centers = np.array([(line[2] - line[0], line[1]) for line in horiz_lines])
    bbc = bbox_center(bbox)
    distances = np.linalg.norm(line_centers - bbc, axis=1)
    min_index = np.argmin(distances)

    return horiz_lines[min_index][2] - horiz_lines[min_index][0]


def match_units_text(res_ocr):
    val_unit_text, i = None, None
    if res_ocr:
        for i, res in enumerate(res_ocr):
            text = res[1][0].replace(" ", "")
            if match("\d+\s*(mm|nm|m)", res[1][0]):
                val_unit_text = text
                break
    return val_unit_text, i


def get_pixel_real_size(
    path: str,
) -> tuple[list[float, float], str]:
    """Finds real value, unit.

    Args:
        path (path): Image file path.

    Returns:
        tuple[float, str]: Real size of pixel side and its units.
    """
    reader = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    # search on full image
    res_ocr = reader.ocr(image)[0]
    val_units_text, pred_id = match_units_text(res_ocr)

    if val_units_text:
        value, unit = re.findall("[a-z]+|[0-9]+", val_units_text)
        line_length_px = get_closest_horiz_line_width(image, res_ocr[pred_id][0])
    else:  # search on crops
        crops = crop_four(image)
        for i, crop in enumerate(crops):
            res_ocr = reader.ocr(crop)[0]
            val_units_text, pred_id = match_units_text(res_ocr)
            if val_units_text:
                value, unit = re.findall("[a-z]+|[0-9]+", val_units_text)
                line_length_px = get_closest_horiz_line_width(crop, res_ocr[pred_id][0])

    unit = "μm" if unit == "m" else unit

    return int(value)/line_length_px, unit


if __name__ == "__main__":
    imgs_folder_path = "pixel_size\imgs"
    reg = Path(imgs_folder_path).glob("*")
    pathes = [p for p in reg]
    for p in pathes:
        res = get_pixel_real_size(str(r'pixel_size/imgs/18a.jpg'))
        print(p, res)