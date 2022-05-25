import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang="en")


def recognize(img):
    """Process a PIL image."""
    img_arr = np.array(img)
    # img_path = "https://raw.githubusercontent.com/alpha2phi/python-apps/main/ocr-ml-viewer/backend/test_images/image_1.png"
    result = ocr.ocr(img_arr, cls=True)
    if result is not None and len(result) > 0:
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        # img = Image.open(img_path).convert("RGB")
        processed_img = draw_ocr(img, boxes, txts, scores, font_path="./fonts/simfang.ttf")
        return {"boxes": boxes, "text": txts, "scores": scores}, processed_img
    return result, ""
