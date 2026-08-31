import cv2
import numpy as np
import os


class ImageProcessingMixin:
    """Perspective correction, deskew/contrast preprocessing, PaddleOCR invocation,
    and field cropping/saving."""

    def auto_perspective_correct(self, img, min_area_ratio=0.2):
        h, w = img.shape[:2]
        img_area = h * w
        if img_area == 0:
            return img, False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img, False

        doc_cnt = None
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < img_area * min_area_ratio:
                break
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Sanity check: a cheque's aspect ratio is roughly 2:1-2.6:1.
                # Without this, a false-positive quad from background clutter
                # (a table edge, a folder, a shadow) can pass the area/shape
                # checks and get warped as if it were the cheque - silently
                # wrecking an otherwise perfectly clear photo before OCR ever
                # runs. This is the most common cause of "clear image, still
                # nothing detected".
                rect = self._order_points(approx)
                (tl, tr, br, bl) = rect
                cand_w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
                cand_h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
                if cand_h == 0:
                    continue
                ar = cand_w / cand_h
                if not (1.6 <= ar <= 3.0):
                    continue
                doc_cnt = approx
                break

        if doc_cnt is None:
            return img, False

        rect = self._order_points(doc_cnt)
        (tl, tr, br, bl) = rect
        max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if max_width < 50 or max_height < 50:
            return img, False

        dst = np.array([
            [0, 0], [max_width - 1, 0],
            [max_width - 1, max_height - 1], [0, max_height - 1],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        return warped, True

    def preprocess_image(self, img):
        img, perspective_applied = self.auto_perspective_correct(img)
        rotated_img = img.copy()
        detected_angle = 0

        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        coords = np.column_stack(np.where(thresh > 0))
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle > 45:
                angle = angle - 90
            # Raised from 15 to 45 deg: real-world phone photos of cheques
            # are often shot at a deliberate angle rather than flat, and a
            # 15 deg cap left that print diagonal going into OCR, which
            # also breaks the position-based field heuristics downstream
            # (they assume roughly horizontal text). Gross 90/180/270 deg
            # rotation is still left to PaddleOCR's own orientation
            # classifier (use_angle_cls=True, set in __init__).
            if abs(angle) < 45:
                (h, w) = rotated_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(rotated_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                detected_angle = angle

        gray_rotated = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray_rotated)
        denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
        processed_ocr_img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return rotated_img, processed_ocr_img, detected_angle, perspective_applied

    def run_ocr(self, img, max_dimension=None, min_dimension=1600):
        # Default to self.det_limit_side_len (see __init__) instead of a
        # hardcoded value, so this pre-resize and PaddleOCR's own internal
        # detector limit always agree - one no longer silently undoes the
        # other.
        if max_dimension is None:
            max_dimension = self.det_limit_side_len

        h, w = img.shape[:2]
        longest_side = max(h, w)
        scale = 1.0
        ocr_img = img
        if longest_side > max_dimension:
            scale = max_dimension / float(longest_side)
            new_w = max(int(round(w * scale)), 1)
            new_h = max(int(round(h * scale)), 1)
            ocr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif longest_side < min_dimension:
            # Upscale floor: this pipeline only ever downscaled before, so
            # a modest-resolution photo (a typical phone shot well under
            # ~1600px, like a cheque held at arm's length rather than
            # macro-photographed) went into the detector at native size
            # with no help. Fine print (IFSC, account no.) at that size can
            # be only a few pixels tall - too small for reliable detection
            # regardless of how "clear"/in-focus the source shot is.
            # INTER_CUBIC gives smoother edges for the detector than a
            # plain nearest/linear upscale.
            scale = min_dimension / float(longest_side)
            new_w = max(int(round(w * scale)), 1)
            new_h = max(int(round(h * scale)), 1)
            ocr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        predictions = list(self.ocr.predict(ocr_img))
        extracted = []
        for res in predictions:
            if 'rec_texts' in res:
                for text, bbox_arr, score in zip(res['rec_texts'], res['rec_polys'], res['rec_scores']):
                    bbox = bbox_arr.tolist()
                    if scale != 1.0:
                        bbox = [[x / scale, y / scale] for x, y in bbox]
                    clean_text = text.strip()
                    if clean_text:
                        extracted.append({
                            "text": clean_text,
                            "bbox": bbox,
                            "confidence": score
                        })
        return extracted

    def crop_field(self, img, bbox, pad=2, expand_ratio=0.12, min_expand_px=4,
                   tight=False, tight_thresh=200, tight_pad=2):
        x0, y0, x1, y1 = self._bbox_rect(bbox)
        w_box = x1 - x0
        h_box = y1 - y0
        if w_box <= 0 or h_box <= 0:
            return None

        expand_x = max(w_box * expand_ratio, min_expand_px)
        expand_y = max(h_box * expand_ratio, min_expand_px)
        h, w = img.shape[:2]
        x0 = int(max(0, x0 - expand_x))
        y0 = int(max(0, y0 - expand_y))
        x1 = int(min(w, x1 + expand_x))
        y1 = int(min(h, y1 + expand_y))
        if x1 <= x0 or y1 <= y0:
            return None

        crop = img[y0:y1, x0:x1].copy()

        if tight:
            crop = self._tight_trim(crop, thresh=tight_thresh, pad=tight_pad)

        if pad > 0:
            crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        return crop

    def _tight_trim(self, crop, thresh=200, pad=2):
        if crop is None or crop.size == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, bin_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

        col_sums = bin_img.sum(axis=0)
        row_sums = bin_img.sum(axis=1)
        cols = np.where(col_sums > 0)[0]
        rows = np.where(row_sums > 0)[0]
        if len(cols) == 0 or len(rows) == 0:
            return crop

        h, w = gray.shape[:2]
        x0 = max(int(cols[0]) - pad, 0)
        x1 = min(int(cols[-1]) + pad + 1, w)
        y0 = max(int(rows[0]) - pad, 1)
        y1 = min(int(rows[-1]) + pad + 1, h)

        return crop[y0:y1, x0:x1]

    def crop_and_save(self, image, extracted_data, output_dir="streamlit_extracted_fields"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cropped_images = {}
        for field, data in extracted_data.items():
            if data and data['bbox']:
                crop = self.crop_field(image, data['bbox'])
                if crop is not None and crop.size > 0:
                    output_path = os.path.join(output_dir, f"{field}.png")
                    cv2.imwrite(output_path, crop)
                    cropped_images[field] = output_path
        return cropped_images