import cv2
import numpy as np
import re
import os
from paddleocr import PaddleOCR


DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


class ChequeOCR:
    def __init__(self, det_limit_side_len=1920):
        # det_limit_side_len must be >= run_ocr()'s max_dimension. Without
        # this, PaddleOCR falls back to its own default detector resize
        # limit (960px on the long side in recent PP-OCR builds) and
        # silently shrinks every image to that before running detection -
        # regardless of how high-res/clear the source photo is. Small
        # print (IFSC code, account number) becomes sub-pixel and never
        # gets detected. Setting det_limit_type='max' + an explicit
        # det_limit_side_len makes the detector only downscale if the
        # image is actually larger than what we've already prepared.
        self.det_limit_side_len = det_limit_side_len
        self.ocr = PaddleOCR(
            lang='en',
            # Disabled: on this deployment box, oneDNN throws
            # NotImplementedError (ConvertPirAttribute2RuntimeAttribute
            # not support pir::ArrayAttribute<pir::DoubleAttribute>)
            # inside the text-detection conv op. This is a real
            # oneDNN/PaddleX PIR incompatibility, not something to tune
            # around - confirmed by testing, not just a cautious guess.
            # Leave this off unless/until you upgrade paddlepaddle and
            # verify the crash is gone.
            enable_mkldnn=False,
            # os.cpu_count() instead of a hardcoded 8 so this doesn't
            # over-subscribe (and thrash on context-switching) if this
            # ever runs in a smaller 2-4 vCPU container.
            cpu_threads=os.cpu_count() or 4,
            # Restores rotation handling cheaply. PaddleOCR's built-in
            # orientation classifier catches sideways/upside-down cheque
            # photos (0/90/180/270 deg). This is required because the
            # manual deskew in preprocess_image() below only corrects
            # fine tilt (<15 deg) - it cannot fix a gross rotation, and
            # there is no other rotation handling left in this file.
            use_angle_cls=True,
            # Brought back down from 0.75 to PaddleOCR's normal default.
            # 0.75 is faster but silently drops faint/faded ink boxes
            # (signatures, stamps, low-contrast print) below threshold -
            # not misread, just never detected. Validate against a batch
            # of your worst real cheque photos before raising this again.
            det_db_box_thresh=0.6,
            # See comment on self.det_limit_side_len above: without these
            # two, PaddleOCR's own default resize limit undoes any
            # resolution we preserved upstream and is the #1 cause of
            # "clear image, but fields still not detected".
            det_limit_side_len=det_limit_side_len,
            det_limit_type='max',
        )
        self.non_field_keywords = [
            'pay', 'bearer', 'rupees', 'order', 'valid for', 'a/c no', 'account no',
            'signature', 'please sign', 'or bearer',
            'cancelled', 'cancel', 'void', 'specimen', 'sample', 'not negotiable',
            'not valid', 'duplicate', 'copy', 'draft'
        ]
        self.ifsc_pattern = re.compile(r'^[A-Z]{4}0[A-Z0-9]{6}$')
        self.account_pattern = re.compile(r'^\d{9,18}$')

        self.confusion_subs = [
            ('O', '0'), ('0', 'O'),
            ('I', '1'), ('1', 'I'),
            ('S', '5'), ('5', 'S'),
            ('B', '8'), ('8', 'B'),
            ('Z', '2'), ('2', 'Z'),
        ]
        self.annotation_labels = {
            'bank_name', 'bank_address', 'ifsc_code', 'account_no', 'account_number',
        }
        self.bank_continuation_re = re.compile(
            r'\b(of|india|india\'s|limited|ltd|co\.?|corporation|corp|the)\b', re.IGNORECASE
        )
        self.branch_code_re = re.compile(r'\(\d{4,5}\)')
        self.person_initial_re = re.compile(r'\b[A-Z]\.\s+[A-Z]')
        self.person_title_re = re.compile(r'^(Mr|Mrs|Ms|Dr|Shri|Smt|Miss)\.?\s', re.IGNORECASE)
        self.known_bank_tokens = (
            'hdfc', 'icici', 'axis', 'sbi', 'syndicate', 'baroda', 'punjab', 'kotak',
            'indusind', 'canara', 'union', 'idbi', 'yes', 'federal', 'rbl', 'bandhan',
        )

    def _has_bank_evidence(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        if 'bank' in t:
            return True
        if self.bank_continuation_re.search(text):
            return True
        return any(token in t for token in self.known_bank_tokens)

    def _looks_like_person_name(self, text: str) -> bool:
        t = text.strip()
        if not t or self._has_bank_evidence(t):
            return False
        if self.person_initial_re.search(t):
            return True
        if self.person_title_re.match(t):
            return True
        words = t.split()
        if 2 <= len(words) <= 5:
            name_like = all(
                re.match(r'^[A-Z]\.?$', w) or re.match(r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$', w)
                for w in words
            )
            corp_markers = ('ltd', 'limited', 'corp', 'corporation', 'bank', 'india', 'co')
            if name_like and not any(m in t.lower() for m in corp_markers):
                return True
        return False

    def _looks_like_bank_name(self, text: str) -> bool:
        if not text or len(text.strip()) < 3:
            return False
        if self._looks_like_person_name(text):
            return False
        return self._has_bank_evidence(text)

    def _normalize_alpha(self, text: str) -> str:
        return re.sub(r'[^a-z]', '', text.lower())

    def _is_annotation_label(self, text: str) -> bool:
        normalized = re.sub(r'[^a-z_]', '', text.lower())
        return normalized in self.annotation_labels

    def _is_overlay_mark(self, text: str) -> bool:
        text_l = text.lower()
        direct_keywords = [
            'cancel', 'cancelled', 'void', 'specimen', 'sample',
            'not negotiable', 'duplicate', 'draft', 'not valid',
        ]
        if any(kw in text_l for kw in direct_keywords):
            return True
        cleaned = self._normalize_alpha(text)
        if len(cleaned) < 5:
            return False
        if cleaned.startswith('canc') and cleaned.endswith(('led', 'lled', 'ed')):
            return True
        if 'camcel' in cleaned or 'cancell' in cleaned or 'cancl' in cleaned:
            return True
        return False

    def is_cancelled(self, extracted) -> bool:
        cancel_keywords = ('cancelled', 'cancel', 'void')
        for item in extracted:
            text = item.get("text", "")
            text_l = text.lower()
            if any(kw in text_l for kw in cancel_keywords):
                return True
            cleaned = self._normalize_alpha(text)
            if len(cleaned) < 5:
                continue
            if cleaned.startswith('canc') and cleaned.endswith(('led', 'lled', 'ed')):
                return True
            if 'camcel' in cleaned or 'cancell' in cleaned or 'cancl' in cleaned:
                return True
        return False

    def detect_annotated_upload(self, extracted) -> bool:
        hits = sum(
            1 for item in extracted
            if self._is_annotation_label(item.get("text", ""))
        )
        return hits >= 2

    def _is_repetitive(self, text: str, min_repeats=3) -> bool:
        tokens = [t.lower() for t in re.findall(r'\b\w+\b', text)]
        if len(tokens) < min_repeats * 2:
            return False
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return max(counts.values()) >= min_repeats

    def _ifsc_candidates(self, raw: str):
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
        seen = {cleaned}
        yield cleaned
        for i, ch in enumerate(cleaned):
            for a, b in self.confusion_subs:
                if ch == a:
                    variant = cleaned[:i] + b + cleaned[i + 1:]
                    if variant not in seen:
                        seen.add(variant)
                        yield variant

    def _has_foreign_long_digit_run(self, text: str, code: str, min_run: int = 9) -> bool:
        """True if `text` contains a digit run of min_run+ characters that
        isn't part of the matched IFSC code itself. A real IFSC line never
        contains a run this long (its own digits are at most 6-7 chars);
        account numbers are 9-18 digits. This catches the case where OCR
        merges an 'A/c No.' label with the adjacent account-number digits
        into one line/box, which can coincidentally match the IFSC regex
        (4 letters + 0/O + alnum) purely from misread label characters."""
        for run in re.findall(r'\d+', text):
            if len(run) >= min_run and run not in code:
                return True
        return False

    def try_match_ifsc(self, raw: str):
        if not raw:
            return None

        search_pattern = re.compile(r'[A-Z]{4}[0O][A-Z0-9]{5,6}')

        for candidate in self._ifsc_candidates(raw):
            cleaned_stripped = re.sub(r'(RTGS|NEFT|IFSC|IFS|C0DE|CODE)', '', candidate)
            for target in (cleaned_stripped, candidate):
                match = search_pattern.search(target)
                if not match:
                    continue
                found = match.group(0)
                ifsc_code = found[:4] + '0' + found[5:]
                tail = ifsc_code[5:]
                if sum(c.isdigit() for c in tail) < 4:
                    continue
                return ifsc_code

        return None

    def _ifsc_token_bbox(self, ifsc_code, source):
        members = source.get("members") or [source]
        for member in members:
            if self.try_match_ifsc(member.get("text", "")) == ifsc_code:
                return member["bbox"]
            tight = re.sub(r'[^A-Za-z0-9]', '', member.get("text", "")).upper()
            if ifsc_code in tight:
                return member["bbox"]

        text = source.get("text", "")
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        idx = cleaned.find(ifsc_code)
        if idx >= 0:
            members_sorted = sorted(members, key=lambda m: self._bbox_left(m["bbox"])) if members else []
            offset = 0
            for member in members_sorted:
                m_cleaned = re.sub(r'[^A-Za-z0-9]', '', member.get("text", "")).upper()
                m_len = len(m_cleaned)
                if offset <= idx < offset + m_len:
                    xs = [p[0] for p in member["bbox"]]
                    ys = [p[1] for p in member["bbox"]]
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                    local_idx = idx - offset
                    span = max(m_len, 1)
                    char_w = (x1 - x0) / span
                    tx0 = x0 + local_idx * char_w
                    tx1 = min(tx0 + len(ifsc_code) * char_w, x1)
                    return [[tx0, y0], [tx1, y0], [tx1, y1], [tx0, y1]]
                offset += m_len
            overlap_bboxes = []
            offset = 0
            for member in members_sorted:
                m_cleaned = re.sub(r'[^A-Za-z0-9]', '', member.get("text", "")).upper()
                m_len = len(m_cleaned)
                if offset < idx + len(ifsc_code) and offset + m_len > idx:
                    overlap_bboxes.append(member["bbox"])
                offset += m_len
            if overlap_bboxes:
                xs = [p[0] for b in overlap_bboxes for p in b]
                ys = [p[1] for b in overlap_bboxes for p in b]
                return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        return source["bbox"]

    def _clean_bank_name(self, text: str) -> str:
        t = text
        t = re.sub(r'IFS?C?\s*CODE\s*[:\-]?\s*[A-Z]{4}0[A-Z0-9]{6}', ' ', t, flags=re.IGNORECASE)
        t = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', ' ', t)
        t = re.sub(r'SWIFT\s*:?', ' ', t, flags=re.IGNORECASE)
        t = re.sub(r'\(\d{4,5}\)[^,]*', ' ', t)
        t = re.sub(r'\b\d{6}\b', ' ', t)
        address_kw_pattern = r'\b(branch|road|marg|street|nagar|dist|district|vpo|teh|tehsil|near|opp|pin|pincode|tel|fax|gujrat|gujarat)\b.*'
        t = re.sub(address_kw_pattern, ' ', t, flags=re.IGNORECASE)
        t = re.sub(r'[^A-Za-z0-9\s\.\'-]', ' ', t)
        t = ' '.join(t.split())
        if 'bank' in t.lower():
            m = re.search(r'([A-Za-z][\w\s\.\'-]*bank[\w\s\.\'-]*)', t, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return t.strip()

    def _strip_devanagari(self, text: str) -> str:
        if not text:
            return text
        tokens = text.split()
        kept = [t for t in tokens if not DEVANAGARI_RE.search(t)]
        return ' '.join(kept).strip()

    def _is_mostly_devanagari(self, text: str, ratio=0.3) -> bool:
        if not text:
            return False
        hits = len(DEVANAGARI_RE.findall(text))
        return hits > len(text) * ratio

    def _is_probable_garbage(self, text: str, confidence: float = 1.0, conf_thresh: float = 0.55) -> bool:
        t = text.strip()
        if not t:
            return True
        if DEVANAGARI_RE.search(t):
            return False
        if re.search(r'[^\x00-\x7F]', t):
            return True
        letters = [c for c in t if c.isalpha()]
        if len(letters) >= 4:
            vowel_ratio = sum(1 for c in letters if c.lower() in 'aeiou') / len(letters)
            if vowel_ratio < 0.12 and confidence < conf_thresh:
                return True
        return False

    def _bank_name_parts_from_block(self, block):
        texts = []
        xs, ys = [], []
        for line in block:
            for member in line.get("members", [line]):
                txt = member.get("text", "")
                txt_l = txt.lower()
                if self._is_annotation_label(txt) or self._is_overlay_mark(txt):
                    continue
                if self._is_probable_garbage(txt, member.get("confidence", 1.0)):
                    continue
                if self.is_address(txt) or self.try_match_ifsc(txt):
                    continue
                if (
                    "bank" in txt_l
                    or self.bank_continuation_re.search(txt)
                    or re.search(r'\b(state|syndicate|hdfc|icici|axis|sbi|punjab|baroda)\b', txt_l)
                ):
                    texts.append(txt.strip())
                    xs.extend(p[0] for p in member["bbox"])
                    ys.extend(p[1] for p in member["bbox"])
        if xs:
            text = " ".join(t for t in texts if t)
            bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
            return text, bbox
        return None, None

    def _bank_name_bbox_from_block(self, block):
        _, bbox = self._bank_name_parts_from_block(block)
        return bbox

    def _watermark_bank_fallback(self, extracted):
        bank_tokens = [
            it for it in extracted
            if "bank" in it["text"].lower() and not self._is_annotation_label(it["text"])
        ]
        if len(bank_tokens) < 3:
            return None

        bank_tokens.sort(key=lambda it: (self._row_center(it), self._bbox_left(it["bbox"])))
        anchor = bank_tokens[0]
        anchor_y = self._row_center(anchor)
        anchor_h = max(self._row_height(anchor), 1)
        anchor_left, _, anchor_right, _ = self._bbox_rect(anchor["bbox"])

        same_row = [
            it for it in extracted
            if abs(self._row_center(it) - anchor_y) < anchor_h * 0.8
        ]
        same_row.sort(key=lambda it: self._bbox_left(it["bbox"]))
        anchor_idx = next((i for i, it in enumerate(same_row) if it is anchor), None)
        if anchor_idx is None:
            window = [anchor]
        else:
            window = same_row[max(0, anchor_idx - 1):anchor_idx + 2]
        if not window:
            return None

        text = " ".join(it["text"].strip() for it in window if it["text"].strip())
        xs = [p[0] for it in window for p in it["bbox"]]
        ys = [p[1] for it in window for p in it["bbox"]]
        bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        conf = sum(it["confidence"] for it in window) / max(len(window), 1)
        return {"text": text.strip(), "bbox": bbox, "confidence": conf}

    def _looks_like_bank_continuation(self, text: str) -> bool:
        txt_l = text.lower()
        if self.bank_continuation_re.search(text):
            return True
        if "bank" in txt_l:
            return True
        return len(text.strip()) <= 20 and sum(c.isalpha() for c in text) >= 2

    def is_address(self, text: str) -> bool:
        text_lower = text.lower()
        if self._is_overlay_mark(text) or self._is_annotation_label(text):
            return False
        if self._is_repetitive(text) or any(kw in text_lower for kw in self.non_field_keywords):
            return False
        grid_keywords = ['valid for', 'months only', 'cash transaction', 'd d m m', 'y y y y']
        if any(kw in text_lower for kw in grid_keywords):
            return False
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if digits > 0 and letters == 0:
            return False
        if re.search(r'\b\d{6}\b', text) or re.search(r'\b\d{3}\s\d{3}\b', text):
            return True
        keywords = [
            'branch', 'road', 'marg', 'street', 'nagar', 'dist', 'district', 'plot',
            'complex', 'floor', 'bldg', 'building', 'mumbai', 'delhi', 'bangalore',
            'bengaluru', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'sector',
            'opp', 'near', 'post', 'pin', 'pincode', 'tel', 'fax', 'hub', 'centre',
            'center', 'chowk', 'lane', 'estate', 'park', 'cross', 'extn', 'extension', 'main'
        ]
        if any(kw in text_lower for kw in keywords):
            return True
        if self.branch_code_re.search(text):
            return True
        if ',' in text and re.search(r'\d+', text):
            return True
        return False

    def _order_points(self, pts):
        pts = pts.reshape(4, 2).astype("float32")
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

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

    def _bbox_rect(self, bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    def _bbox_left(self, bbox):
        return self._bbox_rect(bbox)[0]

    def _bbox_right(self, bbox):
        return self._bbox_rect(bbox)[2]

    def _row_center(self, item):
        ys = [p[1] for p in item["bbox"]]
        return sum(ys) / len(ys)

    def _row_height(self, item):
        ys = [p[1] for p in item["bbox"]]
        return max(ys) - min(ys)

    def merge_same_line(self, extracted, overlap_ratio_threshold=0.5, max_x_gap_ratio=4.0, max_x_gap_width_ratio=0.12):
        if not extracted:
            return []
        all_xs = [p[0] for it in extracted for p in it["bbox"]]
        all_ys = [p[1] for it in extracted for p in it["bbox"]]
        width_est = max(all_xs) if all_xs else 1
        height_est = max(all_ys) if all_ys else 1
        items = sorted(extracted, key=self._row_center)
        lines = []
        for item in items:
            item_top = min(p[1] for p in item["bbox"])
            item_bottom = max(p[1] for p in item["bbox"])
            item_left = min(p[0] for p in item["bbox"])
            item_right = max(p[0] for p in item["bbox"])
            item_height = max(item_bottom - item_top, 1)
            item_y_ratio = self._row_center({"bbox": item["bbox"]}) / max(height_est, 1)

            placed = False
            for line in lines:
                overlap = min(line["bottom"], item_bottom) - max(line["top"], item_top)
                shorter = min(item_height, line["bottom"] - line["top"])
                if overlap <= 0 or overlap / shorter < overlap_ratio_threshold:
                    continue
                if item_left > line["right"]:
                    x_gap = item_left - line["right"]
                elif line["left"] > item_right:
                    x_gap = line["left"] - item_right
                else:
                    x_gap = 0
                height_based_cap = max(item_height, line["bottom"] - line["top"], 10) * max_x_gap_ratio
                width_based_cap = width_est * max_x_gap_width_ratio
                max_allowed_gap = min(height_based_cap, width_based_cap)
                line_y_ratio = ((line["top"] + line["bottom"]) / 2) / max(height_est, 1)
                if item_y_ratio < 0.28 or line_y_ratio < 0.28:
                    max_allowed_gap = min(max_allowed_gap, width_est * 0.045)
                if x_gap > max_allowed_gap:
                    continue
                line["members"].append(item)
                line["top"] = min(line["top"], item_top)
                line["bottom"] = max(line["bottom"], item_bottom)
                line["left"] = min(line["left"], item_left)
                line["right"] = max(line["right"], item_right)
                placed = True
                break
            if not placed:
                lines.append({"top": item_top, "bottom": item_bottom,
                               "left": item_left, "right": item_right, "members": [item]})

        merged = []
        for line in lines:
            line_sorted = sorted(line["members"], key=lambda it: self._bbox_left(it["bbox"]))
            text = " ".join(it["text"].strip() for it in line_sorted)
            text_tight = "".join(it["text"].strip() for it in line_sorted)
            xs = [p[0] for it in line_sorted for p in it["bbox"]]
            ys = [p[1] for it in line_sorted for p in it["bbox"]]
            bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
            conf = sum(it["confidence"] for it in line_sorted) / len(line_sorted)
            merged.append({
                "text": text, "text_tight": text_tight, "bbox": bbox,
                "confidence": conf, "members": line_sorted,
            })
        merged.sort(key=self._row_center)
        return merged

    def merge_address_block(self, lines, img_height, max_gap_ratio=0.05):
        candidates = [
            l for l in lines
            if self._row_center(l) / img_height < 0.30
            and not self._is_overlay_mark(l["text"])
            and not self._is_annotation_label(l["text"])
            and not self._is_probable_garbage(l["text"], l["confidence"])
        ]
        candidates.sort(key=self._row_center)
        blocks = []
        current = []
        for l in candidates:
            if not self.is_address(l["text"]) and not self.branch_code_re.search(l["text"]):
                continue
            if not current:
                current = [l]
                continue
            gap = self._row_center(l) - self._row_center(current[-1])
            if gap / img_height <= max_gap_ratio:
                current.append(l)
            else:
                blocks.append(current)
                current = [l]
        if current:
            blocks.append(current)

        results = []
        for block in blocks:
            text = ", ".join(b["text"].strip() for b in block if b["text"].strip())
            xs = [p[0] for b in block for p in b["bbox"]]
            ys = [p[1] for b in block for p in b["bbox"]]
            bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
            conf = sum(b["confidence"] for b in block) / len(block)
            results.append({"text": text, "bbox": bbox, "confidence": conf})
        return results

    def _extract_address_by_geometry(self, lines, bank_bbox, ifsc_bbox, width_est, pad_ratio=0.35, x_limit_ratio=0.62):
        if not bank_bbox or not ifsc_bbox:
            return None

        _, bank_top, _, bank_bottom = self._bbox_rect(bank_bbox)
        ifsc_left, ifsc_top, _, _ = self._bbox_rect(ifsc_bbox)
        if ifsc_top <= bank_bottom:
            return None

        band_height = max(ifsc_top - bank_bottom, 1)
        pad = band_height * pad_ratio
        y_start = bank_bottom - pad
        y_end = ifsc_top + pad
        x_limit = width_est * x_limit_ratio

        candidates = []
        for line in lines:
            raw_txt = line["text"].strip()
            if not raw_txt:
                continue
            if self._is_overlay_mark(raw_txt) or self._is_annotation_label(raw_txt):
                continue
            if self._is_probable_garbage(raw_txt, line["confidence"]):
                continue
            if any(kw in raw_txt.lower() for kw in self.non_field_keywords):
                continue

            txt = raw_txt
            if self.try_match_ifsc(raw_txt):
                stripped = re.sub(r'IFS?C?\s*:?\s*[A-Z0-9]{6,11}', '', raw_txt, flags=re.IGNORECASE)
                stripped = re.sub(r'SWIFT\s*:?\s*[A-Z0-9]*', '', stripped, flags=re.IGNORECASE)
                stripped = stripped.strip(' :,-')
                if len(stripped) < 5:
                    continue
                txt = stripped

            cy = self._row_center(line)
            cx = self._bbox_left(line["bbox"])
            if not (y_start <= cy <= y_end) or cx > x_limit:
                continue
            candidates.append({**line, "text": txt})

        if not candidates:
            return None

        candidates.sort(key=self._row_center)
        text = ", ".join(c["text"].strip() for c in candidates if c["text"].strip())
        xs = [p[0] for c in candidates for p in c["bbox"]]
        ys = [p[1] for c in candidates for p in c["bbox"]]
        bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        conf = sum(c["confidence"] for c in candidates) / len(candidates)
        return {"text": text, "bbox": bbox, "confidence": conf}

    def merge_header_block(self, lines, img_height, max_gap_ratio=0.10, y_limit=0.25,
                            height_ratio_thresh=0.55):
        all_xs = [p[0] for l in lines for p in l["bbox"]]
        width_est = max(all_xs) if all_xs else 1
        x_limit = 0.58
        candidates = [
            l for l in lines
            if self._row_center(l) / img_height < y_limit
            and self._bbox_left(l["bbox"]) / width_est < x_limit
            and not self._is_annotation_label(l["text"])
            and not self._is_overlay_mark(l["text"])
            and not self._is_probable_garbage(l["text"], l["confidence"])
            and (not self.is_address(l["text"]) or self._has_bank_evidence(l["text"]))
            and not self.try_match_ifsc(l["text"])
            and not self.branch_code_re.search(l["text"])
            and not self._looks_like_person_name(l["text"])
            and not any(kw in l["text"].lower() for kw in self.non_field_keywords)
        ]
        bankish = [l for l in candidates if self._has_bank_evidence(l["text"])]
        if not bankish:
            return None, []
        candidates = bankish

        anchor = max(candidates, key=self._row_height)
        anchor_height = max(self._row_height(anchor), 1)

        block = [anchor]
        remaining = [l for l in candidates if l is not anchor]

        changed = True
        while changed and remaining:
            changed = False
            for l in list(remaining):
                gap = min(abs(self._row_center(l) - self._row_center(b)) for b in block)
                if gap / img_height > max_gap_ratio:
                    continue
                txt = l["text"]
                height_ok = self._row_height(l) >= anchor_height * height_ratio_thresh
                keyword_continuation = bool(
                    self.bank_continuation_re.search(txt) or "bank" in txt.lower()
                )
                if height_ok or keyword_continuation:
                    block.append(l)
                    remaining.remove(l)
                    changed = True

        block.sort(key=self._row_center)
        raw_text, bbox = self._bank_name_parts_from_block(block)
        if not raw_text or not bbox:
            return None, []
        text = self._clean_bank_name(raw_text) or raw_text
        if not self._looks_like_bank_name(text):
            return None, []
        conf = sum(b["confidence"] for b in block) / len(block)
        return {"text": text, "bbox": bbox, "confidence": conf}, block

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

    def _overlay_exclusion_zones(self, extracted):
        zones = []
        for item in extracted:
            if self._is_overlay_mark(item["text"]) or self._is_annotation_label(item["text"]):
                xs = [p[0] for p in item["bbox"]]
                ys = [p[1] for p in item["bbox"]]
                zones.append((min(xs), min(ys), max(xs), max(ys)))
        return zones

    def _in_exclusion_zone(self, bbox, zones, overlap_thresh=0.3):
        if not zones:
            return False
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        b_area = max(bx1 - bx0, 1) * max(by1 - by0, 1)
        for zx0, zy0, zx1, zy1 in zones:
            ix0, iy0 = max(bx0, zx0), max(by0, zy0)
            ix1, iy1 = min(bx1, zx1), min(by1, zy1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / b_area > overlap_thresh:
                    return True
        return False

    def _looks_like_payer_name_text(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        tl = t.lower()
        block_kw = ('rupees', 'signature', 'valid for', 'a/c', 'account',
                    'ifsc', 'branch', 'cheque', 'chq', 'date', 'dd mm yyyy')
        if any(kw in tl for kw in block_kw):
            return False
        if self._is_overlay_mark(t) or self._is_annotation_label(t):
            return False
        return True

    def extract_payer_name(self, extracted, img_height):
        pay_anchor = None
        for item in extracted:
            t = item["text"].strip().lower().rstrip(':')
            if t != 'pay':
                continue
            y_ratio = self._row_center(item) / img_height if img_height else 0
            if y_ratio > 0.6:
                continue
            if pay_anchor is None or self._row_center(item) < self._row_center(pay_anchor):
                pay_anchor = item

        if pay_anchor is None:
            return None

        anchor_y = self._row_center(pay_anchor)
        anchor_h = max(self._row_height(pay_anchor), 1)
        anchor_right = self._bbox_rect(pay_anchor["bbox"])[2]

        same_row = [
            it for it in extracted
            if it is not pay_anchor
            and abs(self._row_center(it) - anchor_y) < anchor_h * 1.1
            and self._bbox_left(it["bbox"]) >= anchor_right - anchor_h * 0.5
        ]
        same_row.sort(key=lambda it: self._bbox_left(it["bbox"]))

        name_parts = []
        for it in same_row:
            t = it["text"].strip()
            tl = t.lower()
            if re.match(r'^or\b', tl) or 'bearer' in tl or re.search(r'\border\b', tl):
                break
            if not self._looks_like_payer_name_text(t):
                continue
            name_parts.append(it)

        if not name_parts:
            return None

        text = " ".join(p["text"].strip() for p in name_parts).strip(' .:-')
        if not text:
            return None
        xs = [p[0] for it in name_parts for p in it["bbox"]]
        ys = [p[1] for it in name_parts for p in it["bbox"]]
        bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        conf = sum(it["confidence"] for it in name_parts) / len(name_parts)
        return {"text": text, "bbox": bbox, "confidence": conf}

    def classify_and_extract(self, extracted, img_height):
        cancelled = self.is_cancelled(extracted)

        exclusion_zones = self._overlay_exclusion_zones(extracted)
        extracted = [it for it in extracted if not self._in_exclusion_zone(it["bbox"], exclusion_zones)]
        lines = self.merge_same_line(extracted)

        all_xs = [p[0] for it in extracted for p in it["bbox"]] if extracted else [1]
        width_est = max(all_xs) if all_xs else 1
        best = {}

        def consider(field_type, text, bbox, confidence):
            if field_type not in best or confidence > best[field_type]["confidence"]:
                best[field_type] = {"text": text, "bbox": bbox, "confidence": confidence, "field_type": field_type}

        def consider_ifsc(source, confidence):
            raw = source.get("text_tight", source.get("text", ""))
            cand = self.try_match_ifsc(raw) or self.try_match_ifsc(source["text"])
            if not cand:
                return
            if any(kw in source["text"].lower() for kw in ('a/c', 'a c no', 'acc no', 'account no')):
                return
            if self._has_foreign_long_digit_run(raw, cand) or self._has_foreign_long_digit_run(source["text"], cand):
                return
            bbox = self._ifsc_token_bbox(cand, source)
            consider("ifsc_code", cand, bbox, confidence)

        anchor_re = re.compile(r'IFS|SWIFT', re.IGNORECASE)
        for item in extracted:
            if anchor_re.search(item["text"]):
                consider_ifsc(item, 1.5)
        for line in lines:
            if anchor_re.search(line["text"]):
                consider_ifsc(line, 1.5)

        for item in extracted:
            if self.try_match_ifsc(item["text"]):
                consider_ifsc(item, item["confidence"])
        if "ifsc_code" not in best:
            for line in lines:
                if self.try_match_ifsc(line.get("text_tight", line["text"])):
                    consider_ifsc(line, line["confidence"])

        account_candidates = []
        def add_account_candidate(digits_only, bbox, confidence, y_ratio):
            if self.account_pattern.match(digits_only) and 0.15 < y_ratio < 0.90:
                account_candidates.append({
                    "text": digits_only, "bbox": bbox, "confidence": confidence,
                    "length": len(digits_only)
                })

        for item in extracted:
            digits_only = re.sub(r'[\s\-\.,]', '', item["text"])
            y_ratio = self._row_center(item) / img_height if img_height > 0 else 0
            add_account_candidate(digits_only, item["bbox"], item["confidence"], y_ratio)

        for line in lines:
            digits_only = re.sub(r'[\s\-\.,]', '', line.get("text_tight", line["text"]))
            y_ratio = self._row_center(line) / img_height if img_height > 0 else 0
            add_account_candidate(digits_only, line["bbox"], line["confidence"], y_ratio)

        if account_candidates:
            account_candidates.sort(key=lambda x: (x["length"], x["confidence"]), reverse=True)
            top_cand = account_candidates[0]
            consider("account_no", top_cand["text"], top_cand["bbox"], top_cand["confidence"])

        bank_candidates = []
        valid_bank_keywords = ['bank', 'sbi', 'hdfc', 'icici', 'axis', 'kotak', 'punjab', 'syndicate', 'baroda', 'canara', 'maharashtra', 'union', 'indian']



        for line in lines:
            txt = line["text"].strip()
            txt_l = txt.lower()

            if self._is_overlay_mark(txt) or self._is_annotation_label(txt):
                continue
            if any(kw in txt_l for kw in self.non_field_keywords):
                continue
            if "private limited" in txt_l or "pvt ltd" in txt_l:
                continue
            if self._looks_like_person_name(txt):
                continue

            if self._has_bank_evidence(txt):
                bank_members = []
                for member in line.get("members", [line]):
                    m_txt = member.get("text", "").strip()
                    if self.is_address(m_txt) or self.branch_code_re.search(m_txt):
                        continue
                    bank_members.append(member)

                if bank_members:
                    combined_txt = " ".join(m["text"] for m in bank_members)
                    cleaned = self._clean_bank_name(combined_txt)

                    if cleaned and any(bk in cleaned.lower() for bk in valid_bank_keywords):
                        xs = [p[0] for m in bank_members for p in m["bbox"]]
                        ys = [p[1] for m in bank_members for p in m["bbox"]]
                        tight_bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]

                        avg_conf = sum(m["confidence"] for m in bank_members) / len(bank_members)
                        bank_candidates.append({
                            "text": cleaned,
                            "bbox": tight_bbox,
                            "confidence": avg_conf
                        })

        if bank_candidates:
            best_bank = max(bank_candidates, key=lambda x: x["confidence"])
            consider("bank_name", best_bank["text"], best_bank["bbox"], best_bank["confidence"] + 0.5)

        ifsc_text = best.get("ifsc_code", {}).get("text", "")

        geo_addr = self._extract_address_by_geometry(
            lines,
            best.get("bank_name", {}).get("bbox"),
            best.get("ifsc_code", {}).get("bbox"),
            width_est,
        )
        if geo_addr and geo_addr["text"]:
            addr_text = geo_addr["text"]
            addr_text = re.sub(r'IFS?C?\s*CODE\s*[:\-]?\s*[A-Z0-9]+', '', addr_text, flags=re.IGNORECASE)
            addr_text = re.sub(r'SWIFT\s*:?\s*[A-Z0-9]*', '', addr_text, flags=re.IGNORECASE)
            consider("bank_address", addr_text.strip(), geo_addr["bbox"], geo_addr["confidence"] + 0.3)

        remaining_lines = [
            l for l in lines
            if not self._is_overlay_mark(l["text"])
            and not self._is_annotation_label(l["text"])
            and l["text"] != ifsc_text
        ]

        address_candidates = []
        for line in remaining_lines:
            txt = line["text"]
            txt_l = txt.lower()
            if any(kw in txt_l for kw in self.non_field_keywords):
                continue
            if "private limited" in txt_l or "pvt ltd" in txt_l:
                continue
            if self._is_probable_garbage(txt, line["confidence"]):
                continue
            if self.is_address(txt) or self.branch_code_re.search(txt):
                address_candidates.append(line)

        if address_candidates:
            address_candidates.sort(key=self._row_center)
            anchor = max(
                address_candidates,
                key=lambda b: (
                    re.search(r'\b\d{6}\b', b["text"]) is not None,
                    any(kw in b["text"].lower() for kw in ['dist', 'nagar', 'road', 'vpo', 'teh', 'marg', 'street']),
                    -1 if ("tel" in b["text"].lower() or "fax" in b["text"].lower()) else 0,
                    len(b["text"])
                ),
            )
            anchor_y = self._row_center(anchor)
            max_gap = max(self._row_height(anchor), 10) * 3.0
            block = [
                l for l in address_candidates
                if abs(self._row_center(l) - anchor_y) <= max_gap
                and not ("tel" in l["text"].lower() or "fax" in l["text"].lower())
            ]
            if not block:
                block = [anchor]
            block.sort(key=self._row_center)

            addr_text = ", ".join(b["text"].strip() for b in block if b["text"].strip())
            xs = [p[0] for b in block for p in b["bbox"]]
            ys = [p[1] for b in block for p in b["bbox"]]
            addr_bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
            addr_conf = sum(b["confidence"] for b in block) / len(block)

            addr_text = re.sub(r'^([a-zA-Z]{1,2}\s+){1,4}', '', addr_text)
            addr_text = re.sub(r'IFS?C?\s*CODE\s*[:\-]?\s*[A-Z0-9]+', '', addr_text, flags=re.IGNORECASE)
            addr_text = re.sub(r'SWIFT\s*:?\s*[A-Z0-9]*', '', addr_text, flags=re.IGNORECASE)
            addr_text = addr_text.strip()

            consider("bank_address", addr_text, addr_bbox, addr_conf)

        for field_type in ("bank_name", "bank_address"):
            if field_type in best:
                original = best[field_type]["text"]
                cleaned = self._strip_devanagari(original)
                best[field_type]["text"] = cleaned if cleaned else original

        for key in best:
            if "text" in best[key] and isinstance(best[key]["text"], str):
                best[key]["text"] = best[key]["text"].upper()   

        if not cancelled:
            payer = self.extract_payer_name(extracted, img_height)
            if payer and payer["text"]:
                consider("payer_name", payer["text"], payer["bbox"], payer["confidence"])

        return best, cancelled

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
