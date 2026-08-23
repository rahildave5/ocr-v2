import cv2
import numpy as np
import re
import os
from paddleocr import PaddleOCR

# Disable oneDNN to prevent crashes on CPU
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'

# Devanagari Unicode block (Hindi and other Indic scripts sharing it)
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


class ChequeOCR:
    def __init__(self):
        # Initialize PaddleOCR with English language and disabled MKLDNN/oneDNN
        self.ocr = PaddleOCR(lang='en', enable_mkldnn=False)
        self.non_field_keywords = [
            'pay', 'bearer', 'rupees', 'order', 'valid for', 'a/c no', 'account no',
            'signature', 'please sign', 'or bearer',
            # Handwritten overlays / stamps that can land anywhere on the cheque and
            # should never be mistaken for a printed field (bank name, address, etc).
            'cancelled', 'cancel', 'void', 'specimen', 'sample', 'not negotiable',
            'not valid', 'duplicate', 'copy', 'draft'
        ]
        # Regex patterns
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
        """Detect payee / account-holder names that must never become bank_name."""
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
        """Detect CANCELLED / VOID and common OCR misspellings (e.g. camcelled)."""
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

    def detect_annotated_upload(self, extracted) -> bool:
        """True when OCR reads Streamlit overlay labels baked into the image."""
        hits = sum(
            1 for item in extracted
            if self._is_annotation_label(item.get("text", ""))
        )
        return hits >= 2

    def _is_repetitive(self, text: str, min_repeats=3) -> bool:
    # Use re.findall to extract words only, ignoring standalone punctuation like colons
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

    def try_match_ifsc(self, raw: str):
        if not raw:
            return None

        # Clean string to uppercase alphanumeric characters only
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw).upper()

        # 1. Global Label Destruction: Erase these words anywhere in the string 
        # so OCR merging (e.g., 'IFSCC0DE') can NEVER create overlapping false matches.
        cleaned_stripped = re.sub(r'(RTGS|NEFT|IFSC|IFS|C0DE|CODE)', '', cleaned)

        # 2. Search Pattern: 4 letters + 0/O + 5 to 6 alphanumeric chars
        # (Relaxed to 5-6 trailing chars to catch the 10-char typo on this dummy cheque)
        search_pattern = re.compile(r'[A-Z]{4}[0O][A-Z0-9]{5,6}')

        # Try stripped version first, fallback to original if needed
        for target in (cleaned_stripped, cleaned):
            match = search_pattern.search(target)
            if match:
                found = match.group(0)
                # Normalize the 5th character to a numeric '0'
                ifsc_code = found[:4] + '0' + found[5:]
                return ifsc_code

        return None

    def _ifsc_token_bbox(self, ifsc_code, source):
        """Narrow bbox to the OCR token that actually contains the IFSC code."""
        members = source.get("members") or [source]
        for member in members:
            if self.try_match_ifsc(member.get("text", "")) == ifsc_code:
                return member["bbox"]
            tight = re.sub(r'[^A-Za-z0-9]', '', member.get("text", "")).upper()
            if ifsc_code in tight:
                return member["bbox"]

        # Fallback: locate which member token the match actually falls inside by
        # walking members in x-order and tracking cumulative cleaned-text offsets,
        # then use THAT member's own bbox/width to interpolate. Do not interpolate
        # over the whole merged line's bbox using a single uniform char width —
        # on a line that mixes an address (wide, sparse) with a short code, that
        # assumption misplaces the box entirely (correct text, wrong crop).
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
            # No single member covers the whole match (code was split across
            # member boundaries) — union just the members whose ranges overlap
            # the match span, instead of the entire line.
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
        # Strip IFSC and SWIFT codes
        t = re.sub(r'IFS?C?\s*CODE\s*[:\-]?\s*[A-Z]{4}0[A-Z0-9]{6}', ' ', t, flags=re.IGNORECASE)
        t = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', ' ', t)
        t = re.sub(r'SWIFT\s*:?', ' ', t, flags=re.IGNORECASE)
        
        # Strip branch codes like (09164) and 6-digit pincodes
        t = re.sub(r'\(\d{4,5}\)[^,]*', ' ', t)
        t = re.sub(r'\b\d{6}\b', ' ', t)
        
        # Strip trailing address words if merged
        address_kw_pattern = r'\b(branch|road|marg|street|nagar|dist|district|vpo|teh|tehsil|near|opp|pin|pincode|tel|fax|gujrat|gujarat)\b.*'
        t = re.sub(address_kw_pattern, ' ', t, flags=re.IGNORECASE)
        
        t = re.sub(r'[^A-Za-z0-9\s\.\'-]', ' ', t)
        t = ' '.join(t.split())
        
        if 'bank' in t.lower():
            m = re.search(r'([A-Za-z][\w\s\.\'-]*bank[\w\s\.\'-]*)', t, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return t.strip()
    # ---------- Devanagari / non-English filtering ----------

    def _strip_devanagari(self, text: str) -> str:
        """Drop any whitespace-separated token containing Devanagari script chars,
        keep the rest in original order. Used so mixed English/Hindi OCR lines
        (e.g. 'Faithful SyndicateBank, ... HYDERABAD - 500001') never leak the
        Hindi portion into a returned field value."""
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

    # ----------------------------------------------------------

    def _bank_name_parts_from_block(self, block):
        """Text and bbox from the same bank-name tokens so crop matches extracted text."""
        texts = []
        xs, ys = [], []
        for line in block:
            for member in line.get("members", [line]):
                txt = member.get("text", "")
                txt_l = txt.lower()
                if self._is_annotation_label(txt) or self._is_overlay_mark(txt):
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
        """Last-resort bank-name source for cheque crops that don't include the
        printed header/logo at all. Many cheques carry a repeating security
        watermark microprint along the page (e.g. "...AXIS BANK LTD AXIS BANK
        LTD AXIS BANK LTD...") — if that's the only text mentioning "bank" and
        it genuinely repeats several times, treat one repeat as the bank name.
        Returns a dict with a TIGHT bbox around a single occurrence (not the
        whole smeared watermark strip), or None if no real repeat exists (so a
        one-off stray mention of "bank" elsewhere is never misused)."""
        bank_tokens = [
            it for it in extracted
            if "bank" in it["text"].lower() and not self._is_annotation_label(it["text"])
        ]
        if len(bank_tokens) < 3:
            return None  # not a repeating watermark, just a stray mention

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
            # One neighbor token on each side is enough to capture a typical
            # "<name> BANK <suffix>" repeat unit without pulling in the next
            # cycle of the watermark.
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
        # NEW: reject date-grid / validity boilerplate that otherwise satisfies the
        # comma+digits or keyword heuristics below (e.g. "D D M M Y Y Y Y",
        # "VALID FOR 3 MONTHS ONLY", "NOT FOR CASH TRANSACTION ONLY").
        grid_keywords = ['valid for', 'months only', 'cash transaction', 'd d m m', 'y y y y']
        if any(kw in text_lower for kw in grid_keywords):
            return False
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if digits > 0 and letters == 0:
            return False  # pure digit/letter grids like "D D M M Y Y Y Y" strip to nothing useful
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

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        coords = np.column_stack(np.where(thresh > 0))
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle > 45:
                angle = angle - 90
            if abs(angle) < 15:
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated_img = img.copy()
        else:
            rotated_img = img.copy()

        gray_rotated = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray_rotated, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(denoised)

        processed_ocr_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        return rotated_img, processed_ocr_img

    def _bbox_rect(self, bbox):
        """Axis-aligned bounds from a PaddleOCR quadrilateral (point order agnostic)."""
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

    def _extract_address_by_geometry(self, lines, bank_bbox, ifsc_bbox, width_est, pad_ratio=0.15, x_limit_ratio=0.62):
        """Geometric address detector: catches lines that fail is_address()'s keyword
    checks (e.g. OCR misreads 'ROAD' as 'RAOD') as long as they sit in the band
    between the bottom of the bank-name block and the top of the IFSC line, in
    the left header column. Falls back to None if either anchor is missing,
    letting the existing keyword-based scan run instead."""
        if not bank_bbox or not ifsc_bbox:
            return None

    _, bank_top, _, bank_bottom = self._bbox_rect(bank_bbox)
    ifsc_left, ifsc_top, _, _ = self._bbox_rect(ifsc_bbox)
    if ifsc_top <= bank_bottom:
        return None  # anchors out of order / overlapping, don't trust the band

    band_height = max(ifsc_top - bank_bottom, 1)
    pad = band_height * pad_ratio
    y_start = bank_bottom - pad
    y_end = ifsc_top + pad
    x_limit = width_est * x_limit_ratio  # stay left, skip the "valid for 3 months" date grid

    candidates = []
    for line in lines:
        txt = line["text"].strip()
        if not txt:
            continue
        if self._is_overlay_mark(txt) or self._is_annotation_label(txt):
            continue
        if any(kw in txt.lower() for kw in self.non_field_keywords):
            continue
        if self.try_match_ifsc(txt):
            continue
        cy = self._row_center(line)
        cx = self._bbox_left(line["bbox"])
        if not (y_start <= cy <= y_end) or cx > x_limit:
            continue
        candidates.append(line)

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

        # Anchor on the tallest candidate. Printed bank names are almost always
        # the largest text near the top of a cheque; small-font mascot captions
        # or taglines (e.g. a Hindi tagline that an English-only OCR model
        # garbles into fake Latin letters, or even a genuine English tagline
        # like "Faithful") sit at a fraction of that height and must not get
        # glued onto the bank name just because they happen to be nearby — the
        # old logic accepted ANY short alphabetic word as a "continuation",
        # which is how junk like "Paearifta Faithful" ended up prefixed onto
        # "SyndicateBank".
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

    def run_ocr(self, img):
        predictions = list(self.ocr.predict(img))
        extracted = []
        for res in predictions:
            if 'rec_texts' in res:
                for text, bbox_arr, score in zip(res['rec_texts'], res['rec_polys'], res['rec_scores']):
                    bbox = bbox_arr.tolist()
                    clean_text = text.strip()
                    if clean_text:
                        extracted.append({
                            "text": clean_text,
                            "bbox": bbox,
                            "confidence": score
                        })
        return extracted

    def _overlay_exclusion_zones(self, extracted):
        """Bounding boxes of handwritten overlay marks (CANCELLED, VOID, SPECIMEN,
        etc). These must never contribute to ANY field — not just the field they
        happen to resemble — because OCR can misread cursive strokes as spurious
        digits or letters that coincidentally satisfy a totally unrelated field's
        pattern (e.g. a garbled digit run matching the account-number regex)."""
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

    def classify_and_extract(self, extracted, img_height):
        # 1. Calculate image bounding bounds early so x_ratio calculations don't fail
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
            cand = self.try_match_ifsc(source.get("text_tight", source.get("text", ""))) or self.try_match_ifsc(source["text"])
            if not cand:
                return
            bbox = self._ifsc_token_bbox(cand, source)
            consider("ifsc_code", cand, bbox, confidence)

        # --- 1. IFSC Extraction
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

        # --- 2. Account Number Extraction
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

        # --- 3. Position-Agnostic Bank Name Extraction
        # --- 3. Strict Position-Agnostic Bank Name Extraction
        bank_candidates = []
        # Must explicitly contain one of these to be recognized as a bank
        valid_bank_keywords = ['bank', 'sbi', 'hdfc', 'icici', 'axis', 'kotak', 'punjab', 'syndicate', 'baroda', 'canara', 'maharashtra', 'union', 'indian']
        
        for line in lines:
            txt = line["text"].strip()
            txt_l = txt.lower()

            if self._is_overlay_mark(txt) or self._is_annotation_label(txt):
                continue
            if any(kw in txt_l for kw in self.non_field_keywords):
                continue
            # Explicitly block company/account holder names
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
                    
                    # STRICT CHECK: Must contain a valid bank keyword
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

        # --- 4. Smart Bank Address Extraction & Cleanup
        ifsc_text = best.get("ifsc_code", {}).get("text", "")
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
            if self.is_address(txt) or self.branch_code_re.search(txt):
                address_candidates.append(line)

        if address_candidates:
            # Sort to explicitly penalize purely Tel/Fax lines and prioritize physical locations
            address_candidates.sort(
                key=lambda b: (
                    re.search(r'\b\d{6}\b', b["text"]) is not None,
                    any(kw in b["text"].lower() for kw in ['dist', 'nagar', 'road', 'vpo', 'teh', 'marg', 'street']),
                    -1 if ("tel" in b["text"].lower() or "fax" in b["text"].lower()) else 0,
                    len(b["text"])
                ),
                reverse=True,
            )
            
            best_addr = address_candidates[0]
            addr_text = best_addr["text"]
            
            # Clean up OCR hallucinations ("ta ia HH") and trailing IFSC/SWIFT gibberish
            addr_text = re.sub(r'^([a-zA-Z]{1,2}\s+){1,4}', '', addr_text) # Strips leading short gibberish
            addr_text = re.sub(r'IFS?C?\s*CODE\s*[:\-]?\s*[A-Z0-9]+', '', addr_text, flags=re.IGNORECASE)
            addr_text = re.sub(r'SWIFT\s*:?\s*[A-Z0-9]*', '', addr_text, flags=re.IGNORECASE)
            addr_text = addr_text.strip()
            
            consider("bank_address", addr_text, best_addr["bbox"], best_addr["confidence"])
        # --- 5. Strip Devanagari (Hindi/vernacular) tokens from text fields
        for field_type in ("bank_name", "bank_address"):
            if field_type in best:
                original = best[field_type]["text"]
                cleaned = self._strip_devanagari(original)
                best[field_type]["text"] = cleaned if cleaned else original

        return best

    def crop_field(self, img, bbox, pad=2, expand_ratio=0.12, min_expand_px=4,
                   tight=False, tight_thresh=200, tight_pad=2):
        """Axis-aligned crop from the detector bbox so x/y always match the ROI."""
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
        """Trim blank margins off a crop so it starts right at the first ink
        column/row instead of the raw (padded) detector bbox."""
        if crop is None or crop.size == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, bin_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

        col_sums = bin_img.sum(axis=0)
        row_sums = bin_img.sum(axis=1)
        cols = np.where(col_sums > 0)[0]
        rows = np.where(row_sums > 0)[0]
        if len(cols) == 0 or len(rows) == 0:
            return crop  # nothing detected as ink, leave crop untouched

        h, w = gray.shape[:2]
        x0 = max(int(cols[0]) - pad, 0)
        x1 = min(int(cols[-1]) + pad + 1, w)
        y0 = max(int(rows[0]) - pad, 0)
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
