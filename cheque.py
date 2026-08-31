import re
import os
from paddleocr import PaddleOCR

from text_utils import TextUtilsMixin
from geometry_utils import GeometryUtilsMixin
from field_detection import FieldDetectionMixin
from line_merging import LineMergingMixin
from image_processing import ImageProcessingMixin


class ChequeOCR(
    TextUtilsMixin,
    GeometryUtilsMixin,
    FieldDetectionMixin,
    LineMergingMixin,
    ImageProcessingMixin,
):
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