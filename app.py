import base64
import json
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from cheque_ocr import ChequeOCR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

ocr_engine = ChequeOCR()

TARGET_FIELDS = ["bank_name", "bank_address", "account_no", "ifsc_code"]

FIELD_LABELS = {
    "bank_name": "Bank Name",
    "bank_address": "Branch Address",
    "account_no": "Account Number",
    "ifsc_code": "IFSC Code",
    "payer_name": "Payer Name",
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def encode_crop(crop):
    if crop is None or crop.size == 0:
        return None
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def run_pipeline_stream(img, debug=False):
    """Runs the real pipeline stages and yields SSE progress after each one
    actually finishes on the backend. Field results are revealed one at a
    time - each field gets an 'active' event right before it's looked up
    (drives a "Searching [Field]..." state in the UI) and a 'done'/'missing'
    event right after. classify_and_extract itself has no separate stage
    event: it runs for all fields together and is fast, so it's folded
    silently between the OCR stage finishing and the first field's
    'active' event.

    debug: when False (the default), the raw per-box OCR detections and
    merged-line debug view are left out of the response entirely - for a
    busy cheque that's easily hundreds of boxes with full bbox arrays, and
    building/serializing that on every request costs real time for a view
    nobody's looking at outside active debugging. Pass ?debug=1 to get it
    back."""
    pipeline_start = time.perf_counter()

    yield sse({"stage": "preprocess", "status": "active"})
    rotated_img, proc_img, detected_angle, perspective_applied = ocr_engine.preprocess_image(img)
    img_height = rotated_img.shape[0]
    yield sse({"stage": "preprocess", "status": "done"})

    yield sse({"stage": "ocr", "status": "active"})
    ocr_start = time.perf_counter()

    # For very large images or difficult cheques, OCR can take 1-3 minutes.
    # Yield a substage event during the downscale+detect phase so the UI
    # shows "Detecting text..." instead of just spinning silently for 60+
    # seconds before recognition starts. This is purely UX feedback -
    # no actual progress tracking (PaddleOCR doesn't expose per-box progress).
    yield sse({"ocr_substage": "downscale_detect", "status": "active"})

    extracted = ocr_engine.run_ocr(proc_img)

    yield sse({"ocr_substage": "downscale_detect", "status": "done"})
    ocr_seconds = time.perf_counter() - ocr_start
    yield sse({"stage": "ocr", "status": "done"})

    key_fields, cancelled = ocr_engine.classify_and_extract(extracted, img_height)
    yield sse({"cancelled": cancelled})

    # Workflow: a cancelled cheque has no payee to read, so payer_name is
    # dropped from the run entirely rather than reported "not found".
    active_fields = TARGET_FIELDS + ([] if cancelled else ["payer_name"])

    fields = {}
    for field in active_fields:
        yield sse({"field": field, "status": "active"})

        if field not in key_fields:
            yield sse({"field": field, "status": "missing"})
            continue

        data = key_fields[field]
        crop = ocr_engine.crop_field(rotated_img, data["bbox"])
        fields[field] = {
            "label": FIELD_LABELS.get(field, field),
            "text": data["text"],
            "low_confidence": float(data["confidence"]) < 0.8,
            "crop": encode_crop(crop),
        }
        yield sse({"field": field, "status": "done"})

    missing = [FIELD_LABELS.get(f, f) for f in active_fields if f not in key_fields]

    result = {
        "fields": fields,
        "missing": missing,
        "cancelled": cancelled,
        "message": "This cheque is CANCELLED." if cancelled else None,
        "timing": {
            # Time spent inside the OCR model itself (detection + recognition).
            "ocr_seconds": round(ocr_seconds, 2),
            # Full pipeline: preprocess + OCR + classify + crop/encode all fields.
            "total_seconds": round(time.perf_counter() - pipeline_start, 2),
        },
    }
    if debug:
        lines_debug = ocr_engine.merge_same_line(extracted)
        result["debug"] = extracted
        result["lines_debug"] = [
            {"text": l["text"], "bbox": l["bbox"], "confidence": l["confidence"]}
            for l in lines_debug
        ]

    yield sse({"status": "complete", "result": to_jsonable(result)})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract-stream", methods=["POST"])
def extract_stream():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read that image."}), 400

    debug_mode = request.args.get("debug") == "1"

    return Response(
        stream_with_context(run_pipeline_stream(img, debug=debug_mode)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
