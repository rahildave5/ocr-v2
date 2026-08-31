const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const dzIdle = document.getElementById("dz-idle");
const dzPreview = document.getElementById("dz-preview");
const previewImg = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const runBtn = document.getElementById("run-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const fieldRowsEl = document.getElementById("field-rows");
const missingBanner = document.getElementById("missing-banner");
const cancelledBanner = document.getElementById("cancelled-banner");
const debugWrap = document.getElementById("debug-wrap");
const debugJson = document.getElementById("debug-json");
const uploadTips = document.querySelector(".upload-tips");

let selectedFile = null;

// ---------- Progress checklist ----------

const STAGE_LABELS = {
  preprocess: "Image Preprocessing",
  ocr: "Running OCR... This could take 5-6 minutes depending upon image quality.",
};
const FIELD_LABELS = {
  bank_name: "Bank Name",
  account_no: "Account Number",
  bank_address: "Branch Address",
  ifsc_code: "IFSC code",
  payer_name: "Payer Name",
};

function statusIcon(state) {
  if (state === "done") return `<span class="pi pi-done">&#10003;</span>`;
  if (state === "missing") return `<span class="pi pi-missing">&#33;</span>`;
  if (state === "active") return `<span class="spinner"></span>`;
  return `<span class="pi pi-pending"></span>`;
}

// The pipeline can finish faster than the browser's next paint. Without
// this, a burst of SSE messages (ocr done, field active, field done, ...)
// all get processed in the same synchronous stretch and only the very
// last DOM update ever actually reaches the screen - every earlier
// "Searching X..." state gets silently skipped, not just rendered too
// fast to notice. Awaiting a short delay after each intermediate update
// forces a real paint in between, and doubles as a readable minimum
// display time for each step.
function pace(ms = 110) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderProgress(stageState, fieldState, ocrSubstage = null) {
  const stageHtml = Object.keys(STAGE_LABELS).map((key) => {
    const s = stageState[key] || "pending";
    let html = `<div class="progress-row">${statusIcon(s)} ${STAGE_LABELS[key]}</div>`;
    
    // If OCR is active and we have a substage update, render it as a nested item
    if (key === "ocr" && s === "active" && ocrSubstage) {
      const substageLabel = ocrSubstage === "downscale_detect"
        ? "Detecting text boxes..."
        : "Processing...";
      html += `<div class="progress-row progress-substage">${statusIcon("active")} ${substageLabel}</div>`;
    }
    
    return html;
  }).join("");

  const fieldHtml = Object.keys(fieldState).length
    ? Object.keys(FIELD_LABELS).map((key) => {
        const s = fieldState[key];
        if (!s) return "";
        let labelText = FIELD_LABELS[key];
        if (s === "active") labelText = `Searching ${FIELD_LABELS[key]}...`;
        else if (s === "done") labelText = `${FIELD_LABELS[key]} extracted`;
        else if (s === "missing") labelText = `${FIELD_LABELS[key]} not found`;
        return `<div class="progress-row">${statusIcon(s)} ${labelText}</div>`;
      }).join("")
    : "";

  statusEl.hidden = false;
  statusEl.classList.remove("is-error");
  statusEl.innerHTML = `<div id="progress" class="progress-list">${stageHtml}${fieldHtml}</div>`;
}

function setStatus(text, { error = false } = {}) {
  if (!text) {
    statusEl.hidden = true;
    statusEl.innerHTML = "";
    return;
  }
  statusEl.hidden = false;
  statusEl.classList.toggle("is-error", error);
  statusEl.innerHTML = `<span>${text}</span>`;
}

function renderTiming(timing) {
  if (!timing) return;
  const progress = document.getElementById("progress");
  if (!progress) return;
  const row = document.createElement("div");
  row.className = "progress-row progress-timing";
  row.innerHTML = `<span class="pi pi-done">&#10003;</span> Done in ${timing.total_seconds}s (model: ${timing.ocr_seconds}s)`;
  progress.appendChild(row);
}

function selectFile(file) {
  if (!file) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewName.textContent = file.name;
  dzIdle.hidden = true;
  dzPreview.hidden = false;
  runBtn.disabled = false;
  setStatus(null);
  resultsEl.hidden = true;
  debugWrap.hidden = true;
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => selectFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dz-drag");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dz-drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) selectFile(file);
});

const copyBtn = document.getElementById("copy-btn");
let fieldsState = {};

function fieldRow(key, data) {
  const row = document.createElement("div");
  row.className = "field-row";
  row.dataset.field = key;

  const body = document.createElement("div");
  body.className = "field-body";

  const label = document.createElement("div");
  label.className = "field-label";
  label.textContent = data.label;

  const isMultiline = key === "bank_address";
  const input = document.createElement(isMultiline ? "textarea" : "input");
  input.className = "field-value-input";
  if (!isMultiline) input.type = "text";
  if (isMultiline) input.rows = 2;
  input.value = data.text || "";
  input.dataset.original = data.text || "";
  input.autocomplete = "off";
  input.spellcheck = false;

  const conf = document.createElement("div");
  conf.className = "field-conf";
  const roiText = data.roi
    ? `ROI x=${data.roi.x} y=${data.roi.y} w=${data.roi.width} h=${data.roi.height} · `
    : "";
  conf.textContent = `${roiText}confidence ${data.confidence.toFixed(2)}`;

  body.append(label, input, conf);

  input.value = (data.text || "").toUpperCase();
  input.dataset.original = input.value;   

  input.addEventListener("input", () => {
    input.value = input.value.toUpperCase();
    fieldsState[key] = input.value;
  });
  fieldsState[key] = input.value;

  row.append(body);
  return row;
}

function renderResult(payload) {
  cancelledBanner.hidden = !payload.cancelled;
  fieldRowsEl.innerHTML = "";
  const fieldKeys = Object.keys(payload.fields);
  if (fieldKeys.length === 0) {
    setStatus("No target fields detected.", { error: true });
  } else {
    fieldKeys.forEach((key) => fieldRowsEl.appendChild(fieldRow(key, payload.fields[key])));
    resultsEl.hidden = false;
  }

  if (payload.missing && payload.missing.length) {
    missingBanner.hidden = false;
    missingBanner.textContent = `Could not find: ${payload.missing.join(", ")}`;
  } else {
    missingBanner.hidden = true;
  }

  if (payload.debug) {
    debugJson.textContent = JSON.stringify(payload.debug, null, 2);
    debugWrap.hidden = false;
  } else {
    debugWrap.hidden = true;
  }
}

async function runExtraction() {
  if (!selectedFile) return;
  runBtn.disabled = true;
  resultsEl.hidden = true;
  debugWrap.hidden = true;
  uploadTips.hidden = true;

  const stageState = {};
  const fieldState = {};
  let ocrSubstage = null;
  stageState.preprocess = "active";
  renderProgress(stageState, fieldState, ocrSubstage);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/extract-stream", { method: "POST", body: formData });
    const contentType = res.headers.get("content-type") || "";

    if (!res.ok || !contentType.includes("text/event-stream")) {
      const payload = await res.json().catch(() => ({}));
      setStatus(payload.error || "Extraction failed.", { error: true });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // last part may be incomplete

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const data = JSON.parse(line.slice(5).trim());

        if (data.ocr_substage) {
          ocrSubstage = data.ocr_substage;
          renderProgress(stageState, fieldState, ocrSubstage);
          await pace();
        } else if (data.stage) {
          stageState[data.stage] = data.status;
          renderProgress(stageState, fieldState, ocrSubstage);
          await pace();
        } else if (data.field) {
          fieldState[data.field] = data.status;
          renderProgress(stageState, fieldState, ocrSubstage);
          await pace();
        } else if (data.status === "complete") {
          renderResult(data.result);
          renderTiming(data.result.timing);
        }
      }
    }
  } catch (err) {
    setStatus("Request failed: " + err.message, { error: true });
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runExtraction);
copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(fieldsState, null, 2));
    const original = copyBtn.textContent;
    copyBtn.textContent = "Copied ✓";
    setTimeout(() => (copyBtn.textContent = original), 1200);
  } catch {
    copyBtn.textContent = "Copy failed";
  }
});