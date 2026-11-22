/* Churn Bureau, logika intake.
   Satu POST /predict per berkas, riwayat disimpan di memori sesi saja. */

(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const form = $("#dossier");
  const submit = $("#submit");
  const submitLabel = $("#submit-label");
  const screen = $("#screen");
  const probEl = $("#prob");
  const probSub = $("#prob-sub");
  const stamp = $("#stamp");
  const verdict = $("#verdict");
  const verdictCopy = $("#verdict-copy");
  const errorBox = $("#error");
  const offlineBox = $("#offline");
  const status = $("#status");
  const statusText = $("#status-text");
  const meter = $("#meter");
  const ledgerBody = $("#ledger-body");
  const ledgerEmpty = $("#ledger-empty");
  const batchCsv = $("#batch-csv");
  const batchBody = $("#batch-body");
  const batchEmpty = $("#batch-empty");
  const batchSummary = $("#batch-summary");
  const batchErrorBox = $("#batch-error");
  const batchSubmit = $("#batch-submit");
  const batchFileName = $("#batch-file-name");

  const fmt3 = new Intl.NumberFormat("id-ID", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  const fmt2 = new Intl.NumberFormat("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtPct = new Intl.NumberFormat("id-ID", { style: "percent", maximumFractionDigits: 1 });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const LABELS = {
    "Month-to-month": "Bulanan",
    "One year": "1 tahun",
    "Two year": "2 tahun",
    "Fiber optic": "Fiber",
    DSL: "DSL",
    No: "Tanpa internet",
  };

  const PRESETS = {
    fresh: {
      customerID: "DEMO-FRESH",
      gender: "Female",
      SeniorCitizen: "0",
      Partner: "Yes",
      Dependents: "No",
      tenure: "2",
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: "70.70",
      TotalCharges: "151.65",
    },
    loyal: {
      customerID: "DEMO-LOYAL",
      gender: "Male",
      SeniorCitizen: "0",
      Partner: "Yes",
      Dependents: "Yes",
      tenure: "60",
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "Yes",
      DeviceProtection: "Yes",
      TechSupport: "Yes",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "Two year",
      PaperlessBilling: "No",
      PaymentMethod: "Bank transfer (automatic)",
      MonthlyCharges: "60.00",
      TotalCharges: "3600.00",
    },
    drift: {
      customerID: "DEMO-DSL",
      gender: "Female",
      SeniorCitizen: "0",
      Partner: "No",
      Dependents: "No",
      tenure: "18",
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "DSL",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: "55.00",
      TotalCharges: "990.00",
    },
  };

  let threshold = 0.17;
  let caseSeq = 0;
  let busy = false;
  let hasResult = false;
  let modelLabel = "model";
  let offline = false;
  const ledger = [];

  const cells = Array.from({ length: 24 }, (_, index) => {
    const cell = document.createElement("span");
    cell.className = "meter__cell";
    cell.style.transitionDelay = `${index * 14}ms`;
    meter.append(cell);
    return cell;
  });

  function setMeter(probability) {
    const filled = Math.round(probability * cells.length);
    cells.forEach((cell, index) => {
      const on = index < filled;
      cell.classList.toggle("is-on", on);
      cell.classList.toggle("is-over", on && (index + 0.5) / cells.length >= threshold);
    });
  }

  function setStatus(state, text) {
    status.dataset.state = state;
    statusText.textContent = text;
  }

  function markStale() {
    if (!hasResult) return;
    screen.dataset.stale = "true";
    $("#stale").hidden = false;
  }

  function setBusy(value) {
    busy = value;
    submit.disabled = value || offline;
    batchSubmit.disabled = value || offline;
    form.setAttribute("aria-busy", String(value));
    submitLabel.textContent = value ? "Memindai…" : "Pindai berkas";
  }

  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function showError(message) {
    errorBox.hidden = false;
    errorBox.textContent = message;
    screen.scrollIntoView({
      behavior: reducedMotion.matches ? "auto" : "smooth",
      block: "nearest",
    });
  }

  function goOffline() {
    offline = true;
    offlineBox.hidden = false;
    setStatus("down", "Terputus");
    submit.disabled = true;
    batchSubmit.disabled = true;
  }

  function goOnline() {
    offline = false;
    offlineBox.hidden = true;
    setStatus("ok", "Siap");
    submit.disabled = false;
    batchSubmit.disabled = false;
  }

  function readPayload() {
    const payload = {};
    for (const element of form.elements) {
      if (!element.name) continue;
      if (element.name === "customerID") {
        const value = element.value.trim();
        if (value) payload.customerID = value;
        continue;
      }
      const numeric = element.type === "number" || element.name === "SeniorCitizen";
      payload[element.name] = numeric ? Number(element.value) : element.value;
    }
    return payload;
  }

  function fill(values) {
    for (const [name, value] of Object.entries(values)) {
      const element = form.elements.namedItem(name);
      if (element) element.value = value;
    }
    hideError();
    markStale();
  }

  function describeError(body, httpStatus) {
    if (body && Array.isArray(body.detail)) {
      return body.detail
        .map((item) => {
          const field = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "input";
          return `${field}: ${item.msg}`;
        })
        .join("\n");
    }
    if (body && typeof body.detail === "string") return body.detail;
    return `Permintaan ditolak (HTTP ${httpStatus}).`;
  }

  function renderChips(payload) {
    const chips = [
      ["tenure", `${payload.tenure} bln`],
      ["kontrak", LABELS[payload.Contract] || payload.Contract],
      ["internet", LABELS[payload.InternetService] || payload.InternetService],
      ["bulanan", `$${fmt2.format(payload.MonthlyCharges)}`],
    ];
    const container = $("#chips");
    container.replaceChildren(
      ...chips.map(([label, value]) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        const key = document.createElement("b");
        key.textContent = label;
        chip.append(key, document.createTextNode(value));
        return chip;
      })
    );
  }

  function addLedgerRow(payload, result) {
    const row = document.createElement("tr");
    const time = document.createElement("td");
    time.textContent = new Date().toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" });

    const subject = document.createElement("td");
    subject.textContent = payload.customerID || "Tanpa ID";

    const probability = document.createElement("td");
    probability.textContent = fmt3.format(result.churn_probability);

    const decision = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.dataset.verdict = result.churn_prediction ? "high" : "low";
    tag.textContent = result.churn_prediction ? "Prioritas" : "Rendah";
    decision.append(tag);

    row.append(time, subject, probability, decision);
    ledgerBody.prepend(row);

    ledger.push(row);
    while (ledger.length > 8) ledger.shift().remove();
    ledgerEmpty.hidden = true;
  }

  function render(payload, result) {
    threshold = result.threshold;
    screen.style.setProperty("--threshold", String(threshold));
    screen.dataset.verdict = result.churn_prediction ? "high" : "low";
    screen.dataset.stale = "false";
    $("#stale").hidden = true;
    hasResult = true;

    caseSeq += 1;
    $("#case-no").textContent = `CB-${String(caseSeq).padStart(4, "0")}`;

    probEl.textContent = fmt3.format(result.churn_probability);
    probSub.textContent = `${fmtPct.format(result.churn_probability)} peluang churn · ${
      result.churn_prediction ? "di atas" : "di bawah"
    } ambang`;

    stamp.textContent = result.churn_prediction ? "Prioritas retensi" : "Risiko rendah";
    verdictCopy.textContent = result.churn_prediction
      ? "Berkas ini melewati ambang biaya, jadi masuk daftar telepon retensi lebih dulu. Konsekuensinya, sebagian pelanggan loyal ikut ditawari diskon."
      : "Berkas ini di bawah ambang biaya, jadi tidak masuk daftar retensi. Peluang churn-nya tetap ada, cuma belum cukup mahal untuk ditelepon sekarang.";
    verdict.hidden = false;

    $("#meta-threshold").textContent = fmt2.format(threshold);
    $("#meta-decision").textContent = result.churn_prediction ? "Prioritas" : "Rendah";
    $("#meta-model").textContent = result.model_version;
    $("#meta-time").textContent = new Date().toLocaleTimeString("id-ID", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    setMeter(result.churn_probability);
    renderChips(payload);
    addLedgerRow(payload, result);
    hideError();

    if (window.matchMedia("(max-width: 1080px)").matches) {
      screen.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
    }
  }

  async function predict(payload) {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const error = new Error(describeError(body, response.status));
      error.fromApi = true;
      throw error;
    }
    return body;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }

    const payload = readPayload();
    setBusy(true);
    hideError();
    try {
      render(payload, await predict(payload));
    } catch (error) {
      if (error.fromApi) {
        showError(error.message);
      } else {
        showError("Tidak bisa menghubungi API. Pastikan server berjalan (`.\\run.ps1 serve`), lalu coba lagi.");
        goOffline();
      }
    } finally {
      setBusy(false);
    }
  });

  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => fill(PRESETS[button.dataset.preset]));
  });

  $("#estimate-total").addEventListener("click", () => {
    const tenure = Number(form.elements.namedItem("tenure").value);
    const monthly = Number(form.elements.namedItem("MonthlyCharges").value);
    if (Number.isFinite(tenure) && Number.isFinite(monthly)) {
      form.elements.namedItem("TotalCharges").value = (tenure * monthly).toFixed(2);
    }
  });

  $("#clear-ledger").addEventListener("click", () => {
    ledgerBody.replaceChildren();
    ledger.length = 0;
    ledgerEmpty.hidden = false;
  });

  for (const element of form.elements) {
    if (element.name && element.name !== "customerID") element.required = true;
  }

  form.addEventListener("input", markStale);
  form.addEventListener("change", markStale);

  // ---- Batch CSV ----
  const CSV_FIELDS = Array.from(form.elements)
    .filter((element) => element.name)
    .map((element) => element.name);
  const CSV_CANONICAL = new Map(CSV_FIELDS.map((name) => [name.toLowerCase(), name]));
  const NUMERIC_FIELDS = new Set(["tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges"]);
  const INTEGER_FIELDS = new Set(["tenure", "SeniorCitizen"]);
  const CSV_HEADER = CSV_FIELDS.join(",");

  const SAMPLE_CSV = [
    CSV_HEADER,
    "DEMO-FRESH,Female,0,Yes,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,151.65",
    "DEMO-LOYAL,Male,0,Yes,Yes,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Bank transfer (automatic),60.00,3600.00",
    "DEMO-DSL,Female,0,No,No,18,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,55.00,990.00",
  ].join("\n");

  const BROKEN_CSV = [
    CSV_HEADER,
    "DEMO-FRESH,Female,0,Yes,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,151.65",
    "DEMO-TYPO,Male,0,Yes,Yes,999,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Bank transfer (automatic),60.00,3600.00",
    "DEMO-DSL,Female,0,No,No,18,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,55.00,990.00",
  ].join("\n");

  function splitCsvLine(line) {
    const cells = [];
    let value = "";
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      if (quoted) {
        if (char === '"' && line[index + 1] === '"') {
          value += '"';
          index += 1;
        } else if (char === '"') {
          quoted = false;
        } else {
          value += char;
        }
      } else if (char === '"') {
        quoted = true;
      } else if (char === ",") {
        cells.push(value.trim());
        value = "";
      } else {
        value += char;
      }
    }
    cells.push(value.trim());
    return cells;
  }

  function parseCsv(text) {
    const lines = text.replace(/\r\n?/g, "\n").split("\n").filter((line) => line.trim() !== "");
    if (lines.length < 2) {
      throw new Error("CSV butuh satu baris header dan minimal satu baris data.");
    }
    const header = splitCsvLine(lines[0]).map((cell) => CSV_CANONICAL.get(cell.toLowerCase()) || null);
    const missing = CSV_FIELDS.filter((name) => name !== "customerID" && !header.includes(name));
    if (missing.length) {
      throw new Error(`Kolom wajib belum ada di header: ${missing.join(", ")}.`);
    }
    return lines.slice(1).map((line) => {
      const cells = splitCsvLine(line);
      const payload = {};
      header.forEach((name, index) => {
        if (!name) return;
        const raw = (cells[index] ?? "").trim();
        if (name === "customerID") {
          if (raw) payload.customerID = raw;
          return;
        }
        if (!NUMERIC_FIELDS.has(name)) {
          payload[name] = raw;
          return;
        }
        if (raw === "") {
          payload[name] = null;
          return;
        }
        payload[name] = INTEGER_FIELDS.has(name) ? Number.parseInt(raw, 10) : Number(raw);
      });
      return payload;
    });
  }

  function hideBatchError() {
    batchErrorBox.hidden = true;
    batchErrorBox.textContent = "";
  }

  function showBatchError(message) {
    batchErrorBox.hidden = false;
    batchErrorBox.textContent = message;
  }

  function describeBatchError(body) {
    if (!body || !Array.isArray(body.detail)) return describeError(body, 422);
    const shown = body.detail.slice(0, 8).map((item) => {
      const loc = Array.isArray(item.loc) ? item.loc.slice(1) : [];
      const rowIndex = loc[0] === "customers" && Number.isInteger(loc[1]) ? loc[1] : null;
      const field = rowIndex === null ? loc.slice(1).join(".") : loc.slice(2).join(".");
      const where = rowIndex === null ? "permintaan" : `baris ${rowIndex + 2}`;
      return `${where}${field ? ` · ${field}` : ""}: ${item.msg}`;
    });
    const rest = body.detail.length - shown.length;
    const tail = rest > 0 ? `\n… dan ${rest} masalah lain.` : "";
    return `Ditolak, tidak ada baris yang diprediksi.\n${shown.join("\n")}${tail}`;
  }

  function renderBatch(body) {
    const predictions = Array.isArray(body.predictions) ? body.predictions : [];
    const priorities = predictions.filter((item) => item.churn_prediction).length;
    batchBody.replaceChildren(
      ...predictions.map((item, index) => {
        const row = document.createElement("tr");
        const number = document.createElement("td");
        number.textContent = String(index + 2);
        const subject = document.createElement("td");
        subject.textContent = item.customer_id || "Tanpa ID";
        const probability = document.createElement("td");
        probability.textContent = fmt3.format(item.churn_probability);
        const decision = document.createElement("td");
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.dataset.verdict = item.churn_prediction ? "high" : "low";
        tag.textContent = item.churn_prediction ? "Prioritas" : "Rendah";
        decision.append(tag);
        row.append(number, subject, probability, decision);
        return row;
      })
    );
    batchEmpty.hidden = predictions.length > 0;
    const usedThreshold = predictions[0] ? predictions[0].threshold : threshold;
    batchSummary.textContent = `${body.count} baris diprediksi · ${priorities} prioritas retensi · ambang ${fmt2.format(usedThreshold)}`;
    hideBatchError();
  }

  async function runBatch() {
    if (offline || batchSubmit.disabled) return;
    hideBatchError();

    let rows;
    try {
      rows = parseCsv(batchCsv.value);
    } catch (error) {
      showBatchError(error.message);
      return;
    }
    if (!rows.length) {
      showBatchError("Tidak ada baris data di CSV.");
      return;
    }
    if (rows.length > 500) {
      showBatchError(`CSV berisi ${rows.length} baris; API menerima maksimal 500.`);
      return;
    }

    batchSubmit.disabled = true;
    const label = batchSubmit.textContent;
    batchSubmit.textContent = "Memindai…";
    try {
      const response = await fetch("/predict/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customers: rows }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        if (response.status === 422) {
          batchBody.replaceChildren();
          batchEmpty.hidden = false;
          batchSummary.textContent = `0 dari ${rows.length} baris diprediksi`;
          showBatchError(describeBatchError(body));
        } else {
          showBatchError(describeError(body, response.status));
        }
        return;
      }
      renderBatch(body);
    } catch (error) {
      showBatchError("Tidak bisa menghubungi API. Pastikan server berjalan, lalu coba lagi.");
      goOffline();
    } finally {
      batchSubmit.textContent = label;
      batchSubmit.disabled = offline;
    }
  }

  function setBatchSource(text, label) {
    batchCsv.value = text;
    batchFileName.textContent = label;
    hideBatchError();
  }

  $("#batch-sample").addEventListener("click", () => setBatchSource(SAMPLE_CSV, "contoh 3 baris"));
  $("#batch-broken").addEventListener("click", () => setBatchSource(BROKEN_CSV, "contoh dengan baris rusak"));
  batchSubmit.addEventListener("click", runBatch);

  $("#batch-file").addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    const text = await file.text();
    const dataRows = text.split("\n").filter((line) => line.trim() !== "").length - 1;
    setBatchSource(text, `${file.name} · ${dataRows} baris`);
  });

  async function loadModel() {
    const [health, info] = await Promise.allSettled([
      fetch("/health").then((response) => {
        if (!response.ok) throw new Error("health gagal");
        return response.json();
      }),
      fetch("/model/info").then((response) => {
        if (!response.ok) throw new Error("info gagal");
        return response.json();
      }),
    ]);

    if (health.status === "rejected" || !health.value.model_loaded) {
      goOffline();
      return;
    }
    goOnline();

    if (info.status === "rejected") return;
    const data = info.value;
    threshold = data.threshold;
    screen.style.setProperty("--threshold", String(threshold));
    $("#threshold-label").textContent = `Ambang ${fmt2.format(threshold)} · garis putus-putus`;
    $("#meta-threshold").textContent = fmt2.format(threshold);

    modelLabel = String(data.model_name).replaceAll("_", " ");
    $("#model-line").textContent = `model ${modelLabel}`;
    $("#meta-model").textContent = data.version;

    const metrics = data.test_metrics || {};
    $("#metric-roc").textContent = metrics.roc_auc != null ? fmt3.format(metrics.roc_auc) : "…";
    $("#metric-pr").textContent = metrics.pr_auc != null ? fmt3.format(metrics.pr_auc) : "…";
    $("#metric-recall").textContent = metrics.recall != null ? fmtPct.format(metrics.recall) : "…";
    $("#metric-precision").textContent = metrics.precision != null ? fmtPct.format(metrics.precision) : "…";
    $("#metric-brier").textContent = metrics.brier != null ? fmt3.format(metrics.brier) : "…";
  }

  loadModel();
})();
