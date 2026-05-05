// ======================================================
// GLOBAL STATE
// ======================================================
let currentGraphType = null;

// ======================================================
// GLOBAL ELEMENTS
// ======================================================
const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const browseBtn = document.getElementById("browseBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");

const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");

const resultCard = document.getElementById("resultCard");
const statusBadge = document.getElementById("statusBadge");
const graphType = document.getElementById("graphType");
const confidence = document.getElementById("confidence");
const analysisTime = document.getElementById("analysisTime");
const trendText = document.getElementById("trendText");
const rulesList = document.getElementById("rulesList");
const featuresList = document.getElementById("featuresList");

const unscaledOptions = document.getElementById("unscaledOptions");

// Unscaled inputs
const yAxisSelect = document.getElementById("yAxis");
const xAxisSelect = document.getElementById("xAxis");
const behaviorSelect = document.getElementById("behavior");
const sensitivitySelect = document.getElementById("sensitivity");

const yMinInput = document.getElementById("yMin");
const yMaxInput = document.getElementById("yMax");
const yUnitSelect = document.getElementById("yUnit");
const yUnitCustom = document.getElementById("yUnitCustom");

// ======================================================
// BASIC UI HELPERS
// ======================================================
function resetResultsUI() {
    resultCard.classList.remove("show");
    rulesList.innerHTML = "<li>No rules evaluated.</li>";
    featuresList.innerHTML = "<li>-</li>";
    trendText.textContent = "-";
    statusBadge.textContent = "UNKNOWN";
    statusBadge.className = "status-badge";
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " bytes";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

// ======================================================
// FILE SELECTION
// ======================================================
browseBtn.addEventListener("click", e => {
    e.stopPropagation();
    imageInput.click();
});

uploadArea.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", async () => {
    if (!imageInput.files.length) return;

    const file = imageInput.files[0];
    resetResultsUI();

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    selectedFile.classList.add("show");

    analyzeBtn.disabled = true;
    unscaledOptions.style.display = "none";

    // Detect graph type FIRST
    const fd = new FormData();
    fd.append("image", file);

    const res = await fetch("/detect-graph-type", {
        method: "POST",
        body: fd
    });

    const data = await res.json();
    currentGraphType = data.graph_type;

    if (currentGraphType === "unscaled") {
    unscaledOptions.style.display = "block";
    } else {
        unscaledOptions.style.display = "none";
    }

    analyzeBtn.disabled = false;
});

// ======================================================
// ANALYZE BUTTON
// ======================================================
analyzeBtn.addEventListener("click", async () => {
    if (!imageInput.files.length) return;

    loading.classList.add("show");
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    if (currentGraphType === "unscaled") {
        // Optional inputs — send only if present
        if (yAxisSelect.value) formData.append("y_axis_meaning", yAxisSelect.value);
        if (xAxisSelect.value) formData.append("x_axis_meaning", xAxisSelect.value);
        if (behaviorSelect.value) formData.append("expected_behavior", behaviorSelect.value);
        if (sensitivitySelect.value) formData.append("sensitivity", sensitivitySelect.value);

        if (yMinInput.value !== "") formData.append("y_min", yMinInput.value);
        if (yMaxInput.value !== "") formData.append("y_max", yMaxInput.value);

        if (yUnitSelect.value === "custom" && yUnitCustom.value) {
            formData.append("y_unit", yUnitCustom.value);
        } else if (yUnitSelect.value) {
            formData.append("y_unit", yUnitSelect.value);
        }
    }

    const res = await fetch("/analyze", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    loading.classList.remove("show");
    analyzeBtn.disabled = false;

    if (data.error) {
        showError(data.error);
        return;
    }

    displayResults(data);
});

// ======================================================
// DISPLAY RESULTS
// ======================================================
function displayResults(data) {

    // ===============================
    // ✅ STATUS (from ML prediction)
    // ===============================
    const status = data.prediction === 1 ? "ABNORMAL" : "NORMAL";

    statusBadge.textContent = status;
    statusBadge.className = "status-badge";

    if (status === "NORMAL") {
        statusBadge.classList.add("status-success");
    } else {
        statusBadge.classList.add("status-danger");
    }

    // ===============================
    // ✅ GRAPH TYPE
    // ===============================
    graphType.textContent = currentGraphType || "Unknown";

    // ===============================
    // ✅ CONFIDENCE
    // ===============================
    confidence.textContent = (data.confidence * 100).toFixed(2) + "%";

    // ===============================
    // ✅ TIME
    // ===============================
    analysisTime.textContent = "Instant";

    // ===============================
    // ❌ REMOVE OLD TREND
    // ===============================
    trendText.innerHTML = "ML-based prediction (no rule engine)";

    // ===============================
    // ❌ REMOVE RULES
    // ===============================
    rulesList.innerHTML = `
        <li>Prediction based on trained ML model</li>
    `;

    // ===============================
    // ✅ FEATURES
    // ===============================
    featuresList.innerHTML = Object.entries(data.features).map(
        ([k, v]) => `
        <li>
            <span class="feature-name">${k.replace(/_/g, " ")}</span>
            <span class="feature-value">${Number(v).toFixed(4)}</span>
        </li>
    `
    ).join("");

    // ===============================
    // ✅ EXPLANATION (SMART)
    // ===============================

    const severityBox = document.getElementById("severityBox");
    const summaryBox = document.getElementById("summaryBox");
    const whyList = document.getElementById("whyList");
    const fixList = document.getElementById("fixList");

    severityBox.className = "explain-box severity";

    if (status === "ABNORMAL") {
        severityBox.classList.add("severity-high");
        severityBox.innerHTML = "<b>Severity:</b> High – Abnormal signal detected.";

        summaryBox.innerHTML = `<b>Summary:</b> The model detected abnormal behavior in the signal.`;

        whyList.innerHTML = `
            <li>Feature distribution deviates from normal patterns</li>
            <li>Signal variability or spikes detected</li>
        `;

        fixList.innerHTML = `
            <li>Check for noise or irregular spikes</li>
            <li>Inspect system for instability</li>
        `;
    } else {
        severityBox.classList.add("severity-low");
        severityBox.innerHTML = "<b>Severity:</b> Low – Signal is normal.";

        summaryBox.innerHTML = `<b>Summary:</b> The signal is stable and within expected range.`;

        whyList.innerHTML = `<li>No abnormal patterns detected</li>`;
        fixList.innerHTML = `<li>No action required</li>`;
    }

    resultCard.classList.add("show");
    resultCard.scrollIntoView({ behavior: "smooth" });
}

// ======================================================
// ERROR DISPLAY
// ======================================================
function showError(msg) {
    resultCard.innerHTML = `
        <div class="card-title">Error</div>
        <p style="color:red">${msg}</p>
    `;
    resultCard.classList.add("show");
}
