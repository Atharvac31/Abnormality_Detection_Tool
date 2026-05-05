const fileInput = document.getElementById("pipelineFile");
const fileNameDisplay = document.getElementById("fileNameDisplay");

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        fileNameDisplay.textContent = fileInput.files[0].name;
    }
});

// =============================
// HELPER
// =============================
function step(title, content) {
    return `
        <div class="step-card">
            <div class="step-title">${title}</div>
            ${content}
        </div>
        <div class="step-arrow">⬇</div>
    `;
}

// =============================
// BASIC PIPELINE
// =============================
async function runPipeline() {
    const container = document.getElementById("pipelineContainer");

    if (!fileInput.files.length) return;

    container.innerHTML = "⏳ Running pipeline...";

    const fd = new FormData();
    fd.append("image", fileInput.files[0]);

    const res = await fetch("/pipeline-analyze", {
        method: "POST",
        body: fd
    });

    const data = await res.json();

    container.innerHTML = "";

    container.innerHTML += step("📊 Graph Type", `<p>${data.graph_type}</p>`);
    container.innerHTML += step("📈 Raw Signal", `<pre>${JSON.stringify(data.raw_signal.slice(0,50), null, 2)}</pre>`);
    container.innerHTML += step("⚙️ Normalized Signal", `<pre>${JSON.stringify(data.normalized_signal.slice(0,50), null, 2)}</pre>`);
    container.innerHTML += step("🧼 Smoothed Signal", `<pre>${JSON.stringify(data.smoothed_signal.slice(0,50), null, 2)}</pre>`);
    container.innerHTML += step("🧠 Features", `<pre>${JSON.stringify(data.features, null, 2)}</pre>`);
    container.innerHTML += step("📥 Model Input", `<pre>${JSON.stringify(data.model_input, null, 2)}</pre>`);

    container.innerHTML += `
        <div class="step-card">
            <div class="step-title">🎯 Prediction</div>
            <p><b>${data.prediction === 1 ? "ABNORMAL 🚨" : "NORMAL ✅"}</b></p>
            <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
        </div>
    `;
}

// =============================
// JOINT PIPELINE (🔥 FINAL)
// =============================
async function runJointPipeline() {

    const container = document.getElementById("pipelineContainer");

    if (!fileInput.files.length) return;

    container.innerHTML = "⏳ Running Joint Pipeline...";

    const fd = new FormData();
    fd.append("image", fileInput.files[0]);

    const res = await fetch("/pipeline-joint", {
        method: "POST",
        body: fd
    });

    const data = await res.json();

    if (data.error) {
        container.innerHTML = `<div style="color:red;">${data.error}</div>`;
        return;
    }

    const steps = data.steps;
    container.innerHTML = "";

    // =============================
    // FEATURES
    // =============================
    container.innerHTML += step(
        "🧠 Extracted Features",
        `<pre>${JSON.stringify(steps.feature_extraction, null, 2)}</pre>`
    );

    // =============================
    // WEAK LABELING
    // =============================
    const lf = steps.weak_labeling.lf_outputs;
    const labels = steps.weak_labeling.weak_labels;

    let lfTable = `
        <table class="lf-table">
            <tr>
                <th>Rule</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
    `;

    Object.keys(lf).forEach(key => {
        const val = lf[key];
        const status = labels[key];

        const badge = status === 1
            ? `<span class="badge red">ABNORMAL</span>`
            : `<span class="badge green">NORMAL</span>`;

        lfTable += `
            <tr>
                <td>${key}</td>
                <td>${val.toFixed(3)}</td>
                <td>${badge}</td>
            </tr>
        `;
    });

    lfTable += `</table>
        <p><b>Score:</b> ${steps.weak_labeling.score.toFixed(2)}</p>
    `;

    container.innerHTML += step("🏷 Weak Labeling", lfTable);

    // =============================
    // BASELINE MODEL (🔥 EXPLAINABLE)
    // =============================
    const bm = steps.baseline_model;

    container.innerHTML += step(
        "📊 Baseline Model",
        `
        <p><b>${bm.prediction === 1 ? "ABNORMAL 🚨" : "NORMAL ✅"}</b></p>
        <p>Confidence: ${(bm.confidence * 100).toFixed(2)}%</p>

        <hr/>

        ${bm.top_features ? `
        <p><b>Top Influencing Features:</b></p>
        <ul>
            ${bm.top_features.map(f => `
                <li>${f.name} (${f.value.toFixed(2)})</li>
            `).join("")}
        </ul>
        ` : ""}

        ${bm.explanation ? `<p><b>Explanation:</b> ${bm.explanation}</p>` : ""}
        `
    );

    // =============================
    // SUBSET SELECTION (🔥 EXPLAINED)
    // =============================
    const ss = steps.subset_selection;

    container.innerHTML += step(
        "🎯 Subset Selection (SPEAR)",
        `
        <p><b>Selected:</b> ${ss.selected_samples}</p>
        ${ss.total_samples ? `<p><b>Total:</b> ${ss.total_samples}</p>` : ""}

        <hr/>

        <p><b>Method:</b> ${ss.method}</p>

        ${ss.strategy ? `<p>${ss.strategy}</p>` : ""}
        ${ss.benefit ? `<p><b>Why:</b> ${ss.benefit}</p>` : ""}
        `
    );

    // =============================
    // JOINT LEARNING (🔥 ADVANCED)
    // =============================
    const jl = steps.joint_learning;

    let color = "gray";
    if (jl.final_prediction.includes("ABNORMAL")) color = "red";
    else if (jl.final_prediction === "NORMAL") color = "green";

    const progress = (jl.joint_score * 100).toFixed(1);

    let reasoning = "Balanced contribution";
    if (jl.weak_score && jl.model_score) {
        reasoning = jl.weak_score > jl.model_score
            ? "Weak rules influenced decision more"
            : "Model influenced decision more";
    }

    container.innerHTML += `
    <div class="step-card">
        <div class="step-title">🔗 Joint Learning (Detailed)</div>

        <p><b>Final:</b> <span class="badge ${color}">${jl.final_prediction}</span></p>
        <p><b>Score:</b> ${jl.joint_score.toFixed(3)}</p>

        <div class="progress-bar">
            <div class="progress-fill" style="width:${progress}%"></div>
        </div>

        <div class="range-labels">
            <span>Normal</span>
            <span>Uncertain</span>
            <span>Abnormal</span>
        </div>

        <hr/>

        ${jl.weak_contribution ? `
        <p><b>Contribution:</b></p>
        <p>🧠 Weak: ${(jl.weak_contribution * 100)}%</p>
        <p>🤖 Model: ${(jl.model_contribution * 100)}%</p>
        ` : ""}

        ${jl.weak_score ? `
        <p><b>Scores:</b></p>
        <p>Weak: ${jl.weak_score.toFixed(3)}</p>
        <p>Model: ${jl.model_score.toFixed(3)}</p>
        ` : ""}

        <p><b>Reasoning:</b> ${reasoning}</p>
    </div>
    `;
}