// PathSnap evaluator UI. Every number shown comes from the API response.
const $ = (id) => document.getElementById(id);
let API = "";
try { API = localStorage.getItem("pathsnap_api") || ""; } catch (e) { /* storage unavailable */ }
API = API || window.PATHSNAP_API_DEFAULT;
$("api").value = API;

function setStatus(msg, err = false) {
  $("status").textContent = msg;
  $("status").className = err ? "err" : "";
}

const pct = (v) => (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : (100 * v).toFixed(2) + " %";

async function call(path, opts = {}) {
  const r = await fetch(API.replace(/\/$/, "") + path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}

async function connect() {
  API = $("api").value.trim();
  try { localStorage.setItem("pathsnap_api", API); } catch (e) { /* ignore */ }
  $("trips").innerHTML = "";
  setStatus("Connecting to " + API + " ...");
  try {
    const h = await call("/health");
    $("health").textContent = `connected - model: ${h.classifier}`;
    const t = await call("/trips");
    if (!t.trips.length) { setStatus("API reachable but it lists no evaluation trips.", true); return; }
    t.trips.forEach((id, i) => {
      const b = document.createElement("button");
      b.textContent = `Trip ${i + 1}: ${id}`;
      b.onclick = () => runTrip(id, b);
      $("trips").appendChild(b);
    });
    setStatus(`${t.trips.length} evaluation trips loaded.`);
  } catch (e) {
    $("health").textContent = "";
    setStatus("Cannot reach API: " + e.message, true);
  }
}

async function runTrip(id, btn) {
  document.querySelectorAll(".trips button").forEach((b) => { b.classList.remove("active"); b.disabled = true; });
  btn.classList.add("active");
  setStatus(`Trip ${id}: request sent, waiting for prediction ...`);
  $("results").textContent = "";
  const t0 = performance.now();
  try {
    const r = await call("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: id }),
    });
    const rt = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Trip ${id}: done in ${rt} s (server ${r.timing_s.request_total_s} s).`);
    render(r);
  } catch (e) {
    setStatus(`Trip ${id}: ${e.message}`, true);
  } finally {
    document.querySelectorAll(".trips button").forEach((b) => (b.disabled = false));
  }
}

function render(r) {
  const m = r.metrics;
  let html = "";
  if (m) {
    html += `<div>Observation-level Top-1 accuracy (HMM)</div><div class="big">${pct(m.hmm.top1_accuracy)}</div>`;
    html += `<table><tr><th></th><th>HMM/Viterbi</th><th>Classifier only</th></tr>
      <tr><td>Top-1 accuracy</td><td>${pct(m.hmm.top1_accuracy)}</td><td>${pct(m.classifier_only.top1_accuracy)}</td></tr>
      <tr><td>Route continuity</td><td>${pct(m.hmm.route_continuity)}</td><td>${pct(m.classifier_only.route_continuity)}</td></tr>
      <tr><td>Route F1 (length)</td><td>${pct(m.hmm.route_f1)}</td><td>${pct(m.classifier_only.route_f1)}</td></tr>
      <tr><td>Route precision</td><td>${pct(m.hmm.route_precision)}</td><td>${pct(m.classifier_only.route_precision)}</td></tr>
      <tr><td>Route recall</td><td>${pct(m.hmm.route_recall)}</td><td>${pct(m.classifier_only.route_recall)}</td></tr></table>
      <div class="muted" style="margin-top:6px">Scored server-side on ${m.n_labelled_obs} observations
      (${m.n_uncertain_obs_excluded} off-network observations excluded).</div>`;
  } else {
    html += `<div class="muted">No reference available on the server for this trip.</div>`;
  }
  html += `<table style="margin-top:10px">
    <tr><td>GPS observations</td><td>${r.n_observations}</td></tr>
    <tr><td>Observations matched</td><td>${r.n_predicted}</td></tr>
    <tr><td>Arcs in predicted route</td><td>${r.predicted_route_arcs.length}</td></tr>
    <tr><td>Candidates + features</td><td>${r.timing_s.candidates_features_s} s</td></tr>
    <tr><td>Classifier</td><td>${r.timing_s.classifier_s} s</td></tr>
    <tr><td>Viterbi</td><td>${r.timing_s.viterbi_s} s</td></tr></table>`;
  $("results").innerHTML = html;
  $("results").className = "";
  $("map").srcdoc = r.map_html;
}

$("connect").onclick = connect;
connect();
