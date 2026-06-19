"use strict";

const PALETTE = ["#6ea8fe", "#3fb950", "#f0883e", "#db61a2", "#e3b341", "#56d4dd", "#f85149", "#a371f7"];
const DEFAULT_A = "rollout/episodic_return";
const DEFAULT_B = "grad_norm/actor/global";

const state = {
  runs: [],            // [{id, status, algo, env, seed}]
  selected: new Set(), // run ids overlaid on charts
  focus: null,         // run id used for the histogram
  metricA: DEFAULT_A,
  metricB: DEFAULT_B,
  charts: {},          // id -> uPlot instance
  colorOf: {},         // run id -> color
};

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}

function colorFor(runId) {
  if (!(runId in state.colorOf)) {
    state.colorOf[runId] = PALETTE[Object.keys(state.colorOf).length % PALETTE.length];
  }
  return state.colorOf[runId];
}

// ---- run sidebar ----------------------------------------------------------
async function loadRuns() {
  state.runs = await api("/api/runs");
  // auto-select everything the first time
  if (state.selected.size === 0 && state.runs.length) {
    state.runs.forEach((r) => state.selected.add(r.id));
    state.focus = state.runs[state.runs.length - 1].id;
  }
  renderRuns();
}

function renderRuns() {
  const el = document.getElementById("runs");
  el.innerHTML = "";
  if (!state.runs.length) {
    el.innerHTML = '<div class="empty">no runs yet</div>';
    return;
  }
  for (const r of state.runs) {
    const row = document.createElement("div");
    row.className = "run" + (state.focus === r.id ? " focus" : "");
    const checked = state.selected.has(r.id) ? "checked" : "";
    row.innerHTML = `
      <input type="checkbox" ${checked} />
      <span class="swatch" style="background:${colorFor(r.id)}"></span>
      <span class="dot ${r.status}"></span>
      <span class="meta">
        <span class="name">${r.algo || "?"} · s${r.seed ?? "?"}</span>
        <span class="env">${r.env || r.id}</span>
      </span>`;
    row.querySelector("input").addEventListener("change", (e) => {
      e.stopPropagation();
      if (e.target.checked) state.selected.add(r.id);
      else state.selected.delete(r.id);
      redraw();
    });
    row.addEventListener("click", () => {
      state.focus = r.id;
      renderRuns();
      drawHist();
      drawVideo();
    });
    el.appendChild(row);
  }
}

// ---- metric selectors -----------------------------------------------------
async function loadTags() {
  const tagSet = new Set();
  for (const id of state.selected) {
    try {
      const t = await api(`/api/runs/${id}/tags`);
      t.scalars.forEach((x) => tagSet.add(x));
    } catch (_) {}
  }
  const tags = [...tagSet].sort();
  fillSelect("metricA", tags, state.metricA);
  fillSelect("metricB", tags, state.metricB);
}

function fillSelect(elId, tags, current) {
  const sel = document.getElementById(elId);
  const keep = tags.includes(current) ? current : (tags[0] || "");
  sel.innerHTML = tags.map((t) => `<option ${t === keep ? "selected" : ""}>${t}</option>`).join("");
  if (elId === "metricA") state.metricA = keep;
  else state.metricB = keep;
}

// ---- chart drawing --------------------------------------------------------
// Merge multiple runs' (steps,values) into a shared x axis with nulls for gaps.
function mergeSeries(seriesList) {
  const xs = new Set();
  seriesList.forEach((s) => s.steps.forEach((x) => xs.add(x)));
  const x = [...xs].sort((a, b) => a - b);
  const index = new Map(x.map((v, i) => [v, i]));
  const cols = seriesList.map((s) => {
    const col = new Array(x.length).fill(null);
    s.steps.forEach((step, i) => (col[index.get(step)] = s.values[i]));
    return col;
  });
  return [x, ...cols];
}

async function drawChart(divId, chartKey, tag) {
  const runIds = [...state.selected];
  const fetched = [];
  for (const id of runIds) {
    try {
      const d = await api(`/api/runs/${id}/scalars?tag=${encodeURIComponent(tag)}`);
      if (d.steps.length) fetched.push({ id, steps: d.steps, values: d.values });
    } catch (_) {}
  }
  const div = document.getElementById(divId);
  if (!fetched.length) {
    if (state.charts[chartKey]) { state.charts[chartKey].destroy(); delete state.charts[chartKey]; }
    div.innerHTML = '<div class="empty">no data for this metric</div>';
    return;
  }
  div.innerHTML = "";
  const data = mergeSeries(fetched);
  const series = [{}].concat(
    fetched.map((s) => ({
      label: s.id.split("-").slice(0, 2).join("-"),
      stroke: colorFor(s.id),
      width: 1.5,
      spanGaps: true,
      points: { show: false },
    }))
  );
  const opts = {
    width: div.clientWidth || 600,
    height: 260,
    title: tag,
    scales: { x: { time: false } },
    axes: [
      { stroke: "#8b94a7", grid: { stroke: "#262b36" } },
      { stroke: "#8b94a7", grid: { stroke: "#262b36" } },
    ],
    series,
  };
  if (state.charts[chartKey]) state.charts[chartKey].destroy();
  state.charts[chartKey] = new uPlot(opts, data, div);
}

async function drawHist() {
  const div = document.getElementById("chartHist");
  const id = state.focus;
  document.getElementById("histTitle").textContent =
    "action distribution" + (id ? ` · ${id.split("-").slice(0, 2).join("-")}` : "");
  if (!id) { div.innerHTML = '<div class="empty">select a run</div>'; return; }
  let h;
  try { h = await api(`/api/runs/${id}/histogram?tag=actions`); }
  catch (_) { div.innerHTML = '<div class="empty">no action histogram</div>'; return; }
  if (!h.counts || !h.counts.length) { div.innerHTML = '<div class="empty">no action histogram</div>'; return; }
  const centers = h.counts.map((_, i) => (h.edges[i] + h.edges[i + 1]) / 2);
  div.innerHTML = "";
  const opts = {
    width: div.clientWidth || 400,
    height: 240,
    scales: { x: { time: false } },
    axes: [
      { stroke: "#8b94a7", grid: { stroke: "#262b36" } },
      { stroke: "#8b94a7", grid: { stroke: "#262b36" } },
    ],
    series: [
      {},
      { label: "count", stroke: colorFor(id), fill: colorFor(id) + "55",
        paths: uPlot.paths.bars({ size: [0.9, 100] }), points: { show: false } },
    ],
  };
  if (state.charts.hist) state.charts.hist.destroy();
  state.charts.hist = new uPlot(opts, [centers, h.counts], div);
}

async function drawVideo() {
  const id = state.focus;
  const sel = document.getElementById("videoStep");
  const video = document.getElementById("rollout");
  if (!id) { sel.innerHTML = ""; video.removeAttribute("src"); return; }
  let data;
  try { data = await api(`/api/runs/${id}/videos`); }
  catch (_) { return; }
  const vids = data.videos || [];
  if (!vids.length) {
    sel.innerHTML = '<option>none (train with --record-video)</option>';
    video.removeAttribute("src");
    return;
  }
  const prev = sel.value;
  sel.innerHTML = vids
    .map((v) => `<option value="${v.url}">step ${v.step ?? v.name}</option>`)
    .join("");
  // keep selection if still present, else jump to latest
  const urls = vids.map((v) => v.url);
  const chosen = urls.includes(prev) ? prev : urls[urls.length - 1];
  sel.value = chosen;
  if (video.getAttribute("src") !== chosen) {
    video.src = chosen;
    video.load();
  }
}

document.getElementById("videoStep").addEventListener("change", (e) => {
  const v = document.getElementById("rollout");
  v.src = e.target.value; v.load();
});

// ---- orchestration --------------------------------------------------------
async function redraw() {
  await loadTags();
  await drawChart("chartA", "A", state.metricA);
  await drawChart("chartB", "B", state.metricB);
  await drawHist();
  await drawVideo();
}

document.getElementById("metricA").addEventListener("change", (e) => {
  state.metricA = e.target.value; drawChart("chartA", "A", state.metricA);
});
document.getElementById("metricB").addEventListener("change", (e) => {
  state.metricB = e.target.value; drawChart("chartB", "B", state.metricB);
});

async function tick() {
  if (!document.getElementById("liveToggle").checked) return;
  await loadRuns();
  await redraw();
}

(async function init() {
  await loadRuns();
  await redraw();
  setInterval(tick, 1500);
  window.addEventListener("resize", () => redraw());
})();
