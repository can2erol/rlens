"use strict";

const PALETTE = ["#6ea8fe", "#3fb950", "#f0883e", "#db61a2", "#e3b341", "#56d4dd", "#f85149", "#a371f7"];
const DEFAULT_A = "rollout/episodic_return";

const state = {
  runs: [],            // [{id, status, algo, env, seed}]
  selected: new Set(), // run ids overlaid on charts
  focus: null,         // run id used for the histogram / config panel
  metricA: DEFAULT_A,
  charts: {},          // id -> uPlot instance (featured chart + histogram)
  gridCharts: {},      // tag -> uPlot instance (metric grid)
  gridData: {},        // run id -> { tag: {steps, values, times} } (cached for re-render)
  colorOf: {},         // run id -> color
  smoothing: 0.6,      // EMA weight on history (0 = raw)
  xaxis: "step",       // "step" | "time"
  configDiff: false,   // config panel: diff selected vs focus only
  compareRows: [],     // cached run summaries for the table
  sortKey: "id",
  sortDir: 1,
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

const shortId = (id) => id.split("-").slice(0, 2).join("-");
const fmtNum = (v) => (v == null ? "—" : (Math.abs(v) >= 1000 ? Math.round(v).toString() : v.toFixed(1)));
const fmtInt = (v) => (v == null ? "—" : Math.round(v).toLocaleString());
function fmtVal(v) {
  if (v == null || v === "") return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v.toString() : v.toFixed(4).replace(/0+$/, "");
  return String(v);
}

// ---- run sidebar ----------------------------------------------------------
async function loadRuns() {
  state.runs = await api("/api/runs");
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
      drawConfig();
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
  const sel = document.getElementById("metricA");
  const keep = tags.includes(state.metricA) ? state.metricA : (tags[0] || "");
  sel.innerHTML = tags.map((t) => `<option ${t === keep ? "selected" : ""}>${t}</option>`).join("");
  state.metricA = keep;
}

// ---- chart drawing --------------------------------------------------------
// Exponential moving average (TensorBoard-style), applied per run before merge.
function ema(values, alpha) {
  if (!alpha) return values;
  const out = [];
  let prev = null;
  for (const v of values) {
    if (v == null) { out.push(null); continue; }
    prev = prev == null ? v : alpha * prev + (1 - alpha) * v;
    out.push(prev);
  }
  return out;
}

// Merge runs with independent x arrays onto a shared, sorted x axis (nulls for gaps).
function mergeSeriesXY(seriesList) {
  const xs = new Set();
  seriesList.forEach((s) => s.x.forEach((v) => xs.add(v)));
  const x = [...xs].sort((a, b) => a - b);
  const index = new Map(x.map((v, i) => [v, i]));
  const cols = seriesList.map((s) => {
    const col = new Array(x.length).fill(null);
    s.x.forEach((xv, i) => (col[index.get(xv)] = s.y[i]));
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
      if (d.steps.length) fetched.push({ id, steps: d.steps, values: d.values, times: d.times || [] });
    } catch (_) {}
  }
  const div = document.getElementById(divId);
  if (!fetched.length) {
    if (state.charts[chartKey]) { state.charts[chartKey].destroy(); delete state.charts[chartKey]; }
    div.innerHTML = '<div class="empty">no data for this metric</div>';
    return;
  }
  div.innerHTML = "";
  const useTime = state.xaxis === "time";
  const prepared = fetched.map((s) => {
    const x = useTime && s.times.length ? s.times.map((t) => (t - s.times[0]) / 60) : s.steps;
    return { id: s.id, x, y: ema(s.values, state.smoothing) };
  });
  const data = mergeSeriesXY(prepared);
  const series = [{}].concat(
    prepared.map((s) => ({
      label: shortId(s.id),
      stroke: colorFor(s.id),
      width: 1.5,
      spanGaps: true,
      points: { show: false },
    }))
  );
  const opts = {
    width: div.clientWidth || 600,
    height: 260,
    title: tag + (useTime ? "  (x: minutes)" : ""),
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
    "action distribution" + (id ? ` · ${shortId(id)}` : "");
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

// ---- metric grid (all logged scalars at once) -----------------------------
function destroyGridCharts() {
  for (const k in state.gridCharts) { try { state.gridCharts[k].destroy(); } catch (_) {} }
  state.gridCharts = {};
}

async function fetchGridData() {
  const data = {};
  for (const id of state.selected) {
    try { data[id] = (await api(`/api/runs/${id}/scalars_all`)).scalars; } catch (_) { data[id] = {}; }
  }
  state.gridData = data;
}

// Render from cached gridData, so smoothing/x-axis changes don't refetch.
function renderMetricGrid() {
  const div = document.getElementById("grid");
  destroyGridCharts();
  const ids = [...state.selected].filter((id) => state.gridData[id]);
  if (!ids.length) { div.innerHTML = '<div class="empty">no runs selected</div>'; return; }
  const tags = [...new Set(ids.flatMap((id) => Object.keys(state.gridData[id])))].sort();
  if (!tags.length) { div.innerHTML = '<div class="empty">no metrics yet</div>'; return; }

  // group by namespace (the part before the first "/")
  const groups = {};
  for (const t of tags) {
    const g = t.includes("/") ? t.split("/")[0] : "misc";
    (groups[g] ||= []).push(t);
  }

  div.innerHTML = "";
  const useTime = state.xaxis === "time";
  for (const g of Object.keys(groups).sort()) {
    const head = document.createElement("div");
    head.className = "mgroup-h";
    head.textContent = g;
    div.appendChild(head);
    for (const tag of groups[g]) {
      const cell = document.createElement("div");
      cell.className = "mcell";
      const title = document.createElement("div");
      title.className = "mtitle";
      title.textContent = tag.includes("/") ? tag.split("/").slice(1).join("/") : tag;
      title.title = tag;
      cell.appendChild(title);
      const plot = document.createElement("div");
      cell.appendChild(plot);
      div.appendChild(cell);

      const prepared = [];
      for (const id of ids) {
        const d = state.gridData[id][tag];
        if (d && d.steps.length) {
          const x = useTime && d.times && d.times.length ? d.times.map((v) => (v - d.times[0]) / 60) : d.steps;
          prepared.push({ id, x, y: ema(d.values, state.smoothing) });
        }
      }
      if (!prepared.length) { plot.innerHTML = '<div class="empty" style="padding:8px">—</div>'; continue; }
      const data = mergeSeriesXY(prepared);
      const series = [{}].concat(
        prepared.map((s) => ({ stroke: colorFor(s.id), width: 1.25, spanGaps: true, points: { show: false } }))
      );
      state.gridCharts[tag] = new uPlot(
        {
          width: plot.clientWidth || 280,
          height: 120,
          scales: { x: { time: false } },
          legend: { show: false },
          cursor: { show: false },
          axes: [
            { stroke: "#6b7280", grid: { stroke: "#20242e" }, size: 22, font: "9px sans-serif" },
            { stroke: "#6b7280", grid: { stroke: "#20242e" }, size: 38, font: "9px sans-serif" },
          ],
          series,
        },
        data,
        plot
      );
    }
  }
}

async function drawMetricGrid() {
  await fetchGridData();
  renderMetricGrid();
}

// ---- run comparison table -------------------------------------------------
async function drawCompare() {
  const ids = [...state.selected];
  const rows = [];
  for (const id of ids) {
    try { rows.push(await api(`/api/runs/${id}/summary`)); } catch (_) {}
  }
  state.compareRows = rows;
  renderCompare();
}

function renderCompare() {
  const div = document.getElementById("compare");
  const rows = [...state.compareRows];
  if (!rows.length) { div.innerHTML = '<div class="empty">no runs selected</div>'; return; }
  const cols = [
    { k: "id", label: "run", fmt: (r) => shortId(r.id) },
    { k: "algo", label: "algo" },
    { k: "env", label: "env" },
    { k: "seed", label: "seed" },
    { k: "status", label: "status" },
    { k: "steps", label: "steps", fmt: (r) => fmtInt(r.steps) },
    { k: "return_best", label: "return (best)", fmt: (r) => fmtNum(r.return_best) },
    { k: "eval_best", label: "eval (best)", fmt: (r) => fmtNum(r.eval_best) },
    { k: "fps", label: "fps", fmt: (r) => fmtInt(r.fps) },
  ];
  rows.sort((a, b) => {
    let x = a[state.sortKey], y = b[state.sortKey];
    if (x == null) x = -Infinity;
    if (y == null) y = -Infinity;
    if (typeof x === "string" || typeof y === "string") return state.sortDir * String(x).localeCompare(String(y));
    return state.sortDir * ((x > y) - (x < y));
  });
  const thead = cols.map((c) => {
    const arrow = state.sortKey === c.k ? (state.sortDir > 0 ? " ▲" : " ▼") : "";
    return `<th data-k="${c.k}">${c.label}${arrow}</th>`;
  }).join("");
  const body = rows.map((r) => {
    const tds = cols.map((c, i) => {
      const sw = i === 0 ? `<span class="swatch" style="background:${colorFor(r.id)}"></span> ` : "";
      const v = c.fmt ? c.fmt(r) : (r[c.k] ?? "—");
      return `<td>${sw}${v}</td>`;
    }).join("");
    return `<tr>${tds}</tr>`;
  }).join("");
  div.innerHTML = `<table class="cmp"><thead><tr>${thead}</tr></thead><tbody>${body}</tbody></table>`;
  div.querySelectorAll("th").forEach((th) =>
    th.addEventListener("click", () => {
      const k = th.dataset.k;
      if (state.sortKey === k) state.sortDir *= -1;
      else { state.sortKey = k; state.sortDir = 1; }
      renderCompare();
    })
  );
}

// ---- config panel ---------------------------------------------------------
function flatten(obj, prefix = "") {
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) Object.assign(out, flatten(v, key));
    else out[key] = Array.isArray(v) ? `[${v.join(", ")}]` : v;
  }
  return out;
}

async function drawConfig() {
  const div = document.getElementById("config");
  if (state.configDiff) return drawConfigDiff(div);
  const id = state.focus;
  if (!id) { div.innerHTML = '<div class="empty">select a run</div>'; return; }
  let meta;
  try { meta = await api(`/api/runs/${id}/meta`); }
  catch (_) { div.innerHTML = '<div class="empty">no config</div>'; return; }
  const head = {
    status: meta.status, final_step: meta.final_step,
    best_return: meta.best_return, best_step: meta.best_step,
  };
  const cfg = flatten(meta.config || {});
  const rows = Object.entries({ ...head, ...cfg })
    .map(([k, v]) => `<tr><td class="k">${k}</td><td>${fmtVal(v)}</td></tr>`)
    .join("");
  div.innerHTML = `<table class="cfg"><tbody>${rows}</tbody></table>`;
}

async function drawConfigDiff(div) {
  const ids = [...state.selected];
  if (ids.length < 2) { div.innerHTML = '<div class="empty">select 2+ runs to diff</div>'; return; }
  const metas = {};
  for (const id of ids) {
    try { metas[id] = flatten((await api(`/api/runs/${id}/meta`)).config || {}); } catch (_) {}
  }
  const present = ids.filter((id) => metas[id]);
  const keys = [...new Set(present.flatMap((id) => Object.keys(metas[id])))].sort();
  const head = `<th>key</th>` + present.map((id) =>
    `<th><span class="swatch" style="background:${colorFor(id)}"></span> ${shortId(id)}</th>`).join("");
  const body = keys.map((k) => {
    const vals = present.map((id) => metas[id][k]);
    const differ = new Set(vals.map((v) => JSON.stringify(v ?? null))).size > 1;
    const tds = vals.map((v) => `<td>${fmtVal(v)}</td>`).join("");
    return `<tr class="${differ ? "diff" : ""}"><td class="k">${k}</td>${tds}</tr>`;
  }).join("");
  div.innerHTML = `<table class="cfg"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
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
  await drawMetricGrid();
  await drawHist();
  await drawCompare();
  await drawConfig();
  await drawVideo();
}

document.getElementById("metricA").addEventListener("change", (e) => {
  state.metricA = e.target.value; drawChart("chartA", "A", state.metricA);
});
// smoothing & x-axis re-render from cache (no refetch) — keeps the slider snappy
document.getElementById("smooth").addEventListener("input", (e) => {
  state.smoothing = Number(e.target.value) / 100;
  drawChart("chartA", "A", state.metricA);
  renderMetricGrid();
});
document.getElementById("xaxis").addEventListener("change", (e) => {
  state.xaxis = e.target.value;
  drawChart("chartA", "A", state.metricA);
  renderMetricGrid();
});
document.getElementById("diffToggle").addEventListener("change", (e) => {
  state.configDiff = e.target.checked;
  drawConfig();
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
