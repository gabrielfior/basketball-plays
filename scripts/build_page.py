"""Build the offline labelling page: name the clusters, label held-out possessions, read results.

    uv run python scripts/build_page.py --out data/plays/page/index.html

Everything the page needs is inlined into a single HTML file -- montage PNGs as base64 data
URIs, per-possession animations as sampled canonical court coordinates, the cluster summaries,
the defence classifications and the current `labels.json`. No external script, style or font is
referenced, so the file works from `file://` with no network.

The user's decisions live in `localStorage` while they work and are exported from the page as a
`labels.Labels` JSON blob to paste back into `data/plays/labels.json`.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import gameinfo, games, halfcourt
from basketball_plays import labels as L
from basketball_plays.court import NCAA

FPS = 5.0
FALLBACK_FPS = 2.5
PRE_S = 1.0
POST_S = 8.0
MAX_VALIDATION = 150
SIZE_LIMIT = 15 * 1024 * 1024
HELD_OUT_SPLITS = ("test", "ncaa")

BUCKET_LABELS = {
    "ato": "After timeout",
    "inbound": "Frontcourt inbound",
    "after_score": "After made basket",
}
BUCKET_ORDER = ("ato", "inbound", "after_score")


# --------------------------------------------------------------------------- inputs


def load_records(path: str | Path) -> dict[str, halfcourt.HalfcourtRecord]:
    """Record key (`"<game_id>:<period>:<index>"`) -> `HalfcourtRecord`."""
    out: dict[str, halfcourt.HalfcourtRecord] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = halfcourt.HalfcourtRecord.from_dict(json.loads(line))
                out[f"{rec.game_id}:{rec.period}:{rec.index}"] = rec
    return out


def _nearest(times: list[float], target: float, tol: float) -> float | None:
    """The stored sample time closest to `target`, or None when the gap is wider than `tol`."""
    if not times:
        return None
    lo, hi = 0, len(times)
    while lo < hi:  # bisect_left
        mid = (lo + hi) // 2
        if times[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    best = None
    for i in (lo - 1, lo):
        if 0 <= i < len(times) and (best is None or abs(times[i] - target) < abs(best - target)):
            best = times[i]
    return best if best is not None and abs(best - target) <= tol else None


def sample_animation(rec: halfcourt.HalfcourtRecord, fps: float = FPS,
                     pre: float = PRE_S, post: float = POST_S) -> dict | None:
    """Duke and opponent positions plus the ball handler, sampled around the setup frame.

    Frames run from `setup - pre` to `setup + post` at `fps`, in canonical court feet rounded to
    one decimal. Records with neither a setup frame nor a `t0` have nothing to animate and
    return None; individual frames outside the tracked window come back empty rather than
    stretching the nearest one.
    """
    anchor = rec.setup if rec.setup is not None else rec.t0
    if anchor is None:
        return None
    duke = halfcourt.tracks_from_record(rec)
    opp = halfcourt.tracks_from_players(rec.opponents)
    if not duke and not opp:
        return None
    times = sorted({t for tr in duke + opp for t in tr.xy})
    handler = {round(row[0], 3): (row[1], row[2]) for row in rec.ball_handler}
    htimes = sorted(handler)
    step = 1.0 / fps
    tol = step / 2 + 1e-6
    n = round((pre + post) * fps) + 1
    frames_duke, frames_opp, frames_ball = [], [], []
    for i in range(n):
        t = anchor - pre + i * step
        td = _nearest(times, t, tol)
        frames_duke.append(_round_pts(halfcourt.positions_at(duke, td)) if td is not None else [])
        frames_opp.append(_round_pts(halfcourt.positions_at(opp, td)) if td is not None else [])
        th = _nearest(htimes, t, tol)
        ball = [round(handler[th][0], 1), round(handler[th][1], 1)] if th is not None else None
        frames_ball.append(ball)
    return {"fps": fps, "pre": pre, "duke": frames_duke, "opp": frames_opp, "handler": frames_ball}


def _round_pts(pts) -> list[list[float]]:
    return [[round(x, 1), round(y, 1)] for x, y in pts]


def opponent_names(game_ids, root: str | Path = games.DEFAULT_ROOT) -> dict[str, str]:
    """Game id -> opponent short name, from the ESPN summary, falling back to the registry."""
    by_id = {g.espn_id: g for g in games.load_registry()}
    out: dict[str, str] = {}
    for gid in game_ids:
        summary = Path(root) / gid / "espn_summary.json"
        name = None
        if summary.exists():
            try:
                info = gameinfo.from_summary(json.loads(summary.read_text()))
                name = info.away if info.home == "Duke" else info.home
            except (KeyError, ValueError, json.JSONDecodeError):
                name = None
        if name is None and gid in by_id:
            name = by_id[gid].opponent
        out[gid] = name or gid
    return out


def encode_montages(montage_dir: str | Path, cluster_ids) -> dict[str, str]:
    """Cluster id -> `data:image/png;base64,...` for every montage that exists on disk."""
    out: dict[str, str] = {}
    for cid in cluster_ids:
        p = Path(montage_dir) / f"cluster_{cid}.png"
        if p.exists():
            out[str(cid)] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return out


def stratified_sample(rows: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """Up to `limit` rows, allocated across buckets in proportion to how common they are."""
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r.get("bucket", "other")].append(r)
    for v in by_bucket.values():
        v.sort(key=lambda r: (r["game_id"], r["period"], r["index"]))
        rng.shuffle(v)
    total = len(rows)
    if total <= limit:
        take = {b: len(v) for b, v in by_bucket.items()}
    else:
        take = {b: int(limit * len(v) / total) for b, v in by_bucket.items()}
        order = sorted(by_bucket, key=lambda b: (-(limit * len(by_bucket[b]) / total % 1), b))
        i = 0
        while sum(take.values()) < limit and i < 10 * len(order):
            b = order[i % len(order)]
            if take[b] < len(by_bucket[b]):
                take[b] += 1
            i += 1
    picked = [r for b, v in by_bucket.items() for r in v[:take[b]]]
    picked.sort(key=lambda r: (r["game_id"], r["period"], r["index"]))
    return picked


def validation_rows(rows: list[dict], limit: int = MAX_VALIDATION,
                    seed: int = 0) -> tuple[list[dict], bool]:
    """The validation sample and whether it had to fall back to training rows."""
    held = [r for r in rows if r.get("split") in HELD_OUT_SPLITS]
    fallback = not held
    return stratified_sample(held or rows, limit, seed), fallback


# --------------------------------------------------------------------------- rendering


def _normalise_assignments(clusters: dict) -> dict[str, dict]:
    """`clusters["labels"]` as `key -> {"cluster": int, "assigned_only": bool}`.

    Tolerates the bare `key -> int` form as well as the `{"cluster", "assigned_only"}` dicts
    `cluster_plays.py` writes.
    """
    out: dict[str, dict] = {}
    for key, val in (clusters.get("labels") or {}).items():
        if isinstance(val, dict):
            out[key] = {"cluster": int(val["cluster"]),
                        "assigned_only": bool(val.get("assigned_only", False))}
        else:
            out[key] = {"cluster": int(val), "assigned_only": False}
    return out


def _json_for_script(obj) -> str:
    """Compact JSON safe to drop inside a `<script>` block."""
    return (json.dumps(obj, separators=(",", ":"), allow_nan=False)
            .replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_page(clusters: dict, defense: dict, animations: dict, meta: dict, montages: dict,
                labels: dict, validation_keys, *, fallback_split: bool = False,
                generated: str = "") -> str:
    """The whole page as one HTML string. No I/O, so the tests can drive it with fixtures."""
    cards = []
    for c in clusters.get("clusters", []):
        cards.append({
            "cluster": int(c["cluster"]),
            "n": c.get("n", 0),
            "n_fit": c.get("n_fit"),
            "n_by_bucket": c.get("n_by_bucket", {}),
            "n_by_split": c.get("n_by_split", {}),
            "top_zones": list(c.get("top_zones", [])),
            "handler_path": list(c.get("handler_path", [])),
            "members": [list(m) for m in c.get("members", [])],
        })
    data = {
        "k": clusters.get("k"),
        "silhouette": clusters.get("silhouette"),
        "stability_ari": clusters.get("stability_ari"),
        "small_corpus": bool(clusters.get("small_corpus", False)),
        "n_fit": clusters.get("n_fit"),
        "n_assigned_only": clusters.get("n_assigned_only"),
        "clusters": cards,
        "assign": _normalise_assignments(clusters),
        "defense": defense,
        "anim": animations,
        "meta": meta,
        "montages": {str(k): v for k, v in (montages or {}).items()},
        "labels": labels or {"clusters": {}, "validation": {}, "defense": {}, "version": 1},
        "validation": list(validation_keys),
        "fallback_split": bool(fallback_split),
        "seeds": list(L.SEED_NAMES),
        "buckets": BUCKET_LABELS,
        "bucket_order": list(BUCKET_ORDER),
        "generated": generated,
        "court": {
            "length": NCAA.length, "width": NCAA.width, "rim": [NCAA.rim_offset, NCAA.width / 2],
            "rim_radius": 0.75,
            "three": NCAA.three_point_radius, "sideline": NCAA.three_point_sideline_offset,
            "straight": round(NCAA.three_point_straight_length, 3),
            "paint_width": NCAA.paint_width, "paint_length": NCAA.paint_length,
        },
    }
    return TEMPLATE.replace("__DATA__", _json_for_script(data))


# --------------------------------------------------------------------------- the page


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duke Set Labeller</title>
<style>
:root{
  --bg:#f7f6f3; --panel:#ffffff; --panel-2:#f2f0ec; --fg:#1c1b19; --muted:#6d6862;
  --line:#e0dbd3; --line-strong:#c9c2b7;
  --accent:#0a5fbd; --accent-fg:#ffffff; --accent-soft:#e6effa;
  --warn:#7a5100; --warn-bg:#fdf3dc; --warn-line:#e6cd8f;
  --court:#efece5; --courtline:#b9b1a4; --duke:#0a5fbd; --opp:#b23b2c; --ball:#d97706;
  --thin:#a29a8e;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#131418; --panel:#1b1d23; --panel-2:#22252c; --fg:#e9e7e3; --muted:#9c968d;
    --line:#2e323a; --line-strong:#434955;
    --accent:#6aa9f5; --accent-fg:#0d1117; --accent-soft:#1d2b3f;
    --warn:#f0c675; --warn-bg:#2e2515; --warn-line:#5c4a22;
    --court:#20232a; --courtline:#4a505c; --duke:#6aa9f5; --opp:#e0705f; --ball:#f0a44a;
    --thin:#6b6660;
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
h1,h2,h3{margin:0; font-weight:650; letter-spacing:-0.011em}
h1{font-size:19px}
h2{font-size:17px}
h3{font-size:15px}
a{color:var(--accent)}
.num{font-variant-numeric:tabular-nums}

header{
  position:sticky; top:0; z-index:20; background:var(--panel);
  border-bottom:1px solid var(--line);
}
.bar{display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; padding:12px 18px 10px}
.bar .sub{color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums}
.bar .spacer{flex:1}
nav{display:flex; gap:2px; padding:0 12px; overflow-x:auto}
nav button{
  appearance:none; background:none; border:0; border-bottom:2px solid transparent;
  color:var(--muted); font:inherit; font-weight:550; padding:8px 12px 9px; cursor:pointer;
  white-space:nowrap;
}
nav button:hover{color:var(--fg)}
nav button[aria-selected="true"]{color:var(--accent); border-bottom-color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:3px}

main{max-width:1160px; margin:0 auto; padding:18px 18px 64px}
section[hidden]{display:none}
.lede{color:var(--muted); font-size:13.5px; margin:0 0 16px; max-width:78ch}
.banner{
  background:var(--warn-bg); border:1px solid var(--warn-line); color:var(--warn);
  border-radius:8px; padding:9px 12px; font-size:13.5px; margin:0 0 16px;
}
.card{
  background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px 16px; margin-bottom:14px;
}
.card > header{position:static; background:none; border:0; padding:0}
.chead{display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; margin-bottom:10px}
.chead .sub{color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums}
.pill{
  display:inline-block; background:var(--panel-2); border:1px solid var(--line);
  border-radius:999px; padding:1px 9px; font-size:12px; color:var(--muted);
}
.pill.on{background:var(--accent-soft); border-color:var(--accent); color:var(--accent)}
.grid2{display:grid; grid-template-columns:minmax(260px,320px) 1fr; gap:18px; align-items:start}
@media (max-width:720px){ .grid2{grid-template-columns:1fr} }
.montage{width:100%; border:1px solid var(--line); border-radius:8px; display:block; background:var(--panel-2)}
.kv{margin:0 0 10px; font-size:13.5px}
.kv dt{color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em}
.kv dd{margin:1px 0 8px; font-variant-numeric:tabular-nums}
.path{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px}

.controls{display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; margin-top:8px;
  padding-top:12px; border-top:1px solid var(--line)}
label.f{display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--muted)}
input[type=text],select,textarea{
  font:inherit; color:var(--fg); background:var(--panel); border:1px solid var(--line-strong);
  border-radius:6px; padding:6px 8px;
}
input[type=text]{min-width:190px; max-width:100%}
select{min-width:150px; max-width:100%}
textarea{width:100%; min-height:190px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:12.5px; line-height:1.45}
label.chk{display:flex; gap:6px; align-items:center; font-size:13.5px; color:var(--fg);
  padding-bottom:7px}
button.btn{
  font:inherit; font-weight:550; background:var(--accent); color:var(--accent-fg);
  border:1px solid var(--accent); border-radius:7px; padding:7px 13px; cursor:pointer;
}
button.btn.ghost{background:var(--panel); color:var(--fg); border-color:var(--line-strong)}
button.btn:hover{filter:brightness(1.06)}
.resolved{font-size:13px; color:var(--muted); padding-bottom:8px}
.resolved b{color:var(--accent); font-weight:600}

.tablewrap{overflow-x:auto; border:1px solid var(--line); border-radius:9px; background:var(--panel)}
table{border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums}
th,td{padding:7px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{
  font-size:11.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
  font-weight:600; background:var(--panel-2);
}
tbody tr:last-child td{border-bottom:0}
tbody tr.total td{font-weight:650; border-top:1px solid var(--line-strong)}
td.thin{color:var(--thin)}
td .ppp{color:var(--muted); margin-left:8px}
td.thin .ppp{color:var(--thin)}

.anim{display:flex; flex-direction:column; gap:6px}
.anim{width:280px; max-width:100%}
canvas{
  display:block; width:100%; height:auto; aspect-ratio:280/298; background:var(--court);
  border:1px solid var(--line); border-radius:8px; touch-action:none;
}
.anim .row{display:flex; gap:8px; align-items:center; width:100%}
.anim input[type=range]{flex:1; accent-color:var(--accent)}
.anim .t{font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; min-width:52px;
  text-align:right}
button.tiny{
  font:inherit; font-size:13px; background:var(--panel); color:var(--fg);
  border:1px solid var(--line-strong); border-radius:6px; padding:3px 9px; cursor:pointer;
  min-width:34px;
}
.legend{font-size:12px; color:var(--muted); display:flex; gap:12px; width:100%; flex-wrap:wrap}
.legend i{display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px}
details.ref{margin:6px 0 2px; font-size:13px}
details.ref summary{cursor:pointer; color:var(--muted)}
details.ref img{width:100%; max-width:320px; margin-top:8px; border:1px solid var(--line);
  border-radius:8px; display:block}
fieldset{border:0; margin:0; padding:0}
fieldset legend{font-size:12px; color:var(--muted); padding:0 0 4px}
.radios{display:flex; gap:12px; flex-wrap:wrap; font-size:13.5px}
.radios label{display:flex; gap:5px; align-items:center}
.done{color:var(--accent); font-weight:600}
.empty{color:var(--muted); font-size:13.5px; padding:10px 0}
.status{font-size:12.5px; color:var(--muted)}
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>Duke Set Labeller</h1>
    <span class="sub" id="subtitle"></span>
    <span class="spacer"></span>
    <span class="status" id="status"></span>
    <button class="btn ghost" id="go-export" type="button">Export decisions</button>
  </div>
  <nav id="tabs" role="tablist">
    <button type="button" role="tab" data-pane="clusters" aria-selected="true">Clusters</button>
    <button type="button" role="tab" data-pane="validation" aria-selected="false">Validation</button>
    <button type="button" role="tab" data-pane="results" aria-selected="false">Results</button>
    <button type="button" role="tab" data-pane="browse" aria-selected="false">Browse</button>
    <button type="button" role="tab" data-pane="export" aria-selected="false">Export</button>
  </nav>
</header>
<main>
  <section id="pane-clusters" role="tabpanel"></section>
  <section id="pane-validation" role="tabpanel" hidden></section>
  <section id="pane-results" role="tabpanel" hidden></section>
  <section id="pane-browse" role="tabpanel" hidden></section>
  <section id="pane-export" role="tabpanel" hidden></section>
</main>
<datalist id="seed-names"></datalist>
<script>
const DATA = __DATA__;
const LS_KEY = "duke-set-labeller/v1";
const C = DATA.court;

/* ---------------------------------------------------------------- state */

function blankState(){ return {clusters:{}, validation:{}, defense:{}, version:1}; }
let S = blankState();
(function restore(){
  let stored = null;
  try { const raw = localStorage.getItem(LS_KEY); if (raw) stored = JSON.parse(raw); } catch(e){}
  const src = stored || DATA.labels || {};
  S = {clusters: src.clusters || {}, validation: src.validation || {},
       defense: src.defense || {}, version: src.version || 1};
})();

let saveTimer = null;
function save(){
  try { localStorage.setItem(LS_KEY, JSON.stringify(S)); setStatus("Saved to this browser"); }
  catch(e){ setStatus("Could not save: " + e.message); }
  clearTimeout(saveTimer);
  saveTimer = setTimeout(()=> setStatus(countLabelled()), 2500);
}
function setStatus(t){ document.getElementById("status").textContent = t; }
function countLabelled(){
  const named = Object.values(S.clusters).filter(v => v && v !== "discard" &&
    v.indexOf("merge:") !== 0).length;
  return named + " named · " + Object.keys(S.validation).length + " possessions labelled";
}

function resolveName(id, depth){
  depth = depth || 0;
  const v = S.clusters[String(id)];
  if (v == null || v === "discard" || depth > 10) return null;
  if (v.indexOf("merge:") === 0) return resolveName(v.slice(6), depth + 1);
  return v;
}
function namedSets(){
  const out = [];
  DATA.clusters.forEach(c => { const n = resolveName(c.cluster); if (n && out.indexOf(n) < 0) out.push(n); });
  out.sort((a,b) => a.localeCompare(b));
  return out;
}

/* ---------------------------------------------------------------- helpers */

function el(tag, cls, text){
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function bucketLabel(b){ return DATA.buckets[b] || b || "unknown"; }
function metaOf(key){ return DATA.meta[key] || {}; }
function clockStr(sec){
  if (sec == null) return "–";
  const s = Math.max(0, Math.round(sec));
  return Math.floor(s/60) + ":" + String(s%60).padStart(2,"0");
}
function keyTitle(key){
  const m = metaOf(key);
  const parts = [m.opponent || m.game || key, "P" + (m.period != null ? m.period : "?"),
                 "#" + (m.index != null ? m.index : "?"), bucketLabel(m.bucket)];
  if (m.clock != null) parts.push(clockStr(m.clock));
  return parts.join(" · ");
}

/* ---------------------------------------------------------------- court canvas */

const RIMX = C.rim[0], RIMY = C.rim[1], HALF = C.length / 2;
const LANE_Y0 = (C.width - C.paint_width) / 2, LANE_Y1 = LANE_Y0 + C.paint_width;

function makeFrame(cv){
  const pad = 6, w = 280, h = 298;
  const sx = (w - 2*pad) / HALF, sy = (h - 2*pad) / C.width;
  const s = Math.min(sx, sy);
  const ox = (w - HALF * s) / 2, oy = (h - C.width * s) / 2;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return {ctx, s, X: x => ox + x * s, Y: y => oy + y * s};
}

function drawCourt(F){
  const {ctx, s, X, Y} = F;
  const css = getComputedStyle(document.body);
  const line = css.getPropertyValue("--courtline").trim() || "#b9b1a4";
  const bg = css.getPropertyValue("--court").trim() || "#efece5";
  ctx.clearRect(0, 0, 400, 400);
  ctx.fillStyle = bg; ctx.fillRect(0, 0, 400, 400);
  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.lineJoin = "round";
  ctx.strokeRect(X(0), Y(0), HALF * s, C.width * s);
  ctx.strokeRect(X(0), Y(LANE_Y0), C.paint_length * s, C.paint_width * s);
  ctx.beginPath();
  ctx.arc(X(C.paint_length), Y(C.width/2), 6 * s, 0, Math.PI * 2);
  ctx.stroke();
  const dy = C.width/2 - C.sideline, dx = C.straight - RIMX;
  const a = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(X(0), Y(C.sideline));
  ctx.lineTo(X(C.straight), Y(C.sideline));
  ctx.arc(X(RIMX), Y(RIMY), C.three * s, -a, a);
  ctx.lineTo(X(0), Y(C.width - C.sideline));
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(X(4), Y(RIMY - 3)); ctx.lineTo(X(4), Y(RIMY + 3));
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(X(RIMX), Y(RIMY), C.rim_radius * s, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(X(HALF), Y(RIMY), 6 * s, Math.PI/2, Math.PI*1.5);
  ctx.stroke();
}

function drawFrame(F, a, i){
  drawCourt(F);
  const {ctx, X, Y} = F;
  const css = getComputedStyle(document.body);
  const dukeC = css.getPropertyValue("--duke").trim() || "#0a5fbd";
  const oppC = css.getPropertyValue("--opp").trim() || "#b23b2c";
  const ballC = css.getPropertyValue("--ball").trim() || "#d97706";
  ctx.lineWidth = 1.5;
  const on = p => p[0] >= -0.5 && p[0] <= HALF + 0.5 && p[1] >= -0.5 && p[1] <= C.width + 0.5;
  (a.opp[i] || []).filter(on).forEach(p => {
    ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5, 0, Math.PI*2);
    ctx.strokeStyle = oppC; ctx.stroke();
  });
  (a.duke[i] || []).filter(on).forEach(p => {
    ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, Math.PI*2);
    ctx.fillStyle = dukeC; ctx.fill();
  });
  if (!(a.duke[i] || []).length && !(a.opp[i] || []).length){
    ctx.fillStyle = css.getPropertyValue("--muted").trim() || "#6d6862";
    ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("no tracking at this moment", X(HALF/2), Y(C.width/2));
    ctx.textAlign = "start";
  }
  const b = a.handler[i];
  if (b && on(b)){
    ctx.beginPath(); ctx.arc(X(b[0]), Y(b[1]), 3, 0, Math.PI*2);
    ctx.fillStyle = ballC; ctx.fill();
    ctx.strokeStyle = ballC; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(b[0]), Y(b[1]), 8.5, 0, Math.PI*2); ctx.stroke();
  }
}

function mountAnim(key){
  const a = DATA.anim[key];
  const box = el("div", "anim");
  if (!a){ box.appendChild(el("p", "empty", "No tracked positions for this possession.")); return box; }
  const cv = el("canvas");
  cv.setAttribute("role", "img");
  cv.setAttribute("aria-label", "Court animation for " + keyTitle(key));
  box.appendChild(cv);
  const row = el("div", "row");
  const play = el("button", "tiny", "▶");
  play.type = "button"; play.title = "Play / pause";
  const range = document.createElement("input");
  range.type = "range"; range.min = 0; range.max = a.duke.length - 1; range.value = 0;
  range.setAttribute("aria-label", "Scrub");
  const tlab = el("span", "t", "");
  row.appendChild(play); row.appendChild(range); row.appendChild(tlab);
  box.appendChild(row);
  const lg = el("div", "legend");
  lg.innerHTML = '<span><i style="background:var(--duke)"></i>Duke</span>' +
    '<span><i style="border:1.5px solid var(--opp)"></i>Opponent</span>' +
    '<span><i style="background:var(--ball)"></i>Ball</span>';
  box.appendChild(lg);

  let F = null, i = 0, raf = null, last = 0;
  function label(){ const t = -a.pre + i / a.fps; tlab.textContent = (t >= 0 ? "+" : "") + t.toFixed(1) + "s"; }
  function show(){
    if (!F) F = makeFrame(cv);
    drawFrame(F, a, i); range.value = i; label();
  }
  function step(ts){
    if (!last) last = ts;
    if (ts - last >= 1000 / a.fps){
      last = ts; i = (i + 1) % a.duke.length; show();
    }
    raf = requestAnimationFrame(step);
  }
  function stop(){ if (raf) cancelAnimationFrame(raf); raf = null; play.textContent = "▶"; }
  play.addEventListener("click", () => {
    if (raf) { stop(); } else { last = 0; play.textContent = "❚❚"; raf = requestAnimationFrame(step); }
  });
  range.addEventListener("input", () => { stop(); i = +range.value; show(); });
  box._draw = show;
  box._stop = stop;
  return box;
}

function drawVisible(root){
  root.querySelectorAll(".anim").forEach(n => { if (n._draw) n._draw(); });
}

/* ---------------------------------------------------------------- clusters tab */

function renderClusters(){
  const root = document.getElementById("pane-clusters");
  root.innerHTML = "";
  const p = el("p", "lede",
    "Name every cluster you recognise. Merge look-alikes into one name, discard the ones that " +
    "are noise. Only fit members shaped a centroid; assigned-only possessions were attached to " +
    "the nearest centroid afterwards, so they are the weaker evidence.");
  root.appendChild(p);
  const q = el("p", "lede");
  q.innerHTML = "k = <b>" + DATA.k + "</b> · silhouette " + fmt(DATA.silhouette, 3) +
    " · stability ARI " + fmt(DATA.stability_ari, 3) + " · " + DATA.n_fit + " fit, " +
    DATA.n_assigned_only + " assigned-only" +
    (DATA.small_corpus ? " · <b>small corpus</b>: treat the split as provisional" : "");
  root.appendChild(q);

  DATA.clusters.forEach(c => {
    const id = String(c.cluster);
    const card = el("div", "card");
    const head = el("div", "chead");
    head.appendChild(el("h2", null, "Cluster " + id));
    const bits = [c.n + " possessions"];
    if (c.n_fit != null) bits.push(c.n_fit + " fit · " + (c.n - c.n_fit) + " assigned-only");
    head.appendChild(el("span", "sub", bits.join(" · ")));
    const badge = el("span", "pill", "unnamed");
    badge.id = "badge-" + id;
    head.appendChild(badge);
    card.appendChild(head);

    const g = el("div", "grid2");
    const left = el("div");
    if (DATA.montages[id]){
      const img = el("img", "montage");
      img.src = DATA.montages[id]; img.loading = "lazy";
      img.alt = "Nine setup frames closest to the centre of cluster " + id;
      left.appendChild(img);
    } else {
      left.appendChild(el("p", "empty", "No montage on disk for this cluster."));
    }
    g.appendChild(left);

    const right = el("div");
    const dl = el("dl", "kv");
    dl.appendChild(el("dt", null, "By bucket"));
    dl.appendChild(el("dd", null, DATA.bucket_order
      .filter(b => c.n_by_bucket[b]).map(b => bucketLabel(b) + " " + c.n_by_bucket[b]).join(" · ")
      || "–"));
    dl.appendChild(el("dt", null, "By split"));
    dl.appendChild(el("dd", null, Object.keys(c.n_by_split)
      .map(k => k + " " + c.n_by_split[k]).join(" · ") || "–"));
    dl.appendChild(el("dt", null, "Top zones"));
    dl.appendChild(el("dd", null, c.top_zones.join(", ") || "–"));
    dl.appendChild(el("dt", null, "Handler path (setup → +4 s)"));
    const hp = el("dd", "path", c.handler_path.join("  →  ") || "–");
    dl.appendChild(hp);
    right.appendChild(dl);
    g.appendChild(right);
    card.appendChild(g);

    const ctr = el("div", "controls");
    const nameL = el("label", "f", "Set name");
    const name = document.createElement("input");
    name.type = "text"; name.setAttribute("list", "seed-names");
    name.placeholder = "e.g. Horns";
    name.value = (S.clusters[id] && S.clusters[id] !== "discard" &&
                  S.clusters[id].indexOf("merge:") !== 0) ? S.clusters[id] : "";
    nameL.appendChild(name);
    ctr.appendChild(nameL);

    const mergeL = el("label", "f", "Merge into");
    const merge = document.createElement("select");
    merge.appendChild(new Option("— no merge —", ""));
    DATA.clusters.forEach(o => {
      if (String(o.cluster) !== id) merge.appendChild(new Option("Cluster " + o.cluster, String(o.cluster)));
    });
    merge.value = (S.clusters[id] || "").indexOf("merge:") === 0 ? S.clusters[id].slice(6) : "";
    mergeL.appendChild(merge);
    ctr.appendChild(mergeL);

    const discL = el("label", "chk");
    const disc = document.createElement("input");
    disc.type = "checkbox"; disc.checked = S.clusters[id] === "discard";
    discL.appendChild(disc); discL.appendChild(document.createTextNode("Discard"));
    ctr.appendChild(discL);

    const res = el("div", "resolved");
    ctr.appendChild(res);
    card.appendChild(ctr);

    function sync(){
      if (disc.checked) S.clusters[id] = "discard";
      else if (merge.value) S.clusters[id] = "merge:" + merge.value;
      else if (name.value.trim()) S.clusters[id] = name.value.trim();
      else delete S.clusters[id];
      name.disabled = disc.checked || !!merge.value;
      merge.disabled = disc.checked;
      paint();
      save();
      refreshDependents();
    }
    function paint(){
      const r = resolveName(id);
      const v = S.clusters[id];
      badge.textContent = v === "discard" ? "discarded" : (r || "unnamed");
      badge.className = "pill" + (r ? " on" : "");
      res.innerHTML = v && v.indexOf("merge:") === 0
        ? "Resolves to " + (r ? "<b>" + esc(r) + "</b>" : "<b>nothing yet</b> (target is unnamed)")
        : "";
    }
    name.addEventListener("input", sync);
    merge.addEventListener("change", sync);
    disc.addEventListener("change", sync);
    paint();
    name.disabled = disc.checked || !!merge.value;
    merge.disabled = disc.checked;
    root.appendChild(card);
  });
}
function fmt(v, d){ return v == null ? "–" : Number(v).toFixed(d); }
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

/* ---------------------------------------------------------------- validation tab */

let validationBuilt = false;
function renderValidation(){
  const root = document.getElementById("pane-validation");
  if (validationBuilt){ refreshValidationOptions(); drawVisible(root); return; }
  root.innerHTML = "";
  const keys = DATA.validation;
  const splits = {};
  keys.forEach(k => { const s = metaOf(k).split || "?"; splits[s] = (splits[s]||0)+1; });
  const p = el("p", "lede",
    "Watch each possession and say what it is, without seeing what the model thinks. " +
    "The animation runs from one second before the setup frame to eight seconds after it.");
  root.appendChild(p);
  if (DATA.fallback_split){
    root.appendChild(el("p", "banner",
      "No held-out games yet — this sample comes from the TRAIN split, the same possessions the " +
      "clusters were fitted on. Numbers from it are not a fair test."));
  }
  const sub = el("p", "lede", keys.length + " possessions · " +
    Object.keys(splits).map(s => s + " " + splits[s]).join(", "));
  root.appendChild(sub);
  const prog = el("p", "lede"); prog.id = "val-progress";
  root.appendChild(prog);

  keys.forEach(key => {
    const m = metaOf(key);
    const card = el("div", "card");
    const head = el("div", "chead");
    head.appendChild(el("h3", null, keyTitle(key)));
    head.appendChild(el("span", "pill", m.split || "?"));
    const done = el("span", "pill"); done.id = "vdone-" + key;
    head.appendChild(done);
    card.appendChild(head);

    const g = el("div", "grid2");
    g.appendChild(mountAnim(key));
    const right = el("div");

    const cid = (DATA.assign[key] || {}).cluster;
    if (cid != null && DATA.montages[String(cid)]){
      const d = el("details", "ref");
      d.appendChild(el("summary", null, "Reference: the montage of the cluster this landed in (may bias you)"));
      const img = document.createElement("img");
      img.src = DATA.montages[String(cid)]; img.loading = "lazy";
      img.alt = "Setup montage of the cluster this possession was assigned to";
      d.appendChild(img);
      right.appendChild(d);
    }

    const ctr = el("div", "controls");
    const setL = el("label", "f", "Set");
    const sel = document.createElement("select");
    sel.dataset.role = "vset"; sel.dataset.key = key;
    setL.appendChild(sel);
    ctr.appendChild(setL);

    const fs = el("fieldset");
    fs.appendChild(el("legend", null, "Defence"));
    const rad = el("div", "radios");
    ["man", "zone", "unclear"].forEach(v => {
      const lab = el("label");
      const r = document.createElement("input");
      r.type = "radio"; r.name = "def-" + key; r.value = v;
      r.checked = S.defense[key] === v;
      r.addEventListener("change", () => { S.defense[key] = v; markDone(key); save(); });
      lab.appendChild(r); lab.appendChild(document.createTextNode(v));
      rad.appendChild(lab);
    });
    fs.appendChild(rad);
    ctr.appendChild(fs);
    right.appendChild(ctr);
    g.appendChild(right);
    card.appendChild(g);
    root.appendChild(card);
    sel.addEventListener("change", () => {
      if (sel.value) S.validation[key] = sel.value; else delete S.validation[key];
      markDone(key); save();
    });
  });
  validationBuilt = true;
  refreshValidationOptions();
  DATA.validation.forEach(markDone);
  drawVisible(root);
}

function refreshValidationOptions(){
  const names = namedSets();
  document.querySelectorAll('select[data-role="vset"]').forEach(sel => {
    const key = sel.dataset.key;
    const cur = S.validation[key] || "";
    sel.innerHTML = "";
    sel.appendChild(new Option("— not labelled —", ""));
    names.forEach(n => sel.appendChild(new Option(n, n)));
    ["other", "unclear"].forEach(n => sel.appendChild(new Option(n, n)));
    if (cur && !Array.prototype.some.call(sel.options, o => o.value === cur))
      sel.appendChild(new Option(cur + " (no longer a cluster name)", cur));
    sel.value = cur;
  });
  updateProgress();
}
function markDone(key){
  const n = document.getElementById("vdone-" + key);
  if (!n) return;
  const a = !!S.validation[key], b = !!S.defense[key];
  n.textContent = a && b ? "done" : (a || b ? "partial" : "todo");
  n.className = "pill" + (a && b ? " on" : "");
  updateProgress();
}
function updateProgress(){
  const p = document.getElementById("val-progress");
  if (!p) return;
  const keys = DATA.validation;
  const both = keys.filter(k => S.validation[k] && S.defense[k]).length;
  p.textContent = both + " of " + keys.length + " fully labelled (set and defence).";
}

/* ---------------------------------------------------------------- results tab */

function renderResults(){
  const root = document.getElementById("pane-results");
  root.innerHTML = "";
  root.appendChild(el("p", "lede",
    "Every clustered possession rolled up by the name you gave its cluster. Points per " +
    "possession come from the play-by-play. Cells under 8 possessions are greyed: that is " +
    "noise, not a finding."));

  const rows = {};
  const unnamed = [];
  Object.keys(DATA.assign).forEach(key => {
    const name = resolveName(DATA.assign[key].cluster);
    if (!name) return;
    const m = metaOf(key);
    const r = rows[name] || (rows[name] = {bucket:{}, def:{}, n:0, pts:0});
    const b = m.bucket || "unknown", d = (DATA.defense[key] || {}).defense || "unknown";
    const pts = m.points || 0;
    (r.bucket[b] || (r.bucket[b] = {n:0, pts:0}));
    r.bucket[b].n++; r.bucket[b].pts += pts;
    (r.def[d] || (r.def[d] = {n:0, pts:0}));
    r.def[d].n++; r.def[d].pts += pts;
    r.n++; r.pts += pts;
  });
  DATA.clusters.forEach(c => {
    const v = S.clusters[String(c.cluster)];
    if (!resolveName(c.cluster)) unnamed.push({c, v});
  });

  const names = Object.keys(rows).sort((a,b) => rows[b].n - rows[a].n || a.localeCompare(b));
  if (!names.length){
    root.appendChild(el("p", "empty", "Nothing named yet — name a cluster on the Clusters tab."));
  } else {
    root.appendChild(sectionTable("By bucket", DATA.bucket_order.map(b => [b, bucketLabel(b)]),
      names, rows, "bucket"));
    root.appendChild(sectionTable("By defence", [["man","Man"],["zone","Zone"],
      ["unknown","Unknown"]], names, rows, "def"));
  }

  const h = el("div", "card");
  h.appendChild(el("h2", null, "Clusters without a usable name"));
  if (!unnamed.length){
    h.appendChild(el("p", "empty", "None — every cluster resolves to a name."));
  } else {
    const ul = el("ul");
    unnamed.forEach(u => {
      const what = u.v === "discard" ? "discarded"
        : (u.v && u.v.indexOf("merge:") === 0 ? "merged into an unnamed cluster" : "unnamed");
      const li = el("li", null, "Cluster " + u.c.cluster + " — " + u.c.n + " possessions (" + what + ")");
      ul.appendChild(li);
    });
    h.appendChild(ul);
  }
  root.appendChild(h);
}

function sectionTable(title, cols, names, rows, field){
  const card = el("div", "card");
  card.appendChild(el("h2", null, title));
  card.appendChild(el("p", "lede", "count · points per possession"));
  const wrap = el("div", "tablewrap");
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  const tr = document.createElement("tr");
  tr.appendChild(el("th", null, "Set"));
  cols.forEach(c => tr.appendChild(el("th", null, c[1])));
  tr.appendChild(el("th", null, "All"));
  thead.appendChild(tr); t.appendChild(thead);
  const tb = document.createElement("tbody");
  const tot = {};
  names.forEach(n => {
    const r = document.createElement("tr");
    r.appendChild(el("td", null, n));
    cols.forEach(c => {
      const cell = rows[n][field][c[0]];
      r.appendChild(statCell(cell));
      if (cell){ const a = tot[c[0]] || (tot[c[0]] = {n:0, pts:0}); a.n += cell.n; a.pts += cell.pts; }
    });
    r.appendChild(statCell({n: rows[n].n, pts: rows[n].pts}));
    tb.appendChild(r);
  });
  const r = document.createElement("tr");
  r.className = "total";
  r.appendChild(el("td", null, "All named"));
  let gn = 0, gp = 0;
  cols.forEach(c => { r.appendChild(statCell(tot[c[0]])); if (tot[c[0]]){ gn += tot[c[0]].n; gp += tot[c[0]].pts; } });
  r.appendChild(statCell({n: gn, pts: gp}));
  tb.appendChild(r);
  t.appendChild(tb); wrap.appendChild(t); card.appendChild(wrap);
  return card;
}
function statCell(cell){
  const td = el("td");
  if (!cell || !cell.n){ td.className = "thin"; td.textContent = "–"; return td; }
  if (cell.n < 8) td.className = "thin";
  td.appendChild(document.createTextNode(String(cell.n)));
  td.appendChild(el("span", "ppp", (cell.pts / cell.n).toFixed(2)));
  return td;
}

/* ---------------------------------------------------------------- browse tab */

let browseBuilt = false;
function renderBrowse(){
  const root = document.getElementById("pane-browse");
  if (browseBuilt){ drawVisible(root); return; }
  root.innerHTML = "";
  root.appendChild(el("p", "lede", "Any possession in the corpus, with what the pipeline made of it."));
  const ctr = el("div", "controls");
  ctr.style.borderTop = "0"; ctr.style.paddingTop = "0";
  const gL = el("label", "f", "Game");
  const gs = document.createElement("select");
  gL.appendChild(gs); ctr.appendChild(gL);
  const rL = el("label", "f", "Possession");
  const rs = document.createElement("select");
  rs.style.minWidth = "240px";
  rL.appendChild(rs); ctr.appendChild(rL);
  root.appendChild(ctr);

  const byGame = {};
  Object.keys(DATA.meta).forEach(k => {
    const m = DATA.meta[k];
    (byGame[m.game] || (byGame[m.game] = [])).push(k);
  });
  const gids = Object.keys(byGame).sort((a,b) =>
    (metaOf(byGame[a][0]).date || "") .localeCompare(metaOf(byGame[b][0]).date || ""));
  gids.forEach(g => {
    const m = metaOf(byGame[g][0]);
    gs.appendChild(new Option((m.date ? m.date + " · " : "") + (m.opponent || g) +
      " (" + byGame[g].length + ")", g));
  });

  const card = el("div", "card");
  const body = el("div", "grid2");
  card.appendChild(body);
  root.appendChild(card);

  let currentAnim = null;
  function fillRecords(){
    rs.innerHTML = "";
    if (!gs.value) gs.value = gids[0];
    (byGame[gs.value] || []).sort((a,b) => {
      const x = metaOf(a), y = metaOf(b);
      return (x.period - y.period) || (x.index - y.index);
    }).forEach(k => {
      const m = metaOf(k);
      rs.appendChild(new Option("P" + m.period + " #" + m.index + " · " + bucketLabel(m.bucket) +
        " · " + clockStr(m.clock) + " · " + (m.outcome || "?"), k));
    });
    showRecord();
  }
  function showRecord(){
    const key = rs.value || (rs.options[0] && rs.options[0].value);
    if (!key) return;
    const m = metaOf(key);
    const a = DATA.assign[key] || {};
    const d = DATA.defense[key] || {};
    if (currentAnim && currentAnim._stop) currentAnim._stop();
    body.innerHTML = "";
    currentAnim = mountAnim(key);
    body.appendChild(currentAnim);
    const right = el("div");
    right.appendChild(el("h3", null, keyTitle(key)));
    const dl = el("dl", "kv");
    function kv(k, v){ dl.appendChild(el("dt", null, k)); dl.appendChild(el("dd", null, v)); }
    kv("Cluster", a.cluster == null ? "–" : "Cluster " + a.cluster +
      (a.assigned_only ? " (assigned-only)" : " (fit member)") +
      (resolveName(a.cluster) ? " — " + resolveName(a.cluster) : ""));
    kv("Defence", (d.defense || "–") + (d.confidence != null ? "  confidence " + Number(d.confidence).toFixed(2) : ""));
    kv("Bucket", bucketLabel(m.bucket) + (m.no_setup ? " · no setup frame" : ""));
    kv("Outcome", (m.outcome || "–") + " · " + (m.points || 0) + " pts");
    kv("Clock / start", clockStr(m.clock) + " · " + (m.start_type || "?") +
      (m.full_court ? " · full court" : ""));
    kv("Split", m.split || "?");
    kv("Key", key);
    right.appendChild(dl);
    body.appendChild(right);
    drawVisible(body);
  }
  gs.addEventListener("change", fillRecords);
  rs.addEventListener("change", showRecord);
  fillRecords();
  browseBuilt = true;
}

/* ---------------------------------------------------------------- export tab */

function renderExport(){
  const root = document.getElementById("pane-export");
  root.innerHTML = "";
  root.appendChild(el("p", "lede",
    "Your decisions live in this browser's localStorage. Copy the JSON below into " +
    "data/plays/labels.json to keep them with the repository."));
  const card = el("div", "card");
  const ctr = el("div", "controls");
  ctr.style.borderTop = "0"; ctr.style.paddingTop = "0";
  const copy = el("button", "btn", "Copy to clipboard"); copy.type = "button";
  const reset = el("button", "btn ghost", "Reset all decisions"); reset.type = "button";
  const note = el("span", "status", "");
  ctr.appendChild(copy); ctr.appendChild(reset); ctr.appendChild(note);
  card.appendChild(ctr);
  const ta = document.createElement("textarea");
  ta.readOnly = true; ta.spellcheck = false;
  ta.setAttribute("aria-label", "Exported labels JSON");
  ta.value = JSON.stringify(S, null, 2) + "\n";
  card.appendChild(ta);
  root.appendChild(card);
  copy.addEventListener("click", () => {
    ta.select();
    const done = () => { note.textContent = "Copied."; setTimeout(()=> note.textContent = "", 2500); };
    if (navigator.clipboard && navigator.clipboard.writeText)
      navigator.clipboard.writeText(ta.value).then(done, () => { document.execCommand("copy"); done(); });
    else { document.execCommand("copy"); done(); }
  });
  reset.addEventListener("click", () => {
    if (!confirm("Discard every name and label stored in this browser?")) return;
    S = blankState();
    try { localStorage.removeItem(LS_KEY); } catch(e){}
    validationBuilt = false; browseBuilt = false;
    renderClusters(); renderExport(); setStatus(countLabelled());
  });
}

function refreshDependents(){
  if (validationBuilt) refreshValidationOptions();
  const r = document.getElementById("pane-results");
  if (!r.hidden) renderResults();
}

/* ---------------------------------------------------------------- tabs */

const RENDER = {clusters: renderClusters, validation: renderValidation, results: renderResults,
                browse: renderBrowse, export: renderExport};
function show(name){
  document.querySelectorAll("#tabs button").forEach(b =>
    b.setAttribute("aria-selected", String(b.dataset.pane === name)));
  ["clusters","validation","results","browse","export"].forEach(n =>
    document.getElementById("pane-" + n).hidden = n !== name);
  RENDER[name]();
  try { location.hash = name; } catch(e){}
}
document.getElementById("tabs").addEventListener("click", e => {
  const b = e.target.closest("button[data-pane]");
  if (b) show(b.dataset.pane);
});
document.getElementById("go-export").addEventListener("click", () => show("export"));

(function init(){
  const dl = document.getElementById("seed-names");
  DATA.seeds.forEach(s => dl.appendChild(new Option(s, s)));
  const n = Object.keys(DATA.meta).length;
  document.getElementById("subtitle").textContent =
    n + " possessions · " + DATA.clusters.length + " clusters · " +
    DATA.validation.length + " in the validation sample" +
    (DATA.generated ? " · built " + DATA.generated : "");
  setStatus(countLabelled());
  const start = (location.hash || "").replace("#", "");
  show(RENDER[start] ? start : "clusters");
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- main


def build(plays_dir: Path, out: Path, fps: float) -> tuple[str, list[dict], bool]:
    index = json.loads((plays_dir / "features_index.json").read_text())
    rows = index["rows"]
    clusters = json.loads((plays_dir / "clusters.json").read_text())
    defense = json.loads((plays_dir / "defense.json").read_text())
    lab = L.load(plays_dir / "labels.json")
    records = load_records(plays_dir / "halfcourt.jsonl")
    opponents = opponent_names(sorted({r["game_id"] for r in rows}))
    dates = {g.espn_id: g.date for g in games.load_registry()}

    animations: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for r in rows:
        key = f"{r['game_id']}:{r['period']}:{r['index']}"
        rec = records.get(key)
        if rec is None:
            continue
        anim = sample_animation(rec, fps=fps)
        if anim is not None:
            animations[key] = anim
        meta[key] = {
            "game": r["game_id"], "opponent": opponents.get(r["game_id"], r["game_id"]),
            "date": dates.get(r["game_id"], ""), "period": r["period"], "index": r["index"],
            "bucket": r["bucket"], "split": r["split"], "no_setup": bool(r["no_setup"]),
            "points": rec.points, "outcome": rec.outcome, "start_type": rec.start_type,
            "full_court": bool(rec.full_court), "clock": rec.clock_start,
        }

    sample, fallback = validation_rows(rows)
    validation_keys = [f"{r['game_id']}:{r['period']}:{r['index']}" for r in sample]
    montages = encode_montages(plays_dir / "montages",
                               [c["cluster"] for c in clusters.get("clusters", [])])
    html = render_page(clusters, defense, animations, meta, montages,
                       {"clusters": lab.clusters, "validation": lab.validation,
                        "defense": lab.defense, "version": lab.version},
                       validation_keys, fallback_split=fallback)
    return html, sample, fallback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plays", default="data/plays", type=Path)
    ap.add_argument("--out", default="data/plays/page/index.html", type=Path)
    ap.add_argument("--fps", default=FPS, type=float)
    args = ap.parse_args(argv)

    html, sample, fallback = build(args.plays, args.out, args.fps)
    if len(html.encode()) > SIZE_LIMIT and args.fps > FALLBACK_FPS:
        print(f"{len(html) / 1e6:.1f} MB is over the budget; resampling at {FALLBACK_FPS} fps")
        html, sample, fallback = build(args.plays, args.out, FALLBACK_FPS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    size = args.out.stat().st_size
    by_bucket: dict[str, int] = defaultdict(int)
    for r in sample:
        by_bucket[r["bucket"]] += 1
    split = "train (no held-out games yet)" if fallback else "test/ncaa"
    print(f"wrote {args.out} ({size / 1e6:.2f} MB)")
    print(f"validation sample: {len(sample)} from {split} — "
          + ", ".join(f"{b} {n}" for b, n in sorted(by_bucket.items())))
    if size > 16 * 1024 * 1024:
        print("WARNING: over the 16 MB budget")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
