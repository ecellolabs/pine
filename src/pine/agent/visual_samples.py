"""Build a self-contained ``visual_samples.html`` that visualises every step of
the first ``max_runs`` runs in an agent-runs folder (works from ``file://``;
all data and thumbnails are inlined, so it is meant for a sample of runs, not a
whole benchmark).  Run folders are discovered from ``manifest.json`` when
present, otherwise every sub-folder with ``00_input/question.json`` is used."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any


def thumb(path: Path, width: int = 520) -> str:
    from PIL import Image

    im = Image.open(path)
    im.thumbnail((width, width * 2))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rj(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


DEFAULT_MAX_RUNS = 20


def discover_runs(samples_dir: Path) -> list[Path]:
    manifest = samples_dir / "manifest.json"
    if manifest.is_file():
        entries = rj(manifest).get("runs", [])
        dirs = [samples_dir / e["run_id"] for e in entries]
        return [d for d in dirs if (d / "00_input" / "question.json").is_file()]
    return sorted(
        d for d in samples_dir.iterdir() if (d / "00_input" / "question.json").is_file()
    )


def collect(d: Path) -> dict[str, Any]:
    q = rj(d / "00_input" / "question.json")
    run = {
        "run_id": q.get("run_id", d.name),
        "doc_id": q.get("doc_id"),
        "question": q.get("question"),
        "why_chosen": q.get("why_chosen", ""),
    }
    parser = rj(d / "01_parser" / "page_stats.json")
    pages = {}
    for f in sorted((d / "01_parser" / "pages").glob("page_*.md")):
        pages[int(f.stem.split("_")[1])] = f.read_text(encoding="utf-8")
    index = rj(d / "02_index" / "index.json")
    outline = (
        (d / "02_index" / "outline.md").read_text(encoding="utf-8")
        if (d / "02_index" / "outline.md").exists()
        else ""
    )
    plans = [rj(f) for f in sorted((d / "03_planner").glob("plan_round*.json"))]
    traces = {
        f.stem: jsonl(f)
        for f in sorted((d / "04_orchestrator").glob("trace_round*.jsonl"))
    }
    anomalies = {
        f.stem: rj(f)
        for f in sorted((d / "04_orchestrator").glob("anomalies_round*.json"))
    }
    finals = {
        f.stem: rj(f) for f in sorted((d / "04_orchestrator").glob("final_round*.json"))
    }
    for f in sorted((d / "04_orchestrator").glob("ledger_round*.json")):
        key = f.stem.replace("ledger_", "final_")
        if key in finals:
            finals[key]["ledger"] = rj(f)
    tool_results = {
        f.stem: f.read_text(encoding="utf-8")
        for f in sorted((d / "04_orchestrator" / "tool_results").glob("*.txt"))
    }
    images = {
        f.name: thumb(f)
        for f in sorted((d / "04_orchestrator" / "images_viewed").glob("*.jpg"))
    }
    gold_imgs = {
        f.name: thumb(f)
        for f in sorted((d / "00_input").glob("gold_evidence_page_*.jpg"))
    }
    verifs = [
        rj(f)
        for f in sorted((d / "05_evidence_verifier").glob("verification_round*.json"))
    ]
    result = rj(d / "06_evaluation" / "result.json")
    cost = rj(d / "cost.json")
    logs = {
        name: (d / name).read_text(encoding="utf-8")
        for name in ["run.log"]
        if (d / name).exists()
    }
    step_logs = {}
    for step, fname in [
        ("01_parser", "parser.log"),
        ("02_index", "index.log"),
        ("03_planner", "planner.log"),
        ("04_orchestrator", "orchestrator.log"),
        ("05_evidence_verifier", "verifier.log"),
        ("06_evaluation", "eval.log"),
    ]:
        p = d / step / fname
        if p.exists():
            step_logs[step] = p.read_text(encoding="utf-8")
    calls = jsonl(d / "llm_calls.jsonl")
    return {
        "meta": run,
        "question": q,
        "parser": parser,
        "pages": pages,
        "index": index,
        "outline": outline,
        "plans": plans,
        "traces": traces,
        "anomalies": anomalies,
        "finals": finals,
        "tool_results": tool_results,
        "images": images,
        "gold_imgs": gold_imgs,
        "verifications": verifs,
        "result": result,
        "cost": cost,
        "logs": logs,
        "step_logs": step_logs,
        "llm_calls": calls,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocQA Visual Samples</title>
<style>
:root{--bg:#f6f7f9;--panel:#ffffff;--ink:#1b1f24;--muted:#5b6470;--line:#d9dee5;--accent:#2457c5;--ok:#1d8a4b;--warn:#b7791f;--bad:#c0392b;--code:#eef1f5;--chip:#e8edf5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1216;--panel:#171b21;--ink:#e6e9ee;--muted:#9aa4b2;--line:#2a313b;--accent:#7aa2ff;--ok:#4ccb7f;--warn:#e0a84a;--bad:#ff7b6b;--code:#11151a;--chip:#232a34}}
:root[data-theme="dark"]{--bg:#0f1216;--panel:#171b21;--ink:#e6e9ee;--muted:#9aa4b2;--line:#2a313b;--accent:#7aa2ff;--ok:#4ccb7f;--warn:#e0a84a;--bad:#ff7b6b;--code:#11151a;--chip:#232a34}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
header h1{font-size:16px;margin:0 12px 0 0}
.tab{border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:6px 12px;border-radius:8px;cursor:pointer;font-size:13px}
.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:1200px;margin:0 auto;padding:16px}
.run{display:none}.run.active{display:block}
.hero{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.hero h2{margin:0 0 6px;font-size:18px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px 16px;margin-top:8px}
.kv div b{display:block;color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.pipe{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0}
.pipe a{text-decoration:none;color:var(--ink);background:var(--chip);border:1px solid var(--line);border-radius:20px;padding:4px 10px;font-size:12px}
.pipe a:hover{border-color:var(--accent)}
.step{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:12px 0;overflow:hidden}
.step>summary{cursor:pointer;padding:12px 16px;font-weight:700;font-size:15px;list-style:none;display:flex;align-items:center;gap:10px}
.step>summary::-webkit-details-marker{display:none}
.step>summary .n{background:var(--accent);color:#fff;border-radius:6px;padding:0 7px;font-size:12px}
.step>summary .st{margin-left:auto;font-weight:500;color:var(--muted);font-size:12px}
.body{padding:0 16px 16px;border-top:1px solid var(--line)}
.badge{display:inline-block;border-radius:6px;padding:1px 8px;font-size:12px;font-weight:600;color:#fff}
.ok{background:var(--ok)}.warn{background:var(--warn)}.bad{background:var(--bad)}.neutral{background:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px}
pre,code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;max-height:420px;white-space:pre-wrap;word-break:break-word;margin:6px 0}
details.inner{border:1px solid var(--line);border-radius:8px;margin:6px 0;background:var(--bg)}
details.inner>summary{cursor:pointer;padding:7px 10px;font-weight:600;font-size:13px}
details.inner>.in{padding:0 10px 10px}
.bars{display:flex;align-items:flex-end;gap:3px;height:120px;border-bottom:1px solid var(--line);padding:4px 0;overflow-x:auto}
.bar{flex:1 0 14px;display:flex;flex-direction:column;justify-content:flex-end;height:100%;position:relative;cursor:pointer}
.bar i{display:block;width:100%;background:var(--accent);opacity:.85;border-radius:2px 2px 0 0}
.bar i.nat{background:var(--muted);opacity:.4;position:absolute;left:0;right:0;bottom:0;z-index:0}
.bar i.doc{position:relative;z-index:1}
.bar.flag i.doc{background:var(--bad)}
.bar.gold::after{content:"★";position:absolute;top:-16px;left:0;right:0;text-align:center;color:var(--warn);font-size:12px}
.bar span{position:absolute;bottom:-18px;left:0;right:0;text-align:center;font-size:10px;color:var(--muted)}
.legend{font-size:12px;color:var(--muted);margin:22px 0 6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:middle}
.tree ul{list-style:none;padding-left:18px;margin:2px 0}
.tree li{margin:2px 0}
.tree .sec{font-weight:600}
.tree .pg{color:var(--muted)}
.tree .pg.gold{color:var(--warn);font-weight:600}
.tl{border-left:3px solid var(--line);margin:8px 0 8px 8px;padding-left:14px}
.ev{position:relative;margin:0 0 12px}
.ev::before{content:"";position:absolute;left:-21px;top:6px;width:11px;height:11px;border-radius:50%;background:var(--accent)}
.ev.final::before{background:var(--ok)}.ev.err::before{background:var(--bad)}
.ev .t{font-weight:700}
.ev .args{color:var(--muted);font-size:12px}
.imgs{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}
.imgs figure{margin:0;width:240px}
.imgs img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.imgs figcaption{font-size:11px;color:var(--muted);margin-top:3px}
.pager{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}
.pager button{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:5px;padding:2px 7px;font-size:12px;cursor:pointer}
.pager button.sel{background:var(--accent);color:#fff;border-color:var(--accent)}
.pager button.flag{border-color:var(--bad)}
.pager button.gold{border-color:var(--warn);font-weight:700}
.small{font-size:12px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:800px){.grid2{grid-template-columns:1fr}.kv{grid-template-columns:1fr 1fr}}
.callrow td{font-size:12px}
.sum{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0}
.sum div{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.sum div b{display:block;font-size:11px;color:var(--muted);text-transform:uppercase}
.sum div span{font-size:16px;font-weight:700}
</style>
</head>
<body>
<header><h1>DocQA visual samples</h1><div id="tabs"></div><button class="tab" id="theme" style="margin-left:auto">◐ theme</button></header>
<main id="main"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pj = o => esc(JSON.stringify(o, null, 2));
const fmt$ = v => '$' + Number(v || 0).toFixed(4);
function badge(txt, cls){return `<span class="badge ${cls}">${esc(txt)}</span>`}
function inner(title, html, open=false){return `<details class="inner"${open?' open':''}><summary>${title}</summary><div class="in">${html}</div></details>`}

function renderRun(r, i){
  const q=r.question, res=r.result, cost=r.cost||{}, gold=q.gold;
  const scoreCls = res && res.correct ? 'ok':'bad';
  const goldPages = new Set(gold.evidence_pages||[]);
  const lastV = r.verifications[r.verifications.length-1]||{};
  const stepsNav = ['0 Input','1 Parser','2 Index','3 Planner','4 Orchestrator','5 Verifier','6 Evaluation','Model calls','Logs'].map((s,k)=>`<a href="#r${i}s${k}">${s}</a>`).join('');
  let h = `<section class="run${i===0?' active':''}" id="run${i}">
  <div class="hero"><h2>${esc(r.meta.run_id)} <span class="small">· ${esc(r.meta.doc_id)} · ${esc(gold.doc_type)} · ${r.index?r.index.n_pages:'?'} pages</span></h2>
  <div style="font-size:15px"><b>Q:</b> ${esc(q.question)}</div>
  <div class="kv">
    <div><b>Gold answer</b>${esc(gold.answer)} <span class="small">(${esc(gold.answer_format)}; pages ${esc(JSON.stringify(gold.evidence_pages))}; ${esc(gold.evidence_sources)})</span></div>
    <div><b>System answer</b>${esc(res?res.raw_system_answer:'-')} ${res?badge((res.score_official_rule).toFixed(2)+' '+(res.correct?'CORRECT':'WRONG'), scoreCls):''}</div>
    <div><b>Agent status / verifier</b>${esc(res?res.agent_status:'')} / ${esc(res?res.verifier_decision:'')}</div>
    <div><b>Cited pages (gold hit)</b>${esc(JSON.stringify(res?res.cited_pages:[]))} (${esc(res?res.cited_page_hit:'')}) · visited ${esc(JSON.stringify(res?res.pages_visited:[]))}</div>
    <div><b>Cost</b>${fmt$(cost.cost_usd)} · ${cost.llm_calls} calls · ${cost.prompt_tokens}+${cost.completion_tokens} tokens</div>
    <div><b>Wall time</b>parse ${r.parser?r.parser.elapsed_s:'?'}s · index ${res?res.timings_s['02_index']:'?'}s · agent ${res?res.timings_s['03-05_agent_loop']:'?'}s · ${res?res.tool_calls_total:'?'} tool calls / ${res?res.rounds:'?'} round(s)</div>
    <div><b>Models</b>${esc(q.text_model)} (text) · ${esc(q.vision_model)} (vision)</div>
  </div>
  <div class="small" style="margin-top:8px"><b>Why chosen:</b> ${esc(r.meta.why_chosen)}</div>
  <div class="pipe">${stepsNav}</div></div>`;

  // ---- 0 input
  const goldImgs = Object.entries(r.gold_imgs).map(([n,src])=>`<figure><img src="${src}"><figcaption>${esc(n)} (gold evidence)</figcaption></figure>`).join('');
  h += step(i,0,'Input', `${goldPages.size} gold page(s)`, `
    <div class="imgs">${goldImgs||'<span class="small">No gold evidence page (question is unanswerable).</span>'}</div>
    ${inner('question.json', `<pre>${pj(q)}</pre>`)}`);

  // ---- 1 parser
  const ps=r.parser; let parserHtml='';
  if(ps){
    const mx=Math.max(1,...ps.page_stats.map(s=>Math.max(s.docling_chars,s.pymupdf_chars)));
    const bars=ps.page_stats.map(s=>`<div class="bar${s.flag?' flag':''}${goldPages.has(s.page)?' gold':''}" title="p${s.page}: docling ${s.docling_chars} chars, native ${s.pymupdf_chars} chars, tables ${s.tables}, pictures ${s.pictures}${s.flag?'\n'+s.flag:''}" onclick="showPage(${i},${s.page})"><i class="nat" style="height:${100*s.pymupdf_chars/mx}%"></i><i class="doc" style="height:${100*s.docling_chars/mx}%"></i><span>${s.page}</span></div>`).join('');
    const flagged=ps.page_stats.filter(s=>s.flag);
    parserHtml=`<div class="sum"><div><b>Docling status</b><span>${esc(String(ps.docling_status).replace('ConversionStatus.',''))}</span></div><div><b>Time</b><span>${ps.elapsed_s}s</span> <span class="small">(${(ps.elapsed_s/ps.pages).toFixed(2)} s/page)</span></div><div><b>OCR</b><span class="small">${esc(ps.ocr_engine)}</span></div><div><b>Tables / pictures</b><span>${ps.total_tables} / ${ps.total_pictures}</span></div><div><b>Pages flagged</b><span>${ps.pages_flagged}/${ps.pages}</span></div><div><b>Chars docling / native</b><span>${ps.total_docling_chars} / ${ps.total_pymupdf_chars}</span></div></div>
    <div class="bars">${bars}</div><div class="legend"><i style="background:var(--accent)"></i>Docling markdown chars <i style="background:var(--muted);opacity:.4"></i>native (PyMuPDF) chars <i style="background:var(--bad)"></i>flagged page · ★ gold evidence page · click a bar to read the page</div>
    ${flagged.length?inner(`Flagged pages (${flagged.length})`,`<table><tr><th>page</th><th>docling chars</th><th>native chars</th><th>flag</th></tr>${flagged.map(s=>`<tr><td>${s.page}</td><td>${s.docling_chars}</td><td>${s.pymupdf_chars}</td><td>${esc(s.flag)}</td></tr>`).join('')}</table>`,true):''}
    <div class="pager" id="pager${i}">${ps.page_stats.map(s=>`<button class="${s.flag?'flag ':''}${goldPages.has(s.page)?'gold':''}" onclick="showPage(${i},${s.page})">${s.page}</button>`).join('')}</div>
    <pre id="pageview${i}" class="small">Click a page number to see its Docling Markdown.</pre>`;
  } else parserHtml='<span class="small">parser output missing</span>';
  h += step(i,1,'Parser — Docling + RapidOCR', ps?`${ps.pages_flagged}/${ps.pages} pages flagged`:'', parserHtml);

  // ---- 2 index
  const ix=r.index; let ixHtml='';
  if(ix){
    const tree = nodes => `<ul>${nodes.map(s=>`<li><span class="sec">[${esc(s.id)}] ${esc(s.title)}</span> <span class="small">p${s.start_page}–${s.end_page}</span>${s.subsections&&s.subsections.length?tree(s.subsections):`<ul>${range(s.start_page,s.end_page).map(p=>{const pInfo=ix.pages[p]||ix.pages[String(p)]||{}; const vList=[...(pInfo.figures||[]),...(pInfo.tables||[])]; const vTag=vList.length?` [${esc(vList.join('; '))}]`: (pInfo.has_visual_elements?' [Visuals]':''); return `<li class="pg${goldPages.has(p)?' gold':''}">p${p}: ${esc(pInfo.title||'Page '+p)} <span class="small">[${esc(pInfo.source||'text')}]${vTag}</span>${goldPages.has(p)?' ★':''}</li>`;}).join('')}</ul>`}</li>`).join('')}</ul>`;
    const attempts=(ix.hierarchy_attempts||[]).map(a=>`<li>attempt ${a.attempt}: ${a.n_sections!==undefined?a.n_sections+' sections, ':''}${a.problems.length?badge(a.problems.length+' problem(s)','bad')+' '+esc(a.problems.join(' · ')):badge('valid','ok')}</li>`).join('');
    ixHtml=`<div class="sum"><div><b>Top-level sections</b><span>${ix.sections.length}</span></div><div><b>Hierarchy</b><span>${ix.hierarchy_fallback_used?badge('REPAIRED (fallback)','bad'):badge('valid','ok')}</span></div><div><b>Pages summarised from image</b><span>${ix.pages_summarised_from_image}</span></div><div><b>Summary failures</b><span>${(ix.page_summary_failures||[]).length}</span></div><div><b>Index cost</b><span>${fmt$((cost.index_build_share||{}).cost_usd)}</span> <span class="small">${(cost.index_build_share||{}).llm_calls} calls</span></div></div>
    ${inner('Hierarchy build attempts (LLM output validation)',`<ul>${attempts}</ul>`,true)}
    ${inner('Document tree (click to expand) — ★ gold page','<div class="tree">'+tree(ix.sections)+'</div>',true)}
    ${inner('Page summaries (LLM-generated node content)',`<table><tr><th>p</th><th>src</th><th>title</th><th>summary</th><th>keywords</th></tr>${Object.values(ix.pages).map(p=>`<tr${goldPages.has(p.page)?' style="background:var(--chip)"':''}><td>${p.page}</td><td>${p.source}</td><td>${esc(p.title)}</td><td>${esc(p.summary)}</td><td class="small">${esc((p.keywords||[]).join(', '))}</td></tr>`).join('')}</table>`)}
    ${inner('outline.md',`<pre>${esc(r.outline)}</pre>`)}`;
  }
  h += step(i,2,'Index — PageIndex-style hierarchy', ix?(ix.hierarchy_fallback_used?'fallback used':'valid'):'', ixHtml);

  // ---- 3 planner
  h += step(i,3,'Planner', `${r.plans.length} round(s)`, r.plans.map(p=>`<div class="sum"><div><b>Round</b><span>${p.round}</span></div><div><b>Question type</b><span>${esc(p.question_type)}</span></div><div><b>Expected format</b><span>${esc(p.expected_answer_format)}</span></div><div><b>Needs visual</b><span>${esc(p.needs_visual_inspection)}</span></div><div><b>Candidate sections</b><span>${esc((p.candidate_sections||[]).join(', '))}</span></div></div>
    <b>Sub-goals</b><ol>${(p.sub_goals||[]).map(s=>`<li>${esc(s)}</li>`).join('')}</ol><b>Search queries</b> <span>${(p.search_queries||[]).map(s=>`<code>${esc(s)}</code>`).join(' ')}</span>${p.gaps_in&&p.gaps_in.length?`<div><b>Gaps from verifier:</b> ${esc(p.gaps_in.join('; '))}</div>`:''}${p._error?badge('PLANNER ERROR: '+p._error,'bad'):''}
    ${inner('plan JSON',`<pre>${pj(p)}</pre>`)}`).join('<hr>'));

  // ---- 4 orchestrator
  let orHtml='';
  for(const [name,ev] of Object.entries(r.traces)){
    const round=name.replace('trace_','');
    const an=r.anomalies['anomalies_'+round]||[];
    const fin=r.finals['final_'+round]||{};
    const start=ev.find(e=>e.event==='start')||{};
    const items=ev.filter(e=>e.event!=='start').map(e=>{
      if(e.event==='assistant'){
        const tcs=(e.tool_calls||[]).map(tc=>`<div><span class="t">→ ${esc(tc.function.name)}</span> <span class="args">${esc(tc.function.arguments)}</span></div>`).join('');
        return `<div class="ev"><div class="small">turn ${e.turn} · assistant</div>${e.content?`<div><i>${esc(e.content)}</i></div>`:''}${tcs||badge('NO TOOL CALL','bad')}</div>`;
      }
      if(e.event==='tool_result'){
        const key=`${round}_call${String(e.turn).padStart(2,'0')}_${e.tool}`;
        const full=r.tool_results[key];
        const isImg=e.tool==='inspect_page_image';
        const imgName=isImg?Object.keys(r.images).find(n=>n.startsWith(round+'_page'+String(e.args.page).padStart(3,'0'))&&true):null;
        const err=String(e.result).startsWith('ERROR');
        return `<div class="ev${err?' err':''}"><div class="small">turn ${e.turn} · tool result · <b>${esc(e.tool)}</b></div>${inner(`${esc(e.tool)}(${esc(JSON.stringify(e.args))})`,`<pre>${esc(full!==undefined?full.replace(/^ARGS:.*\n\n/,''):e.result)}</pre>${isImg?imgFor(r,round,e.args.page,e.turn):''}`, e.tool==='inspect_page_image'||e.tool==='record_evidence')}</div>`;
      }
      if(e.event==='final') return `<div class="ev final"><div class="t">final_answer → ${esc(e.status)}: <code>${esc(e.answer)}</code> · pages ${esc(JSON.stringify(e.evidence_pages))}</div><div class="small">${esc(e.reasoning)}</div></div>`;
      return '';
    }).join('');
    orHtml+=`<h4 style="margin:10px 0 4px">${esc(round)} <span class="small">· ${fin.tool_calls_used} tool calls · pages read ${esc(JSON.stringify(fin.pages_read))} · ${an.length?badge(an.length+' anomal'+(an.length>1?'ies':'y'),'bad'):badge('0 anomalies','ok')}</span></h4>
    ${an.length?inner('Anomalies (tool-calling failures)',`<pre>${pj(an)}</pre>`,true):''}
    <div class="tl">${items}</div>
    ${inner('Evidence ledger (record_evidence calls)',`<pre>${pj(fin.ledger||[])}</pre>`)}
    ${inner('System prompt + user message sent to the orchestrator',`<pre>${esc(start.system)}\n\n--- user ---\n${esc(start.user)}</pre>`)}`;
  }
  const allImgs=Object.entries(r.images).map(([n,src])=>`<figure><img src="${src}"><figcaption>${esc(n)}</figcaption></figure>`).join('');
  if(allImgs) orHtml+=inner('All images sent to the vision model (exact JPEGs)',`<div class="imgs">${allImgs}</div>`);
  h += step(i,4,'Orchestrator — tool-calling navigation', `${res?res.tool_calls_total:'?'} tool calls`, orHtml);

  // ---- 5 verifier
  h += step(i,5,'Evidence ledger + Verifier', esc(lastV.decision||''), r.verifications.map(v=>`<div class="sum"><div><b>Round</b><span>${v.round}</span></div><div><b>Status in</b><span>${esc(v.status_in)}</span></div><div><b>LLM verdict</b><span>${esc((v.llm_verdict||{}).verdict)}</span></div><div><b>Quote verified</b><span>${v.any_quote_verified?badge('yes','ok'):badge('no','bad')}</span></div><div><b>Decision</b><span>${badge(v.decision, v.decision==='ACCEPT'?'ok':(v.decision.startsWith('ACCEPT')?'warn':'bad'))}</span></div><div><b>Final answer</b><span>${esc(v.final_answer)}</span></div></div>
    <div><b>Verifier explanation:</b> ${esc((v.llm_verdict||{}).explanation)}</div><div><b>Gaps:</b> ${esc((v.gaps||[]).join('; ')||'none')}</div>
    <table><tr><th>#</th><th>page</th><th>candidate answer</th><th>quote</th><th>in page text (sim)</th><th>in image inspection (sim)</th></tr>${(v.quote_checks||[]).map(c=>`<tr><td>${c.entry}</td><td>${c.page}</td><td>${esc(c.candidate_answer)}</td><td>${esc(c.quote)}</td><td>${c.found_in_page_text?badge('found','ok'):badge('not found','bad')} ${c.sim_page_text}</td><td>${c.found_in_image_inspection?badge('found','ok'):badge('not found','neutral')} ${c.sim_image_inspection}</td></tr>`).join('')||'<tr><td colspan=6 class="small">no ledger entries recorded by the agent</td></tr>'}</table>
    ${inner('verification JSON',`<pre>${pj(v)}</pre>`)}`).join('<hr>'));

  // ---- 6 eval
  h += step(i,6,'Evaluation — official MMLongBench-Doc rules', res?badge(res.score_official_rule.toFixed(2),scoreCls):'', res?`<div class="sum"><div><b>Gold</b><span>${esc(res.gold_answer)}</span> <span class="small">${esc(res.gold_answer_format)}</span></div><div><b>Extracted answer</b><span>${esc(res.extracted_answer)}</span></div><div><b>Score</b><span>${res.score_official_rule.toFixed(2)}</span></div><div><b>Cited page hit</b><span>${esc(res.cited_page_hit)}</span></div><div><b>Gold page visited</b><span>${esc(res.gold_page_visited_by_agent)}</span></div></div><div class="small">${esc(res.scorer_note)}</div>${inner('result.json',`<pre>${pj(res)}</pre>`)}`:'');

  // ---- model calls
  const calls=r.llm_calls;
  const byStep={}; calls.forEach(c=>{const s=byStep[c.step]||(byStep[c.step]={n:0,pt:0,ct:0,cost:0,lat:0}); s.n++; s.pt+=c.usage.prompt_tokens||0; s.ct+=c.usage.completion_tokens||0; s.cost+=Number(c.usage.cost||0); s.lat+=c.latency_s||0;});
  const stepTable=`<table><tr><th>step</th><th>calls</th><th>prompt tok</th><th>completion tok</th><th>cost</th><th>latency</th></tr>${Object.entries(byStep).map(([s,v])=>`<tr><td>${esc(s)}</td><td>${v.n}</td><td>${v.pt}</td><td>${v.ct}</td><td>${fmt$(v.cost)}</td><td>${v.lat.toFixed(1)}s</td></tr>`).join('')}</table>`;
  const callRows=calls.map((c,k)=>`<tr class="callrow"><td>${k+1}</td><td>${esc(c.step)}</td><td>${esc(c.purpose)}</td><td>${esc(c.model.split('/')[1])}</td><td>${c.usage.prompt_tokens||''}/${c.usage.completion_tokens||''}</td><td>${fmt$(c.usage.cost)}</td><td>${c.latency_s}s</td><td>${c.error?badge('error','bad'):esc(c.finish_reason)}</td><td><details><summary>view</summary><pre>${esc(c.request.messages.map(m=>`[${m.role}]\n${typeof m.content==='string'?m.content:JSON.stringify(m.content,null,1)}`).join('\n\n'))}\n\n===== RESPONSE =====\n${esc(c.response_message.content||'')}${c.response_message.tool_calls?'\n'+esc(JSON.stringify(c.response_message.tool_calls,null,1)):''}${c.error?'\nERROR: '+esc(c.error):''}</pre></details></td></tr>`).join('');
  h += step(i,7,'Model calls', `${calls.length} calls · ${fmt$(cost.cost_usd)}`, `${stepTable}${inner(`All ${calls.length} calls (prompt + response)`,`<table><tr><th>#</th><th>step</th><th>purpose</th><th>model</th><th>tok in/out</th><th>cost</th><th>lat</th><th>finish</th><th></th></tr>${callRows}</table>`)}`);

  // ---- logs
  h += step(i,8,'Logs', '', Object.entries(r.step_logs).map(([s,t])=>inner(`${s} log`,`<pre>${esc(t)}</pre>`)).join('')+inner('run.log (all steps merged)',`<pre>${esc(r.logs['run.log']||'')}</pre>`));
  return h+'</section>';
}
function imgFor(r,round,page,turn){
  const cands=Object.keys(r.images).filter(n=>n.startsWith(round+'_page'+String(page).padStart(3,'0')));
  if(!cands.length) return '';
  return `<div class="imgs">${cands.map(n=>`<figure><img src="${r.images[n]}"><figcaption>${esc(n)}</figcaption></figure>`).join('')}</div>`;
}
function range(a,b){const o=[];for(let p=a;p<=b;p++)o.push(p);return o}
function step(i,k,title,status,body){return `<details class="step" id="r${i}s${k}"${k<=6?' open':''}><summary><span class="n">${k<=6?k:'·'}</span>${esc(title)}<span class="st">${status}</span></summary><div class="body">${body}</div></details>`}
function showPage(i,p){
  const r=DATA[i]; const md=r.pages[p]||'(no markdown)'; const st=r.parser.page_stats.find(s=>s.page===p)||{};
  document.getElementById('pageview'+i).textContent=`=== page ${p} === docling ${st.docling_chars} chars · native ${st.pymupdf_chars} chars · tables ${st.tables} · pictures ${st.pictures}${st.flag?' · FLAG: '+st.flag:''}\n\n`+(md.trim()||'(empty — Docling produced no text for this page)');
  document.querySelectorAll('#pager'+i+' button').forEach(b=>b.classList.toggle('sel',b.textContent==String(p)));
  document.getElementById('pageview'+i).scrollIntoView({block:'nearest'});
}
const main=document.getElementById('main'), tabs=document.getElementById('tabs');
DATA.forEach((r,i)=>{
  const b=document.createElement('button'); b.className='tab'+(i===0?' active':''); b.textContent=`Run ${i+1}: ${r.meta.run_id.replace(/^run\d_/,'')}` + (r.result?` (${r.result.score_official_rule.toFixed(2)})`:''); b.onclick=()=>{document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active')); b.classList.add('active'); document.querySelectorAll('.run').forEach(s=>s.classList.remove('active')); document.getElementById('run'+i).classList.add('active'); window.scrollTo(0,0)}; tabs.appendChild(b);
  main.insertAdjacentHTML('beforeend', renderRun(r,i));
});
document.getElementById('theme').onclick=()=>{const h=document.documentElement; const cur=h.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'); h.dataset.theme=cur==='dark'?'light':'dark'};
</script>
</body></html>
"""


def build_visual_samples(
    runs_dir: Path, out: Path | None = None, max_runs: int = DEFAULT_MAX_RUNS
) -> Path:
    """Collect the first ``max_runs`` runs under ``runs_dir`` and write
    ``visual_samples.html`` (one tab per run)."""
    runs = discover_runs(runs_dir)
    if not runs:
        raise FileNotFoundError(
            f"no run folders with 00_input/question.json in {runs_dir}"
        )
    runs = runs[: max(1, max_runs)]
    data = [collect(d) for d in runs]
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = out or runs_dir / "visual_samples.html"
    out.write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    return out
