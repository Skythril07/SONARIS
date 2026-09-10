import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const stages = ["UPLOAD", "DECODE", "CORRECT", "ENHANCE", "DETECT", "VERIFY", "LOCATE", "REPORT", "DONE"];
const navItems = [
  ["workspace", "Workspace", "dashboard"],
  ["surveys", "Surveys", "upload"],
  ["detections", "Detections", "target"],
  ["map", "Map", "map"],
  ["reports", "Reports", "file"],
  ["settings", "Settings", "settings"],
];

const seedDetections = [
  { id: 17, type: "Fishing Gear", short: "NET", confidence: 91, lat: 8.7604, lon: 78.1321, size: 3.2, persistence: 94, verdict: "HIGH", x: 39, y: 30, shadow: "PASS", uncertainty: 4.7, note: "Strong object-shadow pair with plausible metric dimensions.", survey: "Survey Line 03" },
  { id: 18, type: "Pipe / Cable", short: "PIPE", confidence: 87, lat: 8.7718, lon: 78.1415, size: 5.8, persistence: 89, verdict: "HIGH", x: 63, y: 48, shadow: "PASS", uncertainty: 6.2, note: "Linear return persists across overlapping tiles.", survey: "Survey Line 03" },
  { id: 19, type: "Cylinder", short: "CYL", confidence: 74, lat: 8.7527, lon: 78.1219, size: 1.4, persistence: 71, verdict: "REVIEW", x: 28, y: 68, shadow: "REVIEW", uncertainty: 8.1, note: "Candidate is plausible but shadow geometry needs operator review.", survey: "Survey Line 03" },
  { id: 20, type: "Wreck", short: "WRK", confidence: 96, lat: 8.7812, lon: 78.1498, size: 12.6, persistence: 98, verdict: "HIGH", x: 78, y: 23, shadow: "PASS", uncertainty: 3.8, note: "Large persistent structure with a high-confidence acoustic shadow.", survey: "Survey Line 03" }
];

const seedSurveys = [
  { id: "SV-003", name: "Survey Line 03", file: "survey_line_03.xtf", date: "10 Sep 2026", status: "ANALYZED", detections: 4 },
  { id: "SV-002", name: "Survey Line 02", file: "survey_line_02.xtf", date: "09 Sep 2026", status: "ANALYZED", detections: 7 },
  { id: "SV-001", name: "Harbour Sweep 01", file: "harbour_sweep_01.xtf", date: "08 Sep 2026", status: "ARCHIVED", detections: 12 },
];

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" };
  const paths = {
    logo: <><path d="M4 16c4.5-7 11.5-7 16 0"/><path d="M6 19c3.5-4 8.5-4 12 0"/><circle cx="12" cy="8" r="2"/></>,
    dashboard: <><rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/><rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/></>,
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/></>,
    map: <><path d="m9 18-6 3V6l6-3 6 3 6-3v15l-6 3-6-3Z"/><path d="M9 3v15M15 6v15"/></>,
    target: <><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></>,
    layers: <><path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06-1.9 1.9-.06-.06A1.7 1.7 0 0 0 16 18.4a1.7 1.7 0 0 0-1 1.55V20h-2.7v-.05A1.7 1.7 0 0 0 11.2 18.4a1.7 1.7 0 0 0-1.87.34l-.06.06-1.9-1.9.06-.06A1.7 1.7 0 0 0 7.6 15a1.7 1.7 0 0 0-1.55-1H6v-2.7h.05A1.7 1.7 0 0 0 7.6 10a1.7 1.7 0 0 0-.34-1.87L7.2 8.07l1.9-1.9.06.06A1.7 1.7 0 0 0 11 6.6 1.7 1.7 0 0 0 12 5.05V5h2.7v.05A1.7 1.7 0 0 0 16 6.6a1.7 1.7 0 0 0 1.87-.34l.06-.06 1.9 1.9-.06.06A1.7 1.7 0 0 0 19.4 10a1.7 1.7 0 0 0 1.55 1H21v2.7h-.05A1.7 1.7 0 0 0 19.4 15Z"/></>,
    moon: <path d="M20.5 15.5A8.5 8.5 0 0 1 8.5 3.5 8.5 8.5 0 1 0 20.5 15.5Z"/>,
    sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></>,
    download: <><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4 21h16"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    x: <><path d="m6 6 12 12M18 6 6 18"/></>,
    activity: <path d="M3 12h4l2-7 4 14 2-7h6"/>,
    file: <><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v5h5"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    menu: <><path d="M4 7h16M4 12h16M4 17h16"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function App() {
  const [dark, setDark] = useState(() => localStorage.getItem("sonaris-theme") === "dark");
  const [page, setPage] = useState("workspace");
  const [selectedId, setSelectedId] = useState(17);
  const [detections, setDetections] = useState(seedDetections);
  const [surveys, setSurveys] = useState(seedSurveys);
  const [stageIndex, setStageIndex] = useState(5);
  const [processing, setProcessing] = useState(false);
  const [fileName, setFileName] = useState("survey_line_03.xtf");
  const [notice, setNotice] = useState("");
  const [filter, setFilter] = useState("ALL");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [query, setQuery] = useState("");
  const [reviewOnly, setReviewOnly] = useState(false);
  const timerRef = useRef(null);
  const selected = detections.find((d) => d.id === selectedId) || detections[0];
  const reviewCount = detections.filter((d) => d.verdict === "REVIEW").length;
  const highCount = detections.filter((d) => d.verdict === "HIGH" || d.verdict === "CONFIRMED").length;
  const avgConfidence = Math.round(detections.reduce((s, d) => s + d.confidence, 0) / Math.max(1, detections.length));

  useEffect(() => { localStorage.setItem("sonaris-theme", dark ? "dark" : "light"); }, [dark]);
  useEffect(() => () => clearInterval(timerRef.current), []);

  function showNotice(text) { setNotice(text); window.clearTimeout(showNotice.timer); showNotice.timer = window.setTimeout(() => setNotice(""), 2600); }
  function navigate(next) { setPage(next); setSidebarOpen(false); window.scrollTo({ top: 0, behavior: "smooth" }); }
  function runDemo() {
    clearInterval(timerRef.current); setProcessing(true); setStageIndex(0); showNotice("Analysis started"); let i = 0;
    timerRef.current = setInterval(() => { i += 1; setStageIndex(Math.min(i, stages.length - 1)); if (i >= stages.length - 1) { clearInterval(timerRef.current); setProcessing(false); showNotice("Analysis complete · 4 detections ready"); } }, 450);
  }
  function handleFile(e) {
    const file = e.target.files?.[0]; if (!file) return;
    setFileName(file.name); setSurveys((s) => [{ id: `SV-${String(s.length + 4).padStart(3, "0")}`, name: file.name.replace(/\.[^.]+$/, ""), file: file.name, date: "10 Sep 2026", status: "PROCESSING", detections: 0 }, ...s]); navigate("workspace"); runDemo();
  }
  function selectDetection(id, scroll = false) { setSelectedId(id); if (scroll) document.querySelector("#evidence")?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
  function updateVerdict(value) { setDetections((prev) => prev.map((d) => d.id === selectedId ? { ...d, verdict: value } : d)); showNotice(`Detection #${selectedId} marked ${value}`); }
  function exportReport(format) {
    const rows = detections.map((d) => ({ id: d.id, class: d.type, confidence: d.confidence / 100, latitude: d.lat, longitude: d.lon, size_m: d.size, persistence: d.persistence / 100, verdict: d.verdict }));
    let content = "", mime = "text/plain";
    if (format === "csv") { content = "id,class,confidence,latitude,longitude,size_m,persistence,verdict\n" + rows.map((r) => `${r.id},"${r.class}",${r.confidence},${r.latitude},${r.longitude},${r.size_m},${r.persistence},${r.verdict}`).join("\n"); mime = "text/csv"; }
    if (format === "geojson") { content = JSON.stringify({ type: "FeatureCollection", features: rows.map((r) => ({ type: "Feature", geometry: { type: "Point", coordinates: [r.longitude, r.latitude] }, properties: r })) }, null, 2); mime = "application/geo+json"; }
    if (format === "gpx") { content = `<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="SONARIS">\n${rows.map((r) => `  <wpt lat="${r.latitude}" lon="${r.longitude}"><name>${r.class} #${r.id}</name></wpt>`).join("\n")}\n</gpx>`; mime = "application/gpx+xml"; }
    const blob = new Blob([content], { type: mime }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = `sonaris-report.${format}`; a.click(); URL.revokeObjectURL(url); showNotice(`${format.toUpperCase()} exported`);
  }

  const filtered = useMemo(() => detections.filter((d) => {
    const matchesFilter = filter === "ALL" ? true : filter === "HIGH" ? (d.verdict === "HIGH" || d.verdict === "CONFIRMED") : d.verdict === filter;
    const matchesReview = reviewOnly ? d.verdict === "REVIEW" : true;
    const matchesQuery = `${d.id} ${d.type} ${d.lat} ${d.lon}`.toLowerCase().includes(query.toLowerCase());
    return matchesFilter && matchesReview && matchesQuery;
  }), [detections, filter, query, reviewOnly]);

  return <div className={dark ? "app dark" : "app"}>
    <Sidebar page={page} navigate={navigate} detections={detections} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
    <button className="mobile-menu" onClick={() => setSidebarOpen((v) => !v)} aria-label="Toggle menu"><Icon name="menu" size={20}/></button>
    {sidebarOpen && <button className="backdrop" onClick={() => setSidebarOpen(false)} aria-label="Close menu"/>}
    <div className="main-shell">
      <header className="header"><div><p className="breadcrumb">SONARIS <span>/</span> {page.toUpperCase()}</p><h1>{pageTitle(page)}</h1><p className="header-sub">{pageSubtitle(page)}</p></div><div className="header-actions"><div className="file-pill"><Icon name="file" size={15}/><span>{fileName}</span></div><button className="theme-button" onClick={() => setDark((v) => !v)} aria-label="Toggle theme"><Icon name={dark ? "sun" : "moon"} size={17}/></button>{page !== "settings" && <button className="primary-button" onClick={runDemo}><Icon name="activity" size={17}/>{processing ? "Processing…" : "Run analysis"}</button>}</div></header>
      <main className="content">
        {page === "workspace" && <Workspace {...{detections, selected, selectedId, selectDetection, stageIndex, processing, setZoom, zoom, filter, setFilter, filtered, reviewCount, highCount, avgConfidence, updateVerdict, exportReport, handleFile, fileName}} />}
        {page === "surveys" && <Surveys surveys={surveys} handleFile={handleFile} navigate={navigate} showNotice={showNotice} />}
        {page === "detections" && <DetectionsPage {...{detections, filtered, selectedId, selectDetection, filter, setFilter, query, setQuery, reviewOnly, setReviewOnly, updateVerdict}} />}
        {page === "map" && <MapPage {...{detections, selectedId, selectDetection, selected, zoom, setZoom}} />}
        {page === "reports" && <Reports detections={detections} surveys={surveys} exportReport={exportReport} />}
        {page === "settings" && <Settings dark={dark} setDark={setDark} reviewOnly={reviewOnly} setReviewOnly={setReviewOnly} />}
      </main>
      <footer className="footer"><span>SONARIS · SIH26057</span><span>AI-POWERED UNDERWATER MARINE DEBRIS & ANOMALY DETECTION</span><span>COORDINATE-CHAIN FIRST</span></footer>
    </div>
    {notice && <div className="toast"><Icon name="check" size={15}/>{notice}</div>}
  </div>;
}

function Sidebar({page, navigate, detections, sidebarOpen}) { return <div className={`sidebar ${sidebarOpen ? "open" : ""}`}><div className="sidebar-brand"><div className="brand-icon"><Icon name="logo" size={21}/></div><div><strong>SONARIS</strong><span>Marine Intelligence</span></div></div><nav className="sidebar-nav">{navItems.map(([id,label,icon]) => <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => navigate(id)}><Icon name={icon} size={17}/><span>{label}</span>{id === "detections" && <em>{detections.length}</em>}</button>)}</nav><div className="sidebar-bottom"><div className="operator-card"><div className="avatar">O</div><div><strong>Operator</strong><span>Demo workspace</span></div><span className="online"></span></div></div></div>; }

function pageTitle(page) { return ({workspace:"Marine survey workspace",surveys:"Survey library",detections:"Detection review",map:"Geospatial overview",reports:"Reports & exports",settings:"Workspace settings"})[page]; }
function pageSubtitle(page) { return ({workspace:"Review sonar detections, verify evidence, and export georeferenced results.",surveys:"Load sonar files, review processing state, and open a survey workspace.",detections:"Search, filter, and verify candidate objects across the active survey.",map:"Inspect georeferenced detections and survey coverage in one view.",reports:"Generate operator-ready exports from the current detection register.",settings:"Control theme, review behavior, and demo workspace preferences."})[page]; }

function Workspace({detections, selected, selectedId, selectDetection, stageIndex, processing, setZoom, zoom, filter, setFilter, filtered, reviewCount, highCount, avgConfidence, updateVerdict, exportReport, handleFile}) {
 return <><section className="summary-grid"><div className="summary-card accent"><span>DETECTIONS</span><strong>{detections.length}</strong><small>{highCount} high confidence</small></div><div className="summary-card"><span>AVG. CONFIDENCE</span><strong>{avgConfidence}%</strong><small>across current survey</small></div><div className="summary-card"><span>REQUIRES REVIEW</span><strong>{reviewCount}</strong><small>{reviewCount ? "operator attention needed" : "all candidates cleared"}</small></div><div className="summary-card"><span>PIPELINE</span><strong>{processing ? "RUNNING" : "READY"}</strong><small>{stageIndex + 1} of {stages.length} stages</small></div></section>
 <section className="card pipeline-card"><div className="section-head compact"><div><span className="overline">PROCESSING PIPELINE</span><h2>Survey analysis status</h2></div><span className="status-dot"><i></i>{processing ? "Processing" : "Ready"}</span></div><div className="pipeline-track">{stages.map((s,i)=><React.Fragment key={s}><div className={`pipeline-step ${i<stageIndex?'done':''} ${i===stageIndex?'current':''}`}><span>{i<stageIndex?'✓':i+1}</span><label>{s}</label></div>{i<stages.length-1&&<div className={`pipeline-connector ${i<stageIndex?'done':''}`}/>}</React.Fragment>)}</div></section>
 <section className="visual-grid"><SonarCard {...{detections,selectedId,selectDetection,zoom,setZoom}}/><MapCard {...{detections,selectedId,selectDetection,selected}}/></section>
 <section className="workspace-grid"><DetectionTable {...{detections,filtered,selectedId,selectDetection,filter,setFilter}}/><Evidence selected={selected} updateVerdict={updateVerdict}/></section>
 <section className="action-bar"><label className="upload-button"><Icon name="upload" size={17}/>Load sonar file<input type="file" accept=".xtf,.dat,.sl2,.rsd,image/*" onChange={handleFile}/></label><div className="exports"><span>EXPORT</span><button onClick={()=>exportReport("csv")}>CSV</button><button onClick={()=>exportReport("geojson")}>GeoJSON</button><button onClick={()=>exportReport("gpx")}>GPX</button></div></section></>;
}

function SonarCard({detections,selectedId,selectDetection,zoom,setZoom}) { return <div className="card visual-card"><div className="section-head"><div><span className="overline">SONAR VIEW</span><h2>Acoustic waterfall</h2></div><div className="viewer-tools"><button onClick={()=>setZoom(z=>Math.max(.8,z-.1))}>−</button><span>{Math.round(zoom*100)}%</span><button onClick={()=>setZoom(z=>Math.min(1.6,z+.1))}>+</button></div></div><div className="sonar-view" style={{"--zoom":zoom}}><div className="sonar-grid"/><div className="sonar-band one"/><div className="sonar-band two"/><div className="sonar-band three"/><div className="sonar-nadir"/>{detections.map(d=><button key={d.id} className={`sonar-box ${selectedId===d.id?'selected':''}`} style={{left:`${d.x}%`,top:`${d.y}%`}} onClick={()=>selectDetection(d.id,true)}><span>{d.short}</span><b>{d.confidence}%</b></button>)}</div><div className="visual-footer"><span>AI detections</span><span>Acoustic shadow</span><span>455 kHz · 100 m range</span></div></div>; }
function MapCard({detections,selectedId,selectDetection,selected}) { return <div className="card visual-card"><div className="section-head"><div><span className="overline">GEOSPATIAL VIEW</span><h2>Survey track</h2></div><span className="tag">LIVE</span></div><MapSurface {...{detections,selectedId,selectDetection,selected}}/></div>; }
function MapSurface({detections,selectedId,selectDetection,selected}) { return <div className="map-view"><div className="map-grid"/><div className="map-depth"/><div className="land-shape"/><svg className="track-svg" viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M7 79 C19 66,18 58,30 61 S44 43,52 51 S65 37,76 43 S83 27,94 18"/><path className="track-main" d="M7 79 C19 66,18 58,30 61 S44 43,52 51 S65 37,76 43 S83 27,94 18"/></svg>{detections.map(d=><button key={d.id} className={`map-pin ${selectedId===d.id?'selected':''}`} style={{left:`${d.x}%`,top:`${d.y}%`}} onClick={()=>selectDetection(d.id)}>{d.id}</button>)}<div className="coords"><small>SELECTED LOCATION</small><strong>{selected.lat.toFixed(4)}° N</strong><strong>{selected.lon.toFixed(4)}° E</strong><span>±{selected.uncertainty} m</span></div></div>; }

function DetectionTable({detections,filtered,selectedId,selectDetection,filter,setFilter}) { return <div className="card table-card"><div className="section-head"><div><span className="overline">DETECTION REGISTER</span><h2>Candidate objects <b>{detections.length}</b></h2></div><div className="filter-group">{["ALL","HIGH","REVIEW"].map(f=><button key={f} className={filter===f?'active':''} onClick={()=>setFilter(f)}>{f}</button>)}</div></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>CLASS</th><th>CONFIDENCE</th><th>LOCATION</th><th>SIZE</th><th>VERDICT</th></tr></thead><tbody>{filtered.map(d=><tr key={d.id} className={selectedId===d.id?'selected-row':''} onClick={()=>selectDetection(d.id,true)}><td><span className="id-chip">#{d.id}</span></td><td><strong>{d.type}</strong><small>{d.persistence}% persistent</small></td><td><div className="confidence"><span><i style={{width:`${d.confidence}%`}}/></span><b>{d.confidence}%</b></div></td><td>{d.lat.toFixed(4)}<br/><span>{d.lon.toFixed(4)}</span></td><td>{d.size} m</td><td><span className={`verdict ${d.verdict.toLowerCase()}`}>{d.verdict}</span></td></tr>)}</tbody></table></div></div>; }
function Evidence({selected,updateVerdict}) { return <aside className="card evidence-card" id="evidence"><div className="section-head"><div><span className="overline">EVIDENCE REVIEW</span><h2>Detection #{selected.id}</h2></div><span className={`verdict ${selected.verdict.toLowerCase()}`}>{selected.verdict}</span></div><div className="evidence-image"><div className="evidence-object"><span>{selected.short}</span></div><div className="evidence-shadow"/><small>TILE {selected.id+104} · FULL RES</small></div><div className="evidence-top"><div><h3>{selected.type}</h3><p>Candidate #{selected.id} · acoustic anomaly</p></div><strong>{selected.confidence}<small>%</small></strong></div><div className="metric-list"><div><span>Shadow consistency</span><b className="pass">✓ {selected.shadow}</b></div><div><span>Size plausibility</span><b className="pass">✓ PASS</b></div><div><span>Tile persistence</span><b>{selected.persistence}%</b></div><div><span>Estimated size</span><b>{selected.size} m</b></div><div><span>Position error</span><b>±{selected.uncertainty} m</b></div><div><span>Coordinates</span><b>{selected.lat.toFixed(4)}, {selected.lon.toFixed(4)}</b></div></div><p className="evidence-note"><Icon name="info" size={15}/>{selected.note}</p><div className="review-actions"><button className="confirm" onClick={()=>updateVerdict("CONFIRMED")}><Icon name="check" size={16}/>Confirm</button><button className="reject" onClick={()=>updateVerdict("REJECTED")}><Icon name="x" size={16}/>Reject</button></div></aside>; }

function Surveys({surveys,handleFile,navigate,showNotice}) { return <div className="page-stack"><div className="split-panel"><div><span className="overline">SURVEY INGESTION</span><h2>Start a new survey</h2><p>Load XTF or supported sonar files to create a reviewable survey workspace.</p></div><label className="primary-upload"><Icon name="upload"/>Choose sonar file<input type="file" accept=".xtf,.dat,.sl2,.rsd,image/*" onChange={handleFile}/></label></div><div className="page-section-head"><div><span className="overline">SURVEY LIBRARY</span><h2>Recent surveys</h2></div><span className="count-chip">{surveys.length} surveys</span></div><div className="survey-grid">{surveys.map(s=><button className="survey-card" key={s.id} onClick={()=>{showNotice(`${s.name} opened`);navigate("workspace");}}><div className="survey-top"><span className={`status-pill ${s.status.toLowerCase()}`}>{s.status}</span><span>{s.id}</span></div><h3>{s.name}</h3><p>{s.file}</p><div className="survey-meta"><span>{s.date}</span><strong>{s.detections || 0} detections</strong></div></button>)}</div></div>; }

function DetectionsPage({detections,filtered,selectedId,selectDetection,filter,setFilter,query,setQuery,reviewOnly,setReviewOnly,updateVerdict}) { const selected = detections.find(d=>d.id===selectedId)||detections[0]; return <div className="page-stack"><div className="toolbar"><div className="searchbox"><Icon name="search" size={16}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search ID, class, coordinate…"/></div><div className="filter-group">{["ALL","HIGH","REVIEW"].map(f=><button key={f} className={filter===f?'active':''} onClick={()=>setFilter(f)}>{f}</button>)}</div><label className="toggle-label"><input type="checkbox" checked={reviewOnly} onChange={e=>setReviewOnly(e.target.checked)}/><span/>Review only</label></div><div className="detection-layout"><div className="card table-card"><div className="section-head"><div><span className="overline">RESULTS</span><h2>{filtered.length} matching candidates</h2></div></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>CLASS</th><th>CONF.</th><th>LOCATION</th><th>VERDICT</th></tr></thead><tbody>{filtered.map(d=><tr key={d.id} className={selectedId===d.id?'selected-row':''} onClick={()=>selectDetection(d.id)}><td>#{d.id}</td><td><strong>{d.type}</strong><small>{d.size} m · {d.persistence}% persistent</small></td><td><b>{d.confidence}%</b></td><td>{d.lat.toFixed(4)}, {d.lon.toFixed(4)}</td><td><span className={`verdict ${d.verdict.toLowerCase()}`}>{d.verdict}</span></td></tr>)}</tbody></table></div></div><Evidence selected={selected} updateVerdict={updateVerdict}/></div></div>; }

function MapPage({detections,selectedId,selectDetection,selected,zoom,setZoom}) { return <div className="page-stack"><div className="card map-page-card"><div className="section-head"><div><span className="overline">SURVEY COVERAGE</span><h2>Geospatial detection map</h2></div><div className="viewer-tools"><button onClick={()=>setZoom(z=>Math.max(.8,z-.1))}>−</button><span>{Math.round(zoom*100)}%</span><button onClick={()=>setZoom(z=>Math.min(1.6,z+.1))}>+</button></div></div><MapSurface {...{detections,selectedId,selectDetection,selected}}/></div><div className="map-stats"><div className="stat-panel"><span>TRACK LENGTH</span><strong>8.42 km</strong><small>current survey</small></div><div className="stat-panel"><span>AREA COVERED</span><strong>0.91 km²</strong><small>estimated swath</small></div><div className="stat-panel"><span>POSITION ERROR</span><strong>±4.7 m</strong><small>selected candidate</small></div></div></div>; }

function Reports({detections,surveys,exportReport}) { return <div className="page-stack"><div className="report-hero"><div><span className="overline">CURRENT DATASET</span><h2>{surveys[0]?.name || "Survey"}</h2><p>{detections.length} detections · georeferenced · operator-reviewed</p></div><button className="primary-button" onClick={()=>exportReport("csv")}><Icon name="download" size={16}/>Export CSV</button></div><div className="report-grid"><button className="report-card" onClick={()=>exportReport("csv")}><Icon name="file" size={19}/><h3>CSV register</h3><p>Tabular detection register for analysis and handoff.</p><strong>Download CSV</strong></button><button className="report-card" onClick={()=>exportReport("geojson")}><Icon name="map" size={19}/><h3>GeoJSON</h3><p>Point features with coordinates and detection metadata.</p><strong>Download GeoJSON</strong></button><button className="report-card" onClick={()=>exportReport("gpx")}><Icon name="target" size={19}/><h3>GPX waypoints</h3><p>Field-friendly coordinates for navigation workflows.</p><strong>Download GPX</strong></button></div><div className="card report-summary"><div className="section-head"><div><span className="overline">QUALITY SNAPSHOT</span><h2>Current report readiness</h2></div><span className="status-pill analyzed">READY</span></div><div className="quality-grid"><div><span>Detections</span><strong>{detections.length}</strong></div><div><span>High confidence</span><strong>{detections.filter(d=>d.confidence>=85).length}</strong></div><div><span>Needs review</span><strong>{detections.filter(d=>d.verdict==='REVIEW').length}</strong></div><div><span>Georeferenced</span><strong>100%</strong></div></div></div></div>; }

function Settings({dark,setDark,reviewOnly,setReviewOnly}) { return <div className="page-stack settings-stack"><div className="card settings-card"><div className="settings-row"><div><span className="overline">APPEARANCE</span><h3>Theme</h3><p>Choose the visual mode for the SONARIS workspace.</p></div><button className="segmented" onClick={()=>setDark(v=>!v)}><span className={!dark?'selected':''}>Light</span><span className={dark?'selected':''}>Dark</span></button></div><div className="settings-row"><div><span className="overline">REVIEW</span><h3>Keep review filter on</h3><p>When enabled, detection screens emphasize unresolved candidates.</p></div><label className="switch"><input type="checkbox" checked={reviewOnly} onChange={e=>setReviewOnly(e.target.checked)}/><span/></label></div><div className="settings-row"><div><span className="overline">WORKSPACE</span><h3>Demo mode</h3><p>Frontend-only demo state. Analysis and exports run locally in this browser.</p></div><span className="status-pill analyzed">ENABLED</span></div></div><div className="card settings-card"><div className="section-head"><div><span className="overline">ABOUT</span><h2>SONARIS</h2></div></div><p className="about-copy">Marine intelligence interface for side-scan sonar detection review, geolocation, verification, and operator-ready exports.</p><div className="about-meta"><span>SIH26057</span><span>Frontend demo</span><span>v1.0</span></div></div></div>; }

createRoot(document.getElementById("root")).render(<App/>);
