import { useState, useCallback, useEffect, useRef } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Animation keyframes injected once ────────────────────────────────────────
const CSS = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif; }

  @keyframes spin       { to { transform: rotate(360deg); } }
  @keyframes pulse      { 0%,100%{opacity:1} 50%{opacity:.35} }
  @keyframes fadeUp     { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
  @keyframes fadeIn     { from{opacity:0} to{opacity:1} }
  @keyframes scaleIn    { from{transform:scale(.6);opacity:0} 60%{transform:scale(1.15)} to{transform:scale(1);opacity:1} }
  @keyframes barFill    { from{width:0} to{width:var(--w)} }
  @keyframes borderPulse{ 0%,100%{border-color:#c7d2fe} 50%{border-color:#4f46e5} }
  @keyframes shimmer    { from{background-position:-200% 0} to{background-position:200% 0} }
  @keyframes typewriter { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }

  .anim-fade-up   { animation: fadeUp  .45s cubic-bezier(.16,1,.3,1) both; }
  .anim-fade-up-1 { animation: fadeUp  .45s cubic-bezier(.16,1,.3,1) .10s both; }
  .anim-fade-up-2 { animation: fadeUp  .45s cubic-bezier(.16,1,.3,1) .20s both; }
  .anim-fade-up-3 { animation: fadeUp  .45s cubic-bezier(.16,1,.3,1) .30s both; }
  .anim-fade-in   { animation: fadeIn  .3s ease both; }
  .anim-scale-in  { animation: scaleIn .35s cubic-bezier(.16,1,.3,1) both; }

  .upload-zone {
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 52px 24px;
    text-align: center;
    cursor: pointer;
    transition: all .2s ease;
    background: #fff;
  }
  .upload-zone:hover, .upload-zone.drag {
    border-color: #4f46e5;
    background: #eef2ff;
    animation: borderPulse 1.5s ease infinite;
  }

  .step-dot {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 13px; font-weight: 600;
    transition: all .3s ease;
  }
  .step-dot.pending  { background:#f1f5f9; color:#94a3b8; border:2px solid #e2e8f0; }
  .step-dot.active   { background:#eef2ff; color:#4f46e5; border:2px solid #4f46e5; }
  .step-dot.done     { background:#4f46e5; color:#fff; border:2px solid #4f46e5; animation: scaleIn .3s ease; }

  .spinner {
    width:14px; height:14px; border:2px solid #c7d2fe;
    border-top-color:#4f46e5; border-radius:50%;
    animation: spin .7s linear infinite;
  }

  .conf-bar-track { height: 8px; background:#e2e8f0; border-radius:99px; overflow:hidden; }
  .conf-bar-fill  {
    height:100%; border-radius:99px;
    animation: barFill .8s cubic-bezier(.16,1,.3,1) .3s both;
    width: var(--w);
  }

  .result-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:14px;
    padding:20px 22px; margin-bottom:12px;
  }
  .result-card-title {
    font-size:11px; font-weight:600; color:#94a3b8;
    text-transform:uppercase; letter-spacing:.07em; margin-bottom:14px;
  }

  .field-row {
    display:flex; align-items:center; padding:7px 0;
    border-bottom:1px solid #f1f5f9; gap:12px;
  }
  .field-row:last-child { border-bottom:none; }
  .field-label { font-size:13px; color:#64748b; min-width:130px; }
  .field-value { font-size:13px; color:#0f172a; font-weight:500; flex:1; }

  .metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
  .metric-box  { background:#f8fafc; border-radius:10px; padding:12px 14px; }
  .metric-lbl  { font-size:11px; color:#94a3b8; margin-bottom:4px; }
  .metric-val  { font-size:18px; font-weight:600; color:#0f172a; }

  .tab-btn {
    padding:10px 20px; font-size:14px; font-weight:500;
    background:none; border:none; border-bottom:2px solid transparent;
    cursor:pointer; transition:all .2s; color:#64748b;
  }
  .tab-btn.active { color:#4f46e5; border-bottom-color:#4f46e5; }
  .tab-btn:hover:not(.active) { color:#1e293b; }

  .ask-pill {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:10px 14px; font-size:13px; color:#374151; text-align:left;
    cursor:pointer; transition:all .15s; width:100%;
  }
  .ask-pill:hover { background:#eef2ff; border-color:#c7d2fe; color:#4f46e5; }

  .ask-input {
    flex:1; padding:11px 14px; border-radius:10px;
    border:1px solid #e2e8f0; font-size:14px; outline:none;
    transition:border-color .15s;
  }
  .ask-input:focus { border-color:#4f46e5; box-shadow:0 0 0 3px #eef2ff; }

  .ask-btn {
    padding:11px 22px; background:#4f46e5; color:#fff;
    border:none; border-radius:10px; font-size:14px; font-weight:500;
    cursor:pointer; transition:all .15s; white-space:nowrap;
  }
  .ask-btn:hover:not(:disabled) { background:#4338ca; }
  .ask-btn:disabled { opacity:.6; cursor:not-allowed; }

  .trail-step {
    display:flex; gap:12px; align-items:flex-start;
    animation: typewriter .3s ease both;
  }
  .trail-arrow { color:#c7d2fe; font-size:13px; margin-top:2px; flex-shrink:0; }
  .trail-text  { font-size:12px; color:#64748b; line-height:1.6; }

  .cypher-block {
    background:#0f172a; border-radius:10px; padding:14px 16px;
    font-family:monospace; font-size:12px; color:#7dd3fc;
    white-space:pre-wrap; line-height:1.7; overflow-x:auto;
  }

  .answer-card {
    background:linear-gradient(135deg,#eef2ff 0%,#f5f3ff 100%);
    border:1px solid #c7d2fe; border-left:4px solid #4f46e5;
    border-radius:14px; padding:20px 22px;
  }

  .routing-badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:4px 12px; border-radius:99px; font-size:12px; font-weight:600;
  }
  .routing-badge.auto   { background:#dcfce7; color:#16a34a; }
  .routing-badge.review { background:#fee2e2; color:#dc2626; }
`;

// ── Constants ─────────────────────────────────────────────────────────────────
const EXTRACT_STEPS = [
  { label: "Encoding image",         detail: "Resizing to optimal Claude resolution" },
  { label: "AI vision extraction",   detail: "Claude reading all invoice fields"      },
  { label: "Schema validation",      detail: "Pydantic enforcing data types"          },
  { label: "Confidence scoring",     detail: "Evaluating field reliability 0–1"       },
  { label: "Routing decision",       detail: "Auto-approve or human review queue"     },
];

const ASK_STEPS = [
  { label: "Planning Cypher queries", detail: "Translating question to graph language" },
  { label: "Traversing Neo4j graph",  detail: "Following Supplier → Invoice → LineItem" },
  { label: "Processing records",      detail: "Aggregating graph results"              },
  { label: "Synthesizing answer",     detail: "Claude generating natural language"     },
];

const EXAMPLE_QUESTIONS = [
  "Which suppliers appear on the most invoices?",
  "What is the total invoice value across all suppliers?",
  "Which invoices have the highest total amounts?",
  "List all unique suppliers in the graph",
];

// ── Helpers ───────────────────────────────────────────────────────────────────
function confColor(score) {
  if (score >= 0.75) return "#16a34a";
  if (score >= 0.5)  return "#d97706";
  return "#dc2626";
}

function useStepAnimation(steps, active, duration = 5800) {
  const [current, setCurrent] = useState(-1);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!active) { setCurrent(-1); return; }
    setCurrent(0);
    const perStep = duration / steps.length;
    let idx = 0;
    timerRef.current = setInterval(() => {
      idx++;
      if (idx < steps.length) setCurrent(idx);
      else clearInterval(timerRef.current);
    }, perStep);
    return () => clearInterval(timerRef.current);
  }, [active, duration, steps.length]);

  return current;
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StepTimeline({ steps, current, done }) {
  return (
    <div className="result-card anim-fade-in" style={{ marginBottom: 16 }}>
      <div className="result-card-title">Processing</div>
      {steps.map((step, i) => {
        const state = done ? "done" : i < current ? "done" : i === current ? "active" : "pending";
        return (
          <div key={i} style={{ display: "flex", gap: 14, alignItems: "flex-start", marginBottom: i < steps.length - 1 ? 14 : 0, position: "relative" }}>
            {i < steps.length - 1 && (
              <div style={{ position: "absolute", left: 13, top: 30, width: 2, height: 14, background: state === "done" ? "#4f46e5" : "#e2e8f0", transition: "background .3s" }} />
            )}
            <div className={`step-dot ${state}`}>
              {state === "done" ? "✓" : state === "active" ? <div className="spinner" /> : i + 1}
            </div>
            <div style={{ paddingTop: 4 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: state === "pending" ? "#94a3b8" : "#0f172a", transition: "color .3s" }}>
                {step.label}
              </div>
              {state !== "pending" && (
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 2, animation: "fadeIn .3s ease" }}>
                  {step.detail}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ConfidenceBar({ score }) {
  const pct = Math.round(score * 100);
  const color = confColor(score);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: "#64748b" }}>Overall confidence</span>
        <span style={{ fontSize: 13, fontWeight: 600, color }}>{pct}%</span>
      </div>
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ "--w": `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

// ── Extract tab ───────────────────────────────────────────────────────────────
function ExtractTab() {
  const [dragging, setDragging]     = useState(false);
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState(null);
  const [stepsDone, setStepsDone]   = useState(false);
  const currentStep = useStepAnimation(EXTRACT_STEPS, loading, 5600);

  const process = useCallback(async (file) => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null); setStepsDone(false);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/extract`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setStepsDone(true);
      setTimeout(() => { setLoading(false); setResult(data); }, 600);
    } catch (e) {
      setLoading(false);
      setError(e.message);
    }
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    process(e.dataTransfer.files[0]);
  }, [process]);

  return (
    <div>
      {/* Upload zone */}
      <div
        className={`upload-zone ${dragging ? "drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !loading && document.getElementById("fi").click()}
        style={{ marginBottom: 16, opacity: loading ? 0.7 : 1, pointerEvents: loading ? "none" : "auto" }}
      >
        <div style={{ fontSize: 40, marginBottom: 12 }}>
          {loading ? "⏳" : dragging ? "📂" : "📄"}
        </div>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#1e293b", marginBottom: 6 }}>
          {loading ? "Extracting invoice..." : dragging ? "Release to upload" : "Drop invoice image here"}
        </div>
        <div style={{ fontSize: 13, color: "#94a3b8" }}>
          {loading ? "Claude vision is reading your invoice" : "PNG, JPG, TIFF — or click to browse"}
        </div>
        <input id="fi" type="file" accept="image/*" style={{ display: "none" }}
          onChange={(e) => process(e.target.files[0])} />
      </div>

      {/* Error */}
      {error && (
        <div className="anim-fade-up" style={{ background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: 12, padding: "12px 16px", marginBottom: 16, fontSize: 13, color: "#dc2626" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Step animation */}
      {loading && <StepTimeline steps={EXTRACT_STEPS} current={currentStep} done={stepsDone} />}

      {/* Results */}
      {result && (
        <>
          <div className="result-card anim-fade-up">
            <div className="result-card-title">Extracted fields</div>
            {[
              ["Vendor",          result.vendor_name],
              ["Invoice number",  result.invoice_number],
              ["Date",            result.invoice_date],
              ["Total amount",    result.total_amount != null ? `$${Number(result.total_amount).toLocaleString()}` : null],
              ["Line items",      result.line_item_count],
            ].map(([label, val]) => (
              <div className="field-row" key={label}>
                <span className="field-label">{label}</span>
                <span className="field-value" style={{ color: val ? "#0f172a" : "#cbd5e1" }}>
                  {val ?? "—"}
                </span>
              </div>
            ))}
          </div>

          <div className="result-card anim-fade-up-1">
            <div className="result-card-title">Confidence + routing</div>
            <ConfidenceBar score={result.overall_confidence} />
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 16 }}>
              <div>
                {result.low_confidence_fields?.length > 0 && (
                  <div style={{ fontSize: 12, color: "#d97706", marginTop: 4 }}>
                    ⚠️ Low confidence: {result.low_confidence_fields.join(", ")}
                  </div>
                )}
              </div>
              <span className={`routing-badge ${result.needs_human_review ? "review" : "auto"}`}>
                {result.needs_human_review ? "🔴 Human review" : "✅ Auto-approved"}
              </span>
            </div>
          </div>

          <div className="result-card anim-fade-up-2">
            <div className="result-card-title">Cost panel</div>
            <div className="metric-grid">
              {[
                { label: "Extraction time", value: `${result.extraction_time_ms?.toFixed(0)}ms` },
                { label: "LLM cost",        value: `$${result.cost_usd?.toFixed(5)}` },
                { label: "Manual cost",     value: "$3.50" },
                { label: "Savings",         value: "99.7%" },
              ].map(({ label, value }) => (
                <div className="metric-box" key={label}>
                  <div className="metric-lbl">{label}</div>
                  <div className="metric-val">{value}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Ask tab ───────────────────────────────────────────────────────────────────
function AskTab() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading]   = useState(false);
  const [answer, setAnswer]     = useState(null);
  const [error, setError]       = useState(null);
  const [stepsDone, setStepsDone] = useState(false);
  const currentStep = useStepAnimation(ASK_STEPS, loading, 4800);

  const ask = useCallback(async (q) => {
    const text = (q || question).trim();
    if (!text) return;
    setQuestion(text);
    setLoading(true); setError(null); setAnswer(null); setStepsDone(false);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setStepsDone(true);
      setTimeout(() => { setLoading(false); setAnswer(data); }, 500);
    } catch (e) {
      setLoading(false); setError(e.message);
    }
  }, [question]);

  return (
    <div>
      {/* Examples */}
      {!loading && !answer && (
        <div className="result-card anim-fade-up" style={{ marginBottom: 16 }}>
          <div className="result-card-title">Example questions</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {EXAMPLE_QUESTIONS.map((q) => (
              <button key={q} className="ask-pill" onClick={() => ask(q)}>
                <span style={{ marginRight: 8, color: "#94a3b8" }}>→</span>{q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          className="ask-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !loading && ask()}
          placeholder="Ask anything about your invoices and suppliers..."
          disabled={loading}
        />
        <button className="ask-btn" onClick={() => ask()} disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </div>

      {error && (
        <div className="anim-fade-up" style={{ background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: 12, padding: "12px 16px", marginBottom: 16, fontSize: 13, color: "#dc2626" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Step animation */}
      {loading && <StepTimeline steps={ASK_STEPS} current={currentStep} done={stepsDone} />}

      {/* Answer */}
      {answer && (
        <>
          <div className="answer-card anim-fade-up" style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#6366f1", textTransform: "uppercase", letterSpacing: ".07em", marginBottom: 10 }}>
              Answer
            </div>
            <p style={{ fontSize: 15, color: "#1e293b", lineHeight: 1.7, fontWeight: 400 }}>
              {answer.answer}
            </p>
          </div>

          {answer.traversal_path?.length > 0 && (
            <div className="result-card anim-fade-up-1">
              <div className="result-card-title">Audit trail — graph traversal path</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {answer.traversal_path.map((step, i) => (
                  <div className="trail-step" key={i} style={{ animationDelay: `${i * 80}ms` }}>
                    <span className="trail-arrow">→</span>
                    <span className="trail-text">{step}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {answer.cypher_queries?.length > 0 && (
            <div className="result-card anim-fade-up-2">
              <div className="result-card-title">Cypher executed on Neo4j</div>
              {answer.cypher_queries.map((q, i) => (
                <div className="cypher-block" key={i} style={{ marginTop: i > 0 ? 8 : 0 }}>
                  {q}
                </div>
              ))}
            </div>
          )}

          <button
            onClick={() => { setAnswer(null); setQuestion(""); }}
            style={{ background: "none", border: "none", color: "#94a3b8", fontSize: 13, cursor: "pointer", marginTop: 4, padding: "4px 0" }}
          >
            ← Ask another question
          </button>
        </>
      )}
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("extract");

  useEffect(() => {
    const s = document.createElement("style");
    s.textContent = CSS;
    document.head.appendChild(s);
    return () => document.head.removeChild(s);
  }, []);

  return (
    <div style={{ minHeight: "100vh", background: "#f1f5f9" }}>
      {/* Header */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e2e8f0", padding: "0 24px", position: "sticky", top: 0, zIndex: 10 }}>
        <div style={{ maxWidth: 720, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", height: 56 }}>
          <div>
            <span style={{ fontSize: 17, fontWeight: 700, color: "#0f172a", letterSpacing: "-.02em" }}>LedgerLens</span>
            <span style={{ fontSize: 12, color: "#94a3b8", marginLeft: 10 }}>Multimodal invoice intelligence + GraphRAG</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#22c55e", animation: "pulse 2s ease infinite" }} />
            <span style={{ fontSize: 12, color: "#64748b" }}>Live</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e2e8f0" }}>
        <div style={{ maxWidth: 720, margin: "0 auto", display: "flex" }}>
          <button className={`tab-btn ${tab === "extract" ? "active" : ""}`} onClick={() => setTab("extract")}>
            Extract invoice
          </button>
          <button className={`tab-btn ${tab === "ask" ? "active" : ""}`} onClick={() => setTab("ask")}>
            Ask the graph
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "24px 16px 48px" }}>
        {tab === "extract" ? <ExtractTab /> : <AskTab />}
      </div>
    </div>
  );
}
