# app.py — Sentinel-AI (Hugging Face Spaces / ZeroGPU)
#
# Architecture:
#   - demo.launch() is the entrypoint (required for ZeroGPU detection)
#   - App.create_app is monkey-patched to inject:
#       * ReactMiddleware  → serves React at / and /assets/*
#       * /api/triage      → POST inference endpoint
#       * /api/stats       → GET benchmark stats
#   - @spaces.GPU is applied to the actual triage function so ZeroGPU
#     detects and registers it at startup time.

import os, sys, csv, hashlib, traceback, mimetypes, time, asyncio
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("CRITICAL: GROQ_API_KEY environment variable is missing. Please check your .env file.")

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
DIST = os.path.join(ROOT, "frontend", "dist")

# ──────────────────────────────────────────────────────────────
# 1.  Imports & Prometheus Observability Metrics
# ──────────────────────────────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, HTMLResponse
from fastapi.responses import JSONResponse
import gradio as gr
import spaces
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

FAISS_LOOKUP_LATENCY = Histogram(
    "faiss_lookup_latency_ms",
    "FAISS vector search latency in milliseconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
)

LLM_INFERENCE_LATENCY = Histogram(
    "llm_inference_latency_ms",
    "Groq LLM inference latency in milliseconds",
    buckets=[100, 250, 500, 1000, 2000, 3500, 5000, 8000, 12000]
)

ESCALATION_GATE_TRIPPED = Counter(
    "escalation_gate_tripped_total",
    "Total queries tripped by the escalation safety gate",
    ["gate_layer"]
)

TOTAL_TRIAGE_LATENCY = Histogram(
    "triage_request_latency_ms",
    "Total end-to-end triage latency in milliseconds",
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

def _process_triage_sync(text: str) -> dict:
    t0 = time.perf_counter()
    from src.classifier import classify
    from src.escalation import gatekeeper, _get_cache, rule_based_escalation
    from src.reply_drafter import draft_reply
    from src.retriever import retrieve

    # Measure FAISS lookup latency
    t_faiss_start = time.perf_counter()
    _ = retrieve(text, k=3)
    t_faiss_end = time.perf_counter()
    faiss_ms = (t_faiss_end - t_faiss_start) * 1000.0
    FAISS_LOOKUP_LATENCY.observe(faiss_ms)

    # Deterministic rule gate check
    rule_esc, rule_reason = rule_based_escalation(text, 1.0)
    if rule_esc:
        ESCALATION_GATE_TRIPPED.labels(gate_layer="rule_based").inc()

    # Classification & LLM reasoning
    t_llm_start = time.perf_counter()
    intent_res = classify(text)
    escalation_decision = gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)
    t_llm_end = time.perf_counter()
    llm_ms = (t_llm_end - t_llm_start) * 1000.0
    LLM_INFERENCE_LATENCY.observe(llm_ms)

    h = hashlib.sha256(text.encode()).hexdigest()
    esc = _get_cache().get(h, {})

    if escalation_decision.should_escalate:
        if not rule_esc:
            ESCALATION_GATE_TRIPPED.labels(gate_layer="llm_gatekeeper").inc()
        draft_reply_text = "[ESCALATED TO HUMAN AGENT] Automated reply withheld for safety review. Case routed to Tier-2 support team."
        policy_adherence_text = f"Safety Gate Triggered: {escalation_decision.reason}. High-risk cases require human specialist intervention."
    else:
        reply_res = draft_reply(text)
        draft_reply_text = reply_res.draft_reply
        policy_adherence_text = reply_res.policy_adherence_check

    t_total_ms = (time.perf_counter() - t0) * 1000.0
    TOTAL_TRIAGE_LATENCY.observe(t_total_ms)

    return {
        "intent":            intent_res.intent.value,
        "confidence":        round(intent_res.confidence, 3),
        "reasoning":         intent_res.reasoning,
        "escalation_score":  round(float(esc.get("escalation_score", 1.0)), 3) if "escalation_score" in esc else 1.0,
        "should_escalate":   escalation_decision.should_escalate,
        "escalation_reason": escalation_decision.reason,
        "draft_reply":       draft_reply_text,
        "policy_adherence":  policy_adherence_text,
        "latency_breakdown": {
            "faiss_lookup_ms": round(faiss_ms, 2),
            "llm_inference_ms": round(llm_ms, 2),
            "total_ms": round(t_total_ms, 2)
        }
    }

# ──────────────────────────────────────────────────────────────
# 2.  React middleware  (intercepts / and /assets/* before Gradio routing)
# ──────────────────────────────────────────────────────────────
class ReactMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Root → React index.html
        if path in ("/", ""):
            idx = os.path.join(DIST, "index.html")
            with open(idx, encoding="utf-8") as f:
                return HTMLResponse(f.read())

        # Static assets (/assets/…)
        if path.startswith("/assets/"):
            rel = path.lstrip("/").replace("/", os.sep)
            fp = os.path.join(DIST, rel)
            if os.path.exists(fp):
                mime, _ = mimetypes.guess_type(fp)
                with open(fp, "rb") as f:
                    return Response(
                        content=f.read(),
                        media_type=mime or "application/octet-stream",
                        headers={"Cache-Control": "public, max-age=31536000"},
                    )

        # SVG assets (favicon, icons)
        if path in ("/favicon.svg", "/icons.svg"):
            fp = os.path.join(DIST, path.lstrip("/"))
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    return Response(content=f.read(), media_type="image/svg+xml")

        return await call_next(request)


# ──────────────────────────────────────────────────────────────
# 3.  Monkey-patch App.create_app to inject middleware + API routes
#     Uses __func__ to get the raw function, avoiding recursion entirely.
# ──────────────────────────────────────────────────────────────
from gradio.routes import App

_orig_create_app = App.create_app   # @staticmethod — just a plain function

@staticmethod
def _patched_create_app(blocks, **kwargs):
    # Call the original directly — zero recursion risk
    app = _orig_create_app(blocks, **kwargs)

    # ── React middleware (runs before every route lookup) ──
    app.add_middleware(ReactMiddleware)

    # ── API: POST /api/triage and /classify (Asynchronous execution) ──
    @app.post("/api/triage")
    @app.post("/classify")
    async def _api_triage(request: Request):
        try:
            body = await request.json()
            text = body.get("text", "").strip()
            if not text:
                return JSONResponse({"error": "text is required"}, status_code=400)

            # Asynchronous offload to threadpool to allow concurrent FAISS + LLM execution
            result = await asyncio.to_thread(_process_triage_sync, text)
            return JSONResponse(result)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    # ── Prometheus Observability: GET /metrics ─────────────
    @app.get("/metrics")
    async def _metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── Health Check: GET /healthz & /health (Liveness/Readiness Probe) ──
    @app.get("/healthz")
    @app.get("/health")
    async def _healthz():
        return JSONResponse({
            "status": "healthy",
            "service": "sentinel-ai",
            "version": "1.0.0",
            "deployment_mode": os.getenv("DEPLOYMENT_MODE", "web_service")
        })

    # ── API: GET /api/stats ────────────────────────────────
    @app.get("/api/stats")
    async def _api_stats():
        csv_path = os.path.join(ROOT, "eval", "results.csv")
        if not os.path.exists(csv_path):
            return JSONResponse({"error": "results.csv not found"}, status_code=404)
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append({k: float(v) for k, v in row.items()})
        best = None
        for row in rows:
            if row["FNR"] < 0.40:
                if best is None or row["Precision"] > best["Precision"]:
                    best = row
        return JSONResponse({
            "rows":          rows,
            "best_threshold": best,
            "macro_f1":      rows[0]["Macro-F1"] if rows else None,
            "total_samples": int(
                rows[0]["TP"] + rows[0]["FN"] + rows[0]["FP"] + rows[0]["TN"]
            ) if rows else 0,
        })

    return app

App.create_app = staticmethod(_patched_create_app)


# ──────────────────────────────────────────────────────────────
# 4.  ZeroGPU-registered inference function
#     @spaces.GPU must be at module level AND wired to a Gradio
#     event for the ZeroGPU runtime to detect it at startup.
# ──────────────────────────────────────────────────────────────
@spaces.GPU
def _gradio_triage(text: str) -> dict:
    """Full triage pipeline exposed via the Gradio UI (ZeroGPU-compatible)."""
    if not text.strip():
        return {"error": "No text provided"}

    from src.classifier import classify
    from src.escalation import gatekeeper, _get_cache
    from src.reply_drafter import draft_reply

    intent_res = classify(text.strip())
    escalation_decision = gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)
    h = hashlib.sha256(text.strip().encode()).hexdigest()
    esc = _get_cache().get(h, {})
    # Two-pass gating: High-risk escalated queries withhold automated reply
    if escalation_decision.should_escalate:
        draft_reply_text = "[ESCALATED TO HUMAN AGENT] Automated reply withheld for safety review. Case routed to Tier-2 support team."
        policy_adherence_text = f"Safety Gate Triggered: {escalation_decision.reason}. High-risk cases require human specialist intervention."
    else:
        reply_res = draft_reply(text)
        draft_reply_text = reply_res.draft_reply
        policy_adherence_text = reply_res.policy_adherence_check

    return {
        "intent":            intent_res.intent.value,
        "confidence":        round(intent_res.confidence, 3),
        "reasoning":         intent_res.reasoning,
        "escalation_score":  round(float(esc.get("escalation_score", 1.0)), 3) if "escalation_score" in esc else 1.0,
        "should_escalate":   escalation_decision.should_escalate,
        "escalation_reason": escalation_decision.reason,
        "draft_reply":       draft_reply_text,
        "policy_adherence":  policy_adherence_text,
    }


# ──────────────────────────────────────────────────────────────
# 5.  Gradio demo  (required as the HF Spaces entrypoint)
#     Gradio's launch() triggers our patched create_app.
#     The React UI takes over / — this panel lives at /gradio.
# ──────────────────────────────────────────────────────────────
with gr.Blocks(title="Sentinel-AI") as demo:
    gr.Markdown(
        "## 🛡️ Sentinel-AI\n"
        "**The React frontend is served at the root URL `/`.**  \n"
        "Use this panel to test the triage API directly."
    )
    with gr.Row():
        inp = gr.Textbox(
            label="Customer message",
            placeholder="Paste a customer message…",
            lines=4,
            scale=2,
        )
        out = gr.JSON(label="Triage result", scale=3)
    gr.Button("Run Triage", variant="primary").click(
        fn=_gradio_triage,
        inputs=inp,
        outputs=out,
    )

# ──────────────────────────────────────────────────────────────
# 6.  Launch  (ssr_mode=False avoids the Node.js error in Gradio 5)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server_host = os.getenv("HOST", os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
    server_port = int(os.getenv("PORT", "7860"))

    # Startup Warmup: Pre-load the embedding model and FAISS vector index
    try:
        print("[Startup] Pre-warming embedding model and FAISS index...")
        from src.retriever import retrieve
        _ = retrieve("warmup query", k=1)
        print("[Startup] ✓ FAISS vector index and SentenceTransformer ready.")
    except Exception as e:
        print(f"[Startup Warning] Warmup deferred: {e}")

    demo.launch(server_name=server_host, server_port=server_port, ssr_mode=False)
