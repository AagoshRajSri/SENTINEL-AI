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

import os, sys, csv, hashlib, traceback, mimetypes
from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
DIST = os.path.join(ROOT, "frontend", "dist")

# ──────────────────────────────────────────────────────────────
# 1.  Imports
# ──────────────────────────────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, HTMLResponse
from fastapi.responses import JSONResponse
import gradio as gr
import spaces

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

    # ── API: POST /api/triage ──────────────────────────────
    @app.post("/api/triage")
    async def _api_triage(request: Request):
        try:
            body = await request.json()
            text = body.get("text", "").strip()
            if not text:
                return JSONResponse({"error": "text is required"}, status_code=400)

            from src.classifier import classify
            from src.escalation import gatekeeper, _get_cache
            from src.reply_drafter import draft_reply

            intent_res = classify(text)
            escalation_decision = gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)

            h = hashlib.sha256(text.encode()).hexdigest()
            esc = _get_cache().get(h, {})
            reply_res = draft_reply(text)

            return JSONResponse({
                "intent":            intent_res.intent.value,
                "confidence":        round(intent_res.confidence, 3),
                "reasoning":         intent_res.reasoning,
                "escalation_score":  round(float(esc.get("escalation_score", 1.0)), 3) if "escalation_score" in esc else 1.0,
                "should_escalate":   escalation_decision.should_escalate,
                "escalation_reason": escalation_decision.reason,
                "draft_reply":       reply_res.draft_reply,
                "policy_adherence":  reply_res.policy_adherence_check,
            })
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

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
    reply_res = draft_reply(text)

    return {
        "intent":            intent_res.intent.value,
        "confidence":        round(intent_res.confidence, 3),
        "reasoning":         intent_res.reasoning,
        "escalation_score":  round(float(esc.get("escalation_score", 1.0)), 3) if "escalation_score" in esc else 1.0,
        "should_escalate":   escalation_decision.should_escalate,
        "escalation_reason": escalation_decision.reason,
        "draft_reply":       reply_res.draft_reply,
        "policy_adherence":  reply_res.policy_adherence_check,
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
demo.launch(server_port=7860, ssr_mode=False)
