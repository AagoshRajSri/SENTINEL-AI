# app.py — Sentinel-AI Web Server
# Serves the single-page frontend + backend API endpoints
# Run with: python app.py

import os
import csv
import json
import sys
import traceback
import hashlib

from dotenv import load_dotenv
load_dotenv()

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
import uvicorn

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.classifier import classify
from src.escalation import gatekeeper, _get_cache
from src.reply_drafter import draft_reply


async def triage(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        text = body.get('text', '').strip()
        if not text:
            return JSONResponse({'error': 'text field is required'}, status_code=400)

        intent_res = classify(text)
        gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)

        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        cache = _get_cache()
        escalation_score = cache.get(text_hash, {}).get('escalation_score', 1.0)
        should_escalate  = cache.get(text_hash, {}).get('should_escalate', True)
        escalation_reason = cache.get(text_hash, {}).get('reason', 'Unknown')

        reply_res = draft_reply(text)

        return JSONResponse({
            'intent':            intent_res.intent.value,
            'confidence':        round(intent_res.confidence, 3),
            'reasoning':         intent_res.reasoning,
            'escalation_score':  round(float(escalation_score), 3),
            'should_escalate':   bool(should_escalate),
            'escalation_reason': escalation_reason,
            'draft_reply':       reply_res.draft_reply,
            'policy_adherence':  reply_res.policy_adherence_check,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)


async def stats(request: Request) -> JSONResponse:
    csv_path = os.path.join(ROOT, 'eval', 'results.csv')
    if not os.path.exists(csv_path):
        return JSONResponse({'error': 'results.csv not found'}, status_code=404)

    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    best = None
    for row in rows:
        if row['FNR'] < 0.40:
            if best is None or row['Precision'] > best['Precision']:
                best = row

    return JSONResponse({
        'rows': rows,
        'best_threshold': best,
        'macro_f1': rows[0]['Macro-F1'] if rows else None,
        'total_samples': int(rows[0]['TP'] + rows[0]['FN'] + rows[0]['FP'] + rows[0]['TN']) if rows else 0,
    })


async def homepage(request: Request) -> HTMLResponse:
    index = os.path.join(ROOT, 'frontend', 'dist', 'index.html')
    with open(index, encoding='utf-8') as f:
        return HTMLResponse(f.read())


routes = [
    Route('/', homepage),
    Route('/api/triage', triage, methods=['POST']),
    Route('/api/stats',  stats,  methods=['GET']),
    Mount('/assets', StaticFiles(directory=os.path.join(ROOT, 'frontend', 'dist', 'assets')), name='assets'),
]

app = Starlette(routes=routes)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

if __name__ == '__main__':
    print('=' * 60)
    print('  Sentinel-AI Web Server')
    print('  Open: http://127.0.0.1:8000')
    print('=' * 60)
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=False)

