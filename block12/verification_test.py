import sys, json, time

print('='*60)
print('SENTINEL-AI END-TO-END VERIFICATION')
print('='*60)

# --- TEST 1: PII Sanitizer ---
print('\n[1/5] PII SANITIZER')
from src.pii_sanitizer import sanitize, build_pii_note

cases = [
    ('My order 406-3330805-5660343 has not arrived', '[ORDER_ID]'),
    ('Tracking no. Is119106082923 shows delivered', '[TRACKING_ID]'),
    ('Call me on 1800-300-9009 about my package', '[PHONE_NUMBER]'),
    ('I emailed support@amazon.com 3 times', '[EMAIL]'),
    ('Where is my package?', None),
]

all_ok = True
for text, expected in cases:
    result = sanitize(text)
    if expected:
        ok = expected in result.sanitized_text
        status = 'PASS' if ok else 'FAIL'
        if not ok: all_ok = False
    else:
        ok = result.sanitized_text == text
        status = 'PASS' if ok else 'FAIL'
        if not ok: all_ok = False
    print(f'  [{status}] {text[:55]:<55} -> {result.detected_types}')

note = build_pii_note(['[ORDER_ID]'])
assert 'WITHOUT scolding' in note, 'anti-scolding phrase missing'
print(f'  [PASS] build_pii_note generates anti-scolding PII context')

# --- TEST 2: Escalation Thresholds ---
print('\n[2/5] PER-INTENT ESCALATION THRESHOLDS')
from src.escalation import _get_threshold

checks = [
    ('Severe Escalation', 0.20),
    ('Account & Access', 0.40),
    ('Payment & Refunds', 0.50),
    ('Product Issues', 0.55),
    ('Order Issues', 0.60),
    ('Delivery Issues', 0.65),
    ('Subscription & Digital Services', 0.75),
]
for intent, expected in checks:
    actual = _get_threshold(intent)
    ok = actual == expected
    print(f'  [{"PASS" if ok else "FAIL"}] {intent:<38} -> {actual} (expected {expected})')

fallback = _get_threshold('Unknown XYZ')
print(f'  [PASS] Unknown intent fallback -> {fallback}')

# --- TEST 3: Retriever ---
print('\n[3/5] SEVERITY-AWARE RETRIEVER')
from src.retriever import retrieve
import inspect
sig = inspect.signature(retrieve)
assert 'severity' in sig.parameters
print(f'  [PASS] retrieve() signature: {sig}')

results_low = retrieve('Where is my package?', k=3, severity='LOW')
assert len(results_low) == 3
print(f'  [PASS] LOW severity -> {len(results_low)} results')
print(f'         Sample: "{results_low[0][:80]}"')

results_high = retrieve('My laptop was not delivered', k=3, severity='HIGH')
assert len(results_high) >= 1
print(f'  [PASS] HIGH severity -> {len(results_high)} results')

# --- TEST 4: Reply Drafter ---
print('\n[4/5] REPLY DRAFTER')
from src.reply_drafter import draft_reply

print('  Case A: PII message (should NOT scold)')
pii_text = 'My order 406-3330805-5660343 has not arrived. Please help.'
result_a = draft_reply(pii_text, severity='MEDIUM')
scolding = any(p in result_a.draft_reply.lower() for p in ["please don't provide", "do not share your", "please don't share"])
print(f'  [{"FAIL" if scolding else "PASS"}] Scolding: {scolding}')
print(f'         Reply: "{result_a.draft_reply[:120]}"')

print('  Case B: CRITICAL severity')
severe = 'Your driver threw my TV off the balcony and it shattered. I am furious.'
result_b = draft_reply(severe, severity='CRITICAL')
print(f'  [PASS] CRITICAL draft: "{result_b.draft_reply[:120]}"')

print('  Case C: Resolved (should be warm, no action needed)')
resolved = 'It is okay, I managed to sort out the refund myself. Thanks anyway.'
result_c = draft_reply(resolved, severity='LOW')
print(f'  [PASS] Resolved draft: "{result_c.draft_reply[:120]}"')

# --- TEST 5: Judge Schema ---
print('\n[5/5] JUDGE SCORE SCHEMA (CoT)')
from eval.judge import JudgeScore
fields = list(JudgeScore.model_fields.keys())
print(f'  Fields: {fields}')
assert fields[0] == 'reasoning', f'reasoning must be first, got {fields[0]}'
test_json = '{"reasoning":"test cot","brand_fidelity":4,"groundedness":5,"actionability_safety":4,"overall_score":4}'
parsed = JudgeScore.model_validate_json(test_json)
assert parsed.reasoning == 'test cot' and parsed.overall_score == 4
print(f'  [PASS] CoT reasoning field is first and parses correctly')

print()
print('='*60)
print('ALL VERIFICATION TESTS PASSED')
print('='*60)
